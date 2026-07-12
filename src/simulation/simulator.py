from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.forecasting.forecaster import Forecaster
from src.models.common import (
    BudgetAdjustment,
    ForecastResult,
    MetricType,
    SimulationResult,
    SimulationScenario,
)
from src.simulation.baselines import CampaignBaseline
from src.utils.config import Config
from src.utils.logger import StructuredLogger


class ScenarioSimulator:
    """Budget-conditioned scenario simulator.

    For each scenario, applies budget adjustments, rebuilds the future
    feature matrix with the modified budget (via Forecaster.predict_scenario),
    and aggregates spend/revenue from the model-based forecast.

    This replaces the previous post-hoc ROAS × Efficiency formula with a
    true model-in-the-loop simulation.
    """

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("simulation.simulator")
        self._max_adjustments = config.get("simulation.max_adjustments_per_run", 1000)
        self._random_seed = config.get("project.random_seed", 42)

    def simulate(
        self,
        scenarios: List[SimulationScenario],
        forecast_result: ForecastResult,
        baselines: Dict[str, CampaignBaseline],
        forecaster: Optional[Forecaster] = None,
        feature_df: Optional[pd.DataFrame] = None,
    ) -> List[SimulationResult]:
        """Run budget-conditioned simulations.

        Parameters
        ----------
        scenarios : list of SimulationScenario
            Each scenario defines a set of BudgetAdjustments.
        forecast_result : ForecastResult
            Baseline forecast (used only if forecaster/feature_df not provided).
        baselines : dict of CampaignBaseline
            Historical baselines per campaign (for spend computation).
        forecaster : Forecaster, optional
            Fitted Forecaster instance.  Required for model-based simulation.
        feature_df : pd.DataFrame, optional
            Full feature DataFrame used for training.  Required for model-based
            simulation.

        Returns
        -------
        list of SimulationResult
        """
        if forecaster is not None and feature_df is not None:
            return self._simulate_model_based(
                scenarios, forecast_result, baselines, forecaster, feature_df,
            )
        return self._simulate_formula_based(scenarios, forecast_result, baselines)

    def _simulate_model_based(
        self,
        scenarios: List[SimulationScenario],
        forecast_result: ForecastResult,
        baselines: Dict[str, CampaignBaseline],
        forecaster: Forecaster,
        feature_df: pd.DataFrame,
    ) -> List[SimulationResult]:
        results: List[SimulationResult] = []
        for scenario in scenarios:
            self._logger.info("scenario_start", label=scenario.label)
            budget_map = self._build_budget_map(scenario, baselines, feature_df)
            scenario_forecast = forecaster.predict_scenario(feature_df, budget_map)
            proj_rev, proj_spend = self._aggregate_scenario(
                scenario_forecast, budget_map, baselines,
            )
            proj_roas = proj_rev / proj_spend if proj_spend > 0 else 0.0
            results.append(SimulationResult(
                scenario=scenario,
                projected_revenue=proj_rev,
                projected_spend=proj_spend,
                projected_roas=proj_roas,
            ))
            self._logger.info("scenario_done", label=scenario.label, spend=proj_spend, rev=proj_rev)
        return results

    def _simulate_formula_based(
        self,
        scenarios: List[SimulationScenario],
        forecast_result: ForecastResult,
        baselines: Dict[str, CampaignBaseline],
    ) -> List[SimulationResult]:
        """Fallback post-hoc formula when forecaster/feature_df not available."""
        self._logger.warning("simulation_fallback_formula")
        results: List[SimulationResult] = []
        for scenario in scenarios:
            results.append(self._formula_one(scenario, forecast_result, baselines))
        return results

    def _formula_one(
        self,
        scenario: SimulationScenario,
        forecast_result: ForecastResult,
        baselines: Dict[str, CampaignBaseline],
    ) -> SimulationResult:
        """Original post-hoc ROAS × Efficiency × Spend formula."""
        adj_deltas: Dict[str, float] = {}
        for adj in scenario.adjustments:
            delta = self._spend_delta_formula(adj, baselines)
            adj_deltas[adj.entity_id] = adj_deltas.get(adj.entity_id, 0.0) + delta

        total_revenue = 0.0
        total_spend = 0.0
        campaign_forecast = self._campaign_revenue_map(forecast_result)

        for eid, bl in baselines.items():
            delta_spend = adj_deltas.get(eid, 0.0)
            new_spend = max(0.0, bl.total_spend + delta_spend)
            actual_delta = new_spend - bl.total_spend
            base_rev = campaign_forecast.get(eid, bl.total_revenue)
            if eid in adj_deltas:
                # Diminishing returns efficiency factor
                denom = 1.0 + abs(actual_delta) / (3.0 * bl.total_spend + 1e-9)
                eff = 1.0 / denom
                delta_rev = actual_delta * bl.historical_roas * eff
            else:
                delta_rev = 0.0
            total_revenue += base_rev + delta_rev
            total_spend += new_spend

        for series in forecast_result.series:
            eid = series.entity_id
            if eid not in baselines and series.metric == MetricType.REVENUE:
                total_revenue += sum(p.values.p50 for p in series.points)

        roas = total_revenue / total_spend if total_spend > 0 else 0.0
        return SimulationResult(
            scenario=scenario, projected_revenue=total_revenue,
            projected_spend=total_spend, projected_roas=roas,
        )

    def _build_budget_map(
        self,
        scenario: SimulationScenario,
        baselines: Dict[str, CampaignBaseline],
        feature_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """Build {campaign_id: new_daily_budget} from scenario adjustments."""
        budget_map: Dict[str, float] = {}
        for adj in scenario.adjustments:
            bl = baselines.get(adj.entity_id)
            if bl is None:
                continue
            last_budget = bl.last_daily_budget
            if adj.absolute is not None:
                new_budget = max(0.0, last_budget + adj.absolute)
            elif adj.relative is not None:
                new_budget = max(0.0, last_budget * (1.0 + adj.relative))
            else:
                new_budget = last_budget
            budget_map[adj.entity_id] = round(new_budget, 2)

        # Include all campaigns with no adjustment at baseline budget
        for cid in baselines:
            if cid not in budget_map:
                budget_map[cid] = round(baselines[cid].last_daily_budget, 2)

        return budget_map

    def _aggregate_scenario(
        self,
        scenario_forecast: ForecastResult,
        budget_map: Dict[str, float],
        baselines: Dict[str, CampaignBaseline],
    ) -> tuple:
        """Aggregate total revenue and spend from a scenario forecast."""
        total_revenue = 0.0
        for series in scenario_forecast.series:
            if series.metric == MetricType.REVENUE:
                total_revenue += sum(p.values.p50 for p in series.points)

        # Projected spend: sum of scenario daily budgets × number of forecast days
        budget_days: Dict[str, int] = {}
        for series in scenario_forecast.series:
            eid = series.entity_id
            budget_days[eid] = max(budget_days.get(eid, 0), len(series.points))

        total_spend = 0.0
        for eid, days in budget_days.items():
            daily_budget = budget_map.get(eid, baselines.get(eid, CampaignBaseline(
                entity_id=eid, channel="", campaign_type="",
                total_spend=0, total_revenue=0,
                last_daily_spend=0, last_daily_budget=0, historical_roas=0,
            )).last_daily_budget)
            total_spend += daily_budget * days

        return total_revenue, total_spend

    def _spend_delta_formula(
        self, adj: BudgetAdjustment, baselines: Dict[str, CampaignBaseline],
    ) -> float:
        """Compute spend delta for formula-based fallback."""
        if adj.absolute is not None:
            return adj.absolute
        if adj.relative is not None:
            bl = baselines.get(adj.entity_id)
            base = bl.total_spend if bl else 0.0
            return base * adj.relative
        return 0.0

    # ── Backward-compatible helpers (used by formula fallback and tests) ─────
    def _spend_delta(
        self, adj: BudgetAdjustment, baselines: Dict[str, CampaignBaseline],
    ) -> float:
        """Compute spend delta (alias for _spend_delta_formula)."""
        return self._spend_delta_formula(adj, baselines)

    @staticmethod
    def _efficiency(delta_spend: float, baseline_spend: float) -> float:
        """Diminishing-returns efficiency factor used by formula fallback."""
        if baseline_spend <= 0:
            return 0.0
        denom = 1.0 + abs(delta_spend) / (3.0 * baseline_spend + 1e-9)
        return 1.0 / denom

    @staticmethod
    def _campaign_revenue_map(forecast_result: ForecastResult) -> Dict[str, float]:
        rev_map: Dict[str, float] = {}
        for series in forecast_result.series:
            if series.metric != MetricType.REVENUE:
                continue
            total = rev_map.get(series.entity_id, 0.0)
            total += sum(p.values.p50 for p in series.points)
            rev_map[series.entity_id] = total
        return rev_map
