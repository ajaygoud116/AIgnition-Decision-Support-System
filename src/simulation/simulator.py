from typing import Dict, List

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
    """Simulates budget adjustments against a ForecastResult.

    For each scenario, applies budget adjustments per campaign, models
    revenue impact via a diminishing-returns-adjusted ROAS, and returns
    projected totals.
    """

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("simulation.simulator")
        self._max_adjustments = config.get("simulation.max_adjustments_per_run", 10)
        self._diminishing_periods = config.get("decision.diminishing_returns_periods", 3)
        self._random_seed = config.get("project.random_seed", 42)

    def simulate(
        self,
        scenarios: List[SimulationScenario],
        forecast_result: ForecastResult,
        baselines: Dict[str, CampaignBaseline],
    ) -> List[SimulationResult]:
        results: List[SimulationResult] = []
        for scenario in scenarios:
            results.append(self._simulate_one(scenario, forecast_result, baselines))
        return results

    def _simulate_one(
        self,
        scenario: SimulationScenario,
        forecast_result: ForecastResult,
        baselines: Dict[str, CampaignBaseline],
    ) -> SimulationResult:
        if len(scenario.adjustments) > self._max_adjustments:
            self._logger.warning(
                "too_many_adjustments",
                count=len(scenario.adjustments),
                scenario=scenario.label,
            )

        adj_deltas: Dict[str, float] = {}
        for adj in scenario.adjustments:
            delta = self._spend_delta(adj, baselines)
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
                eff = self._efficiency(actual_delta, bl.total_spend)
                delta_rev = actual_delta * bl.historical_roas * eff
            else:
                delta_rev = 0.0

            total_revenue += base_rev + delta_rev
            total_spend += new_spend

        for series in forecast_result.series:
            eid = series.entity_id
            if eid not in baselines:
                if series.metric == MetricType.REVENUE:
                    total_revenue += sum(p.values.p50 for p in series.points)

        projected_roas = total_revenue / total_spend if total_spend > 0 else 0.0

        return SimulationResult(
            scenario=scenario,
            projected_revenue=total_revenue,
            projected_spend=total_spend,
            projected_roas=projected_roas,
        )

    def _spend_delta(
        self,
        adj: BudgetAdjustment,
        baselines: Dict[str, CampaignBaseline],
    ) -> float:
        if adj.absolute is not None:
            return adj.absolute
        if adj.relative is not None:
            bl = baselines.get(adj.entity_id)
            base = bl.total_spend if bl else 0.0
            return base * adj.relative
        return 0.0

    def _efficiency(self, delta_spend: float, baseline_spend: float) -> float:
        if baseline_spend <= 0:
            return 0.0
        denom = 1.0 + abs(delta_spend) / (
            self._diminishing_periods * baseline_spend + 1e-9
        )
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
