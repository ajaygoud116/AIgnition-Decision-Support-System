import numpy as np
import pandas as pd
import pytest

from src.models.common import (
    BudgetAdjustment,
    ForecastPoint,
    ForecastResult,
    ForecastSeries,
    Granularity,
    Horizon,
    MetricType,
    QuantileValue,
    SimulationResult,
    SimulationScenario,
)
from src.simulation.baselines import CampaignBaseline, extract_baselines
from src.simulation.simulator import ScenarioSimulator
from src.utils.config import Config


def make_baseline(eid, channel, total_spend=500.0, total_revenue=2000.0,
                  last_spend=50.0, last_budget=100.0, roas=None):
    if roas is None:
        roas = total_revenue / total_spend if total_spend > 0 else 0.0
    return CampaignBaseline(
        entity_id=eid,
        channel=channel,
        campaign_type="SEARCH",
        total_spend=total_spend,
        total_revenue=total_revenue,
        last_daily_spend=last_spend,
        last_daily_budget=last_budget,
        historical_roas=roas,
    )


def make_series(eid, channel, horizon=Horizon.D30, p50_base=100.0):
    h = horizon.value
    points = [
        ForecastPoint(
            date=type("d", (), {"date": f"2025-01-{i+1}"})(),
            values=QuantileValue(
                p10=float(p50_base * 0.9 + i),
                p50=float(p50_base + i),
                p90=float(p50_base * 1.1 + i),
            ),
        )
        for i in range(h)
    ]
    return ForecastSeries(
        entity_id=eid,
        channel=channel,
        granularity=Granularity.CAMPAIGN,
        metric=MetricType.REVENUE,
        horizon=horizon,
        points=points,
    )


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def simulator(config):
    return ScenarioSimulator(config)


@pytest.fixture
def single_campaign_baselines():
    return {"camp_a": make_baseline("camp_a", "google")}


@pytest.fixture
def single_forecast():
    series = [make_series("camp_a", "google", Horizon.D30, p50_base=100.0)]
    return ForecastResult(series=series)


@pytest.fixture
def multi_campaign_baselines():
    return {
        "camp_a": make_baseline("camp_a", "google", total_spend=500.0, total_revenue=2000.0),
        "camp_b": make_baseline("camp_b", "meta", total_spend=300.0, total_revenue=900.0, roas=3.0),
    }


@pytest.fixture
def multi_forecast():
    series = [
        make_series("camp_a", "google", Horizon.D30, p50_base=100.0),
        make_series("camp_b", "meta", Horizon.D30, p50_base=80.0),
    ]
    return ForecastResult(series=series)


class TestScenarioSimulator:
    def test_simulate_returns_list(self, simulator, single_forecast, single_campaign_baselines):
        scenarios = [SimulationScenario(label="baseline", adjustments=[])]
        results = simulator.simulate(scenarios, single_forecast, single_campaign_baselines)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_result_is_simulation_result(self, simulator, single_forecast, single_campaign_baselines):
        scenarios = [SimulationScenario(label="baseline", adjustments=[])]
        results = simulator.simulate(scenarios, single_forecast, single_campaign_baselines)
        assert isinstance(results[0], SimulationResult)

    def test_result_has_scenario_label(self, simulator, single_forecast, single_campaign_baselines):
        scenarios = [SimulationScenario(label="test_scenario", adjustments=[])]
        results = simulator.simulate(scenarios, single_forecast, single_campaign_baselines)
        assert results[0].scenario.label == "test_scenario"

    def test_baseline_no_adjustments(self, simulator, multi_forecast, multi_campaign_baselines):
        scenarios = [SimulationScenario(label="baseline", adjustments=[])]
        results = simulator.simulate(scenarios, multi_forecast, multi_campaign_baselines)

        expected_spend = 500.0 + 300.0
        seq_sum = 30 * 29 / 2  # 0 + 1 + ... + 29
        expected_revenue = (100.0 * 30 + seq_sum) + (80.0 * 30 + seq_sum)
        assert results[0].projected_spend == pytest.approx(expected_spend)
        assert results[0].projected_revenue == pytest.approx(expected_revenue, rel=1e-3)
        assert results[0].projected_roas > 0

    def test_absolute_budget_increase(self, simulator, single_forecast, single_campaign_baselines):
        adj = BudgetAdjustment(entity_id="camp_a", channel="google", absolute=100.0)
        scenario = SimulationScenario(label="increase", adjustments=[adj])
        results = simulator.simulate([scenario], single_forecast, single_campaign_baselines)
        r = results[0]
        assert r.projected_spend > 500.0
        assert r.projected_revenue > (100.0 * 30)

    def test_absolute_budget_decrease(self, simulator, single_forecast, single_campaign_baselines):
        adj = BudgetAdjustment(entity_id="camp_a", channel="google", absolute=-100.0)
        scenario = SimulationScenario(label="decrease", adjustments=[adj])
        results = simulator.simulate([scenario], single_forecast, single_campaign_baselines)
        r = results[0]
        assert r.projected_spend < 500.0
        seq_sum = 30 * 29 / 2
        base_rev = 100.0 * 30 + seq_sum
        assert r.projected_revenue < base_rev

    def test_relative_budget_increase(self, simulator, single_forecast, single_campaign_baselines):
        adj = BudgetAdjustment(entity_id="camp_a", channel="google", relative=0.1)
        scenario = SimulationScenario(label="increase_10pct", adjustments=[adj])
        results = simulator.simulate([scenario], single_forecast, single_campaign_baselines)
        r = results[0]
        expected_spend = 500.0 + 500.0 * 0.1
        assert r.projected_spend == pytest.approx(expected_spend)

    def test_multiple_adjustments_same_campaign_accumulate(self, simulator, single_forecast, single_campaign_baselines):
        adj1 = BudgetAdjustment(entity_id="camp_a", channel="google", absolute=50.0)
        adj2 = BudgetAdjustment(entity_id="camp_a", channel="google", absolute=30.0)
        scenario = SimulationScenario(label="accumulate", adjustments=[adj1, adj2])
        results = simulator.simulate([scenario], single_forecast, single_campaign_baselines)
        r = results[0]
        assert r.projected_spend == pytest.approx(500.0 + 80.0)

    def test_multiple_campaigns_adjusted(self, simulator, multi_forecast, multi_campaign_baselines):
        adj_a = BudgetAdjustment(entity_id="camp_a", channel="google", absolute=100.0)
        adj_b = BudgetAdjustment(entity_id="camp_b", channel="meta", absolute=-50.0)
        scenario = SimulationScenario(label="mixed", adjustments=[adj_a, adj_b])
        results = simulator.simulate([scenario], multi_forecast, multi_campaign_baselines)
        r = results[0]
        assert r.projected_spend == pytest.approx(600.0 + 250.0)

    def test_unknown_campaign_adjustment_does_not_crash(self, simulator, single_forecast, single_campaign_baselines):
        adj = BudgetAdjustment(entity_id="unknown", channel="google", absolute=100.0)
        scenario = SimulationScenario(label="unknown", adjustments=[adj])
        results = simulator.simulate([scenario], single_forecast, single_campaign_baselines)
        assert len(results) == 1

    def test_multiple_scenarios(self, simulator, single_forecast, single_campaign_baselines):
        s1 = SimulationScenario(label="s1", adjustments=[])
        s2 = SimulationScenario(label="s2", adjustments=[BudgetAdjustment(entity_id="camp_a", channel="google", absolute=50.0)])
        results = simulator.simulate([s1, s2], single_forecast, single_campaign_baselines)
        assert len(results) == 2
        assert results[0].scenario.label == "s1"
        assert results[1].scenario.label == "s2"

    def test_empty_scenarios(self, simulator, single_forecast, single_campaign_baselines):
        results = simulator.simulate([], single_forecast, single_campaign_baselines)
        assert results == []

    def test_roas_positive_with_revenue(self, simulator, single_forecast, single_campaign_baselines):
        scenario = SimulationScenario(label="test", adjustments=[BudgetAdjustment(entity_id="camp_a", channel="google", absolute=50.0)])
        results = simulator.simulate([scenario], single_forecast, single_campaign_baselines)
        assert results[0].projected_roas > 0

    def test_diminishing_returns_large_increase_less_efficient(self, simulator):
        bl = {"camp_a": make_baseline("camp_a", "google", total_spend=100.0, total_revenue=300.0, roas=3.0)}
        forecast = ForecastResult(series=[make_series("camp_a", "google", Horizon.D30, p50_base=10.0)])

        small_adj = BudgetAdjustment(entity_id="camp_a", channel="google", relative=0.1)
        large_adj = BudgetAdjustment(entity_id="camp_a", channel="google", relative=10.0)

        small_result = simulator.simulate(
            [SimulationScenario(label="small", adjustments=[small_adj])], forecast, bl
        )[0]
        large_result = simulator.simulate(
            [SimulationScenario(label="large", adjustments=[large_adj])], forecast, bl
        )[0]

        small_delta_rev = small_result.projected_revenue - (10.0 * 30)
        large_delta_rev = large_result.projected_revenue - (10.0 * 30)
        # Efficiency decreases for larger relative changes
        # small: eff ≈ 1/(1+10/(3*100)) = 1/(1+0.033) ≈ 0.97
        # large: eff ≈ 1/(1+1000/(3*100)) = 1/(1+3.33) ≈ 0.23
        small_eff = small_delta_rev / (10.0 * 3.0)
        large_eff = large_delta_rev / (1000.0 * 3.0)
        assert small_eff > large_eff

    def test_zero_spend_baseline_does_not_crash(self, simulator):
        bl = {"camp_a": make_baseline("camp_a", "google", total_spend=0.0, total_revenue=0.0, roas=0.0)}
        # Build a series where every p50 is exactly 0 (no i offset)
        points = [
            ForecastPoint(
                date=type("d", (), {"date": f"2025-01-{i+1}"})(),
                values=QuantileValue(p10=0.0, p50=0.0, p90=0.0),
            )
            for i in range(30)
        ]
        series = [
            ForecastSeries(
                entity_id="camp_a", channel="google", granularity=Granularity.CAMPAIGN,
                metric=MetricType.REVENUE, horizon=Horizon.D30, points=points,
            )
        ]
        forecast = ForecastResult(series=series)
        adj = BudgetAdjustment(entity_id="camp_a", channel="google", absolute=100.0)
        scenario = SimulationScenario(label="zero_base", adjustments=[adj])
        results = simulator.simulate([scenario], forecast, bl)
        assert results[0].projected_spend == pytest.approx(100.0)
        assert results[0].projected_revenue == 0.0

    def test_forecast_campaign_not_in_baseline(self, simulator):
        bl = {}
        forecast = ForecastResult(series=[make_series("orphan", "google", Horizon.D30, p50_base=100.0)])
        scenario = SimulationScenario(label="no_baseline", adjustments=[])
        results = simulator.simulate([scenario], forecast, bl)
        seq_sum = 30 * 29 / 2
        assert results[0].projected_revenue == pytest.approx(100.0 * 30 + seq_sum)

    def test_efficiency_function(self, simulator):
        # When delta is zero, efficiency = 1.0
        eff_zero = simulator._efficiency(0.0, 100.0)
        assert eff_zero == pytest.approx(1.0)

        # When delta equals baseline * diminishing_periods, efficiency = 0.5
        eff_half = simulator._efficiency(300.0, 100.0)
        assert eff_half == pytest.approx(0.5, rel=0.01)

    def test_spend_delta_absolute(self, simulator, single_campaign_baselines):
        adj = BudgetAdjustment(entity_id="camp_a", channel="google", absolute=200.0)
        delta = simulator._spend_delta(adj, single_campaign_baselines)
        assert delta == 200.0

    def test_spend_delta_relative(self, simulator, single_campaign_baselines):
        adj = BudgetAdjustment(entity_id="camp_a", channel="google", relative=0.5)
        delta = simulator._spend_delta(adj, single_campaign_baselines)
        # total_spend is 500.0, so 50% = 250.0
        assert delta == 250.0

    def test_adjustment_budget_cannot_go_below_zero(self, simulator, single_forecast, single_campaign_baselines):
        adj = BudgetAdjustment(entity_id="camp_a", channel="google", absolute=-1000.0)
        scenario = SimulationScenario(label="excessive_cut", adjustments=[adj])
        results = simulator.simulate([scenario], single_forecast, single_campaign_baselines)
        assert results[0].projected_spend == 0.0
