import json
from pathlib import Path

import pandas as pd
import pytest

from src.decision.models import CampaignAssessment, OptimizationReport
from src.models.common import (
    AllocationRecommendation,
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
from src.report.generator import ReportGenerator
from src.uncertainty.models import ChannelUncertainty, EntityUncertainty, UncertaintyReport
from src.utils.config import Config


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def generator(config):
    return ReportGenerator(config)


@pytest.fixture
def forecast_result():
    points = [
        ForecastPoint(
            date=type("d", (), {"date": f"2025-01-{i+1}"})(),
            values=QuantileValue(p10=90.0, p50=100.0, p90=110.0),
        )
        for i in range(3)
    ]
    series = [
        ForecastSeries(
            entity_id="camp_a", channel="google",
            granularity=Granularity.CAMPAIGN, metric=MetricType.REVENUE,
            horizon=Horizon.D30, points=points,
        ),
    ]
    return ForecastResult(series=series)


@pytest.fixture
def uncertainty_report():
    entities = [
        EntityUncertainty(
            entity_id="camp_a", channel="google",
            avg_interval_width=20.0, avg_relative_width=0.2,
            confidence_score=0.85, volatility=0.3,
            stability_trend="stable",
        ),
    ]
    return UncertaintyReport(
        entities=entities,
        channels=[
            ChannelUncertainty(
                channel="google", avg_confidence=0.85,
                avg_volatility=0.3, campaign_count=1,
            ),
        ],
        overall_confidence=0.85,
        overall_volatility=0.3,
        high_uncertainty_count=0,
    )


@pytest.fixture
def simulation_results():
    return [
        SimulationResult(
            scenario=SimulationScenario(label="baseline", adjustments=[]),
            projected_revenue=10000.0,
            projected_spend=3000.0,
            projected_roas=3.33,
        ),
    ]


@pytest.fixture
def optimization_report():
    recommendations = [
        AllocationRecommendation(
            entity_id="camp_a", channel="google",
            current_budget=5000.0, recommended_budget=5500.0,
            rationale="Increase budget 10%. Campaign performing well.",
            expected_roas=4.0,
        ),
    ]
    assessments = [
        CampaignAssessment(
            entity_id="camp_a", channel="google", campaign_type="SEARCH",
            current_spend=5000.0, current_roas=4.0,
            confidence_score=0.85, volatility=0.3,
            stability_trend="stable", flags=[], score=2.0,
        ),
    ]
    return OptimizationReport(
        assessments=assessments,
        recommendations=recommendations,
        summary={
            "total_current_budget": 5000.0,
            "total_recommended_budget": 5500.0,
            "campaigns_assessed": 1,
            "campaigns_flagged": 0,
            "total_flags": 0,
            "recommendations_generated": 1,
        },
    )


class TestReportGenerator:
    def test_generate_creates_csv_files(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        assert (tmp_path / "forecasts.csv").exists()
        assert (tmp_path / "uncertainty.csv").exists()
        assert (tmp_path / "simulations.csv").exists()
        assert (tmp_path / "recommendations.csv").exists()

    def test_summary_json_exists(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        assert (tmp_path / "summary.json").exists()

    def test_forecasts_csv_has_expected_columns(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        df = pd.read_csv(tmp_path / "forecasts.csv")
        expected = {"entity_id", "channel", "horizon", "metric", "date", "p10", "p50", "p90"}
        assert expected.issubset(set(df.columns))

    def test_uncertainty_csv_has_expected_columns(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        df = pd.read_csv(tmp_path / "uncertainty.csv")
        expected = {"entity_id", "channel", "confidence_score", "volatility", "stability_trend"}
        assert expected.issubset(set(df.columns))

    def test_simulations_csv_has_expected_columns(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        df = pd.read_csv(tmp_path / "simulations.csv")
        expected = {"scenario", "projected_revenue", "projected_spend", "projected_roas"}
        assert expected.issubset(set(df.columns))

    def test_recommendations_csv_has_expected_columns(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        df = pd.read_csv(tmp_path / "recommendations.csv")
        expected = {"entity_id", "channel", "current_budget", "recommended_budget", "rationale"}
        assert expected.issubset(set(df.columns))

    def test_summary_json_has_expected_keys(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        with open(tmp_path / "summary.json") as f:
            data = json.load(f)
        expected = {"campaigns_forecasted", "total_forecast_revenue_p50", "overall_uncertainty_confidence",
                    "scenarios_simulated", "recommendations"}
        assert expected.issubset(set(data.keys()))

    def test_forecast_values_rounded_to_decimal_places(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        df = pd.read_csv(tmp_path / "forecasts.csv")
        for col in ["p10", "p50", "p90"]:
            assert all(df[col] == df[col].round(2))

    def test_single_campaign_in_forecast(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        df = pd.read_csv(tmp_path / "forecasts.csv")
        assert "camp_a" in df["entity_id"].values

    def test_simulation_values(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        df = pd.read_csv(tmp_path / "simulations.csv")
        row = df.iloc[0]
        assert row["scenario"] == "baseline"
        assert row["projected_revenue"] == 10000.0

    def test_empty_forecast_does_not_crash(self, generator, uncertainty_report, simulation_results, optimization_report, tmp_path):
        empty_forecast = ForecastResult(series=[])
        generator.generate(empty_forecast, uncertainty_report, simulation_results, optimization_report, tmp_path)
        df = pd.read_csv(tmp_path / "forecasts.csv")
        assert len(df) == 0

    def test_empty_recommendations_does_not_crash(self, generator, forecast_result, uncertainty_report, simulation_results, tmp_path):
        empty_opt = OptimizationReport(summary={"campaigns_flagged": 0, "total_flags": 0})
        generator.generate(forecast_result, uncertainty_report, simulation_results, empty_opt, tmp_path)
        df = pd.read_csv(tmp_path / "recommendations.csv")
        assert len(df) == 0

    def test_recommendation_budget_change_column(self, generator, forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path):
        generator.generate(forecast_result, uncertainty_report, simulation_results, optimization_report, tmp_path)
        df = pd.read_csv(tmp_path / "recommendations.csv")
        assert "budget_change" in df.columns
        assert df.iloc[0]["budget_change"] == 500.0
