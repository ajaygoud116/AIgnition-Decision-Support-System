import pandas as pd
import pytest

from src.decision.engine import DecisionEngine
from src.decision.models import OptimizationReport
from src.models.common import ForecastPoint, ForecastResult, ForecastSeries, Granularity, Horizon, MetricType, QuantileValue
from src.simulation.baselines import CampaignBaseline
from src.uncertainty.models import EntityUncertainty, UncertaintyReport
from src.utils.config import Config


def make_bl(eid, ch, spend=500.0, rev=2000.0):
    roas = rev / spend if spend > 0 else 0.0
    return CampaignBaseline(
        entity_id=eid, channel=ch, campaign_type="SEARCH",
        total_spend=spend, total_revenue=rev,
        last_daily_spend=50.0, last_daily_budget=100.0,
        historical_roas=roas,
    )


def make_unc(eid, ch, conf=0.8, vol=0.3):
    return EntityUncertainty(
        entity_id=eid, channel=ch,
        avg_interval_width=10.0, avg_relative_width=0.5,
        confidence_score=conf, volatility=vol,
        stability_trend="stable",
    )


def make_forecast(eid, ch, p50=100.0):
    points = [
        ForecastPoint(
            date=type("d", (), {"date": "2025-01-01"})(),
            values=QuantileValue(p10=p50 * 0.9, p50=p50, p90=p50 * 1.1),
        )
    ]
    return ForecastResult(series=[
        ForecastSeries(
            entity_id=eid, channel=ch,
            granularity=Granularity.CAMPAIGN, metric=MetricType.REVENUE,
            horizon=Horizon.D30, points=points * 30,
        )
    ])


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def engine(config):
    return DecisionEngine(config)


class TestDecisionEngine:
    def test_analyze_returns_optimization_report(self, engine):
        baselines = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google")])
        forecast = make_forecast("a", "google")
        report = engine.analyze(forecast, unc, baselines)
        assert isinstance(report, OptimizationReport)

    def test_has_assessments(self, engine):
        baselines = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google")])
        forecast = make_forecast("a", "google")
        report = engine.analyze(forecast, unc, baselines)
        assert len(report.assessments) == 1

    def test_has_recommendations(self, engine):
        baselines = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google")])
        forecast = make_forecast("a", "google")
        report = engine.analyze(forecast, unc, baselines)
        assert len(report.recommendations) >= 1

    def test_summary_has_total_current_budget(self, engine):
        baselines = {"a": make_bl("a", "google", spend=100.0), "b": make_bl("b", "meta", spend=200.0)}
        unc = UncertaintyReport(entities=[make_unc("a", "google"), make_unc("b", "meta")])
        forecast = ForecastResult(series=[])
        report = engine.analyze(forecast, unc, baselines)
        assert report.summary["total_current_budget"] == 300.0

    def test_summary_has_total_recommended_budget(self, engine):
        baselines = {"a": make_bl("a", "google", spend=100.0)}
        unc = UncertaintyReport(entities=[make_unc("a", "google")])
        forecast = make_forecast("a", "google")
        report = engine.analyze(forecast, unc, baselines)
        assert report.summary["total_recommended_budget"] > 0

    def test_summary_has_campaigns_assessed(self, engine):
        baselines = {"a": make_bl("a", "google"), "b": make_bl("b", "meta")}
        unc = UncertaintyReport(entities=[make_unc("a", "google"), make_unc("b", "meta")])
        forecast = ForecastResult(series=[])
        report = engine.analyze(forecast, unc, baselines)
        assert report.summary["campaigns_assessed"] == 2

    def test_summary_tracks_flagged_count(self, engine):
        low_roas_bl = make_bl("low", "google", spend=100.0, rev=50.0)
        good_bl = make_bl("good", "meta", spend=100.0, rev=500.0)
        baselines = {"low": low_roas_bl, "good": good_bl}
        unc = UncertaintyReport(entities=[make_unc("low", "google", conf=0.3), make_unc("good", "meta", conf=0.9)])
        forecast = ForecastResult(series=[])
        report = engine.analyze(forecast, unc, baselines)
        assert report.summary["campaigns_flagged"] >= 1

    def test_with_feature_df_does_not_crash(self, engine):
        baselines = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google")])
        forecast = make_forecast("a", "google")
        df = pd.DataFrame({
            "date": [pd.Timestamp("2025-01-01")],
            "campaign_id": ["a"],
            "revenue": [100.0],
            "spend": [50.0],
        })
        report = engine.analyze(forecast, unc, baselines, feature_df=df)
        assert isinstance(report, OptimizationReport)

    def test_empty_baselines_returns_empty_report(self, engine):
        report = engine.analyze(ForecastResult(), UncertaintyReport(), {})
        assert report.assessments == []
        assert report.recommendations == []
        assert report.summary["campaigns_assessed"] == 0

    def test_recommendation_current_budget_matches_baseline(self, engine):
        baselines = {"a": make_bl("a", "google", spend=750.0)}
        unc = UncertaintyReport(entities=[make_unc("a", "google")])
        forecast = make_forecast("a", "google")
        report = engine.analyze(forecast, unc, baselines)
        rec = report.recommendations[0]
        assert rec.current_budget == 750.0

    def test_recommendation_channel_matches(self, engine):
        baselines = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google")])
        forecast = make_forecast("a", "google")
        report = engine.analyze(forecast, unc, baselines)
        assert report.recommendations[0].channel == "google"

    def test_budget_change_reflected_in_summary(self, engine):
        baselines = {"a": make_bl("a", "google", spend=500.0)}
        unc = UncertaintyReport(entities=[make_unc("a", "google", conf=0.9, vol=0.1)])
        forecast = make_forecast("a", "google")
        report = engine.analyze(forecast, unc, baselines)
        assert report.summary["budget_change"] == pytest.approx(0.0, abs=1.0)
