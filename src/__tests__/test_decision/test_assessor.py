import pandas as pd
import pytest

from src.decision.assessor import CampaignAssessor
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


def make_unc(eid, ch, conf=0.8, vol=0.3, trend="stable"):
    return EntityUncertainty(
        entity_id=eid, channel=ch,
        avg_interval_width=10.0, avg_relative_width=0.5,
        confidence_score=conf, volatility=vol,
        stability_trend=trend,
    )


def make_forecast(eid, ch):
    points = [
        ForecastPoint(
            date=type("d", (), {"date": "2025-01-01"})(),
            values=QuantileValue(p10=90.0, p50=100.0, p90=110.0),
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
def assessor(config):
    return CampaignAssessor(config)


class TestAssessAll:
    def test_returns_one_assessment_per_baseline(self, assessor):
        bl = {"a": make_bl("a", "google"), "b": make_bl("b", "meta")}
        unc = UncertaintyReport(entities=[make_unc("a", "google"), make_unc("b", "meta")])
        forecast = ForecastResult()
        result = assessor.assess_all(bl, unc, forecast)
        assert len(result) == 2

    def test_assessment_has_entity_id(self, assessor):
        bl = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google")])
        result = assessor.assess_all(bl, unc, ForecastResult())
        assert result[0].entity_id == "a"

    def test_assessment_has_channel(self, assessor):
        bl = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google")])
        result = assessor.assess_all(bl, unc, ForecastResult())
        assert result[0].channel == "google"

    def test_below_roas_flag(self, assessor):
        bl = {"a": make_bl("a", "google", rev=100.0, spend=100.0)}  # ROAS=1.0 < 3.0
        unc = UncertaintyReport(entities=[make_unc("a", "google", conf=0.9)])
        result = assessor.assess_all(bl, unc, ForecastResult())
        assert "below_roas_target" in result[0].flags

    def test_no_roas_flag_when_above_target(self, assessor):
        bl = {"a": make_bl("a", "google", rev=400.0, spend=100.0)}  # ROAS=4.0 >= 3.0
        unc = UncertaintyReport(entities=[make_unc("a", "google", conf=0.9)])
        result = assessor.assess_all(bl, unc, ForecastResult())
        assert "below_roas_target" not in result[0].flags

    def test_high_uncertainty_flag(self, assessor):
        bl = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google", conf=0.3)])  # < 0.5
        result = assessor.assess_all(bl, unc, ForecastResult())
        assert "high_uncertainty" in result[0].flags

    def test_no_uncertainty_flag_when_confident(self, assessor):
        bl = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google", conf=0.9)])
        result = assessor.assess_all(bl, unc, ForecastResult())
        assert "high_uncertainty" not in result[0].flags

    def test_zero_revenue_flag(self, assessor):
        bl = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google", conf=0.9)])
        n = assessor.get_zero_revenue_days()
        rows = [{"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                 "campaign_id": "a", "revenue": 0.0, "spend": 10.0}
                for i in range(n + 1)]
        df = pd.DataFrame(rows)
        result = assessor.assess_all(bl, unc, ForecastResult(), feature_df=df)
        assert "zero_revenue" in result[0].flags

    def test_no_zero_revenue_flag_when_revenue_present(self, assessor):
        bl = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google", conf=0.9)])
        n = assessor.get_zero_revenue_days()
        rows = [{"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                 "campaign_id": "a", "revenue": 100.0, "spend": 10.0}
                for i in range(n + 10)]
        df = pd.DataFrame(rows)
        result = assessor.assess_all(bl, unc, ForecastResult(), feature_df=df)
        assert "zero_revenue" not in result[0].flags

    def test_concentration_flag(self, assessor):
        bl = {
            "a": make_bl("a", "google", spend=900.0),
            "b": make_bl("b", "google", spend=100.0),
        }
        unc = UncertaintyReport(entities=[make_unc("a", "google"), make_unc("b", "google")])
        result = assessor.assess_all(bl, unc, ForecastResult())
        assert "concentration_risk" in result[0].flags
        assert "concentration_risk" not in result[1].flags

    def test_score_higher_for_better_campaigns(self, assessor):
        good = make_bl("good", "google", rev=500.0, spend=100.0)  # ROAS=5
        bad = make_bl("bad", "google", rev=100.0, spend=100.0)    # ROAS=1
        bl = {"good": good, "bad": bad}
        unc = UncertaintyReport(entities=[
            make_unc("good", "google", conf=0.9, vol=0.1),
            make_unc("bad", "google", conf=0.3, vol=0.8),
        ])
        forecast = ForecastResult()
        result = assessor.assess_all(bl, unc, forecast)
        scores = {a.entity_id: a.score for a in result}
        assert scores["good"] > scores["bad"]

    def test_empty_baselines(self, assessor):
        result = assessor.assess_all({}, UncertaintyReport(), ForecastResult())
        assert result == []

    def test_no_uncertainty_for_campaign_defaults_half(self, assessor):
        bl = {"a": make_bl("a", "google")}
        result = assessor.assess_all(bl, UncertaintyReport(), ForecastResult())
        assert result[0].confidence_score == 0.5
        assert result[0].volatility == 0.5

    def test_cost_inflation_flag(self, assessor):
        bl = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google", conf=0.9)])
        # Early: spend=10, rev=100 → ratio=0.1
        # Late: spend=30, rev=100 → ratio=0.3
        # Inflation = (0.3 - 0.1) / 0.1 = 2.0 > 0.2
        rows = []
        for i in range(30):
            rows.append({"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                         "campaign_id": "a", "spend": 10.0, "revenue": 100.0})
        for i in range(30, 60):
            rows.append({"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                         "campaign_id": "a", "spend": 30.0, "revenue": 100.0})
        df = pd.DataFrame(rows)
        result = assessor.assess_all(bl, unc, ForecastResult(), feature_df=df)
        assert "cost_inflation" in result[0].flags

    def test_no_cost_inflation_when_ratios_stable(self, assessor):
        bl = {"a": make_bl("a", "google")}
        unc = UncertaintyReport(entities=[make_unc("a", "google", conf=0.9)])
        rows = [{"date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                 "campaign_id": "a", "spend": 10.0, "revenue": 100.0}
                for i in range(60)]
        df = pd.DataFrame(rows)
        result = assessor.assess_all(bl, unc, ForecastResult(), feature_df=df)
        assert "cost_inflation" not in result[0].flags
