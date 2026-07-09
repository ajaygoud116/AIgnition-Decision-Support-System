import pytest

from src.decision.models import CampaignAssessment
from src.decision.optimizer import BudgetOptimizer
from src.models.common import AllocationRecommendation
from src.simulation.baselines import CampaignBaseline
from src.utils.config import Config


def make_bl(eid, ch, spend=500.0, rev=2000.0):
    roas = rev / spend if spend > 0 else 0.0
    return CampaignBaseline(
        entity_id=eid, channel=ch, campaign_type="SEARCH",
        total_spend=spend, total_revenue=rev,
        last_daily_spend=50.0, last_daily_budget=100.0,
        historical_roas=roas,
    )


def make_assessment(eid, ch, spend=500.0, roas=4.0, conf=0.8, vol=0.3, flags=None):
    from src.decision.assessor import CampaignAssessor
    assessor = CampaignAssessor(Config())
    bl = make_bl(eid, ch, spend=spend, rev=spend * roas)
    from src.uncertainty.models import EntityUncertainty
    unc = EntityUncertainty(
        entity_id=eid, channel=ch,
        avg_interval_width=10.0, avg_relative_width=0.5,
        confidence_score=conf, volatility=vol,
        stability_trend="stable",
    )
    score = assessor._compute_score(bl, unc)
    return CampaignAssessment(
        entity_id=eid, channel=ch, campaign_type="SEARCH",
        current_spend=spend, current_roas=roas,
        confidence_score=conf, volatility=vol,
        stability_trend="stable", flags=flags or [],
        score=score,
    )


@pytest.fixture
def config():
    return Config()


@pytest.fixture
def optimizer(config):
    return BudgetOptimizer(config)


class TestOptimize:
    def test_returns_list(self, optimizer):
        baselines = {"a": make_bl("a", "google")}
        assessments = [make_assessment("a", "google")]
        result = optimizer.optimize(assessments, baselines)
        assert isinstance(result, list)

    def test_returns_allocation_recommendations(self, optimizer):
        baselines = {"a": make_bl("a", "google")}
        assessments = [make_assessment("a", "google")]
        result = optimizer.optimize(assessments, baselines)
        assert isinstance(result[0], AllocationRecommendation)

    def test_entity_id_matches(self, optimizer):
        baselines = {"a": make_bl("a", "google")}
        assessments = [make_assessment("a", "google")]
        result = optimizer.optimize(assessments, baselines)
        assert result[0].entity_id == "a"

    def test_channel_matches(self, optimizer):
        baselines = {"a": make_bl("a", "google")}
        assessments = [make_assessment("a", "google")]
        result = optimizer.optimize(assessments, baselines)
        assert result[0].channel == "google"

    def test_current_budget_matches_baseline_spend(self, optimizer):
        baselines = {"a": make_bl("a", "google", spend=1000.0)}
        assessments = [make_assessment("a", "google", spend=1000.0)]
        result = optimizer.optimize(assessments, baselines)
        assert result[0].current_budget == 1000.0

    def test_recommended_budget_for_single_campaign(self, optimizer):
        bl = {"a": make_bl("a", "google", spend=500.0)}
        assessments = [make_assessment("a", "google", spend=500.0)]
        result = optimizer.optimize(assessments, bl)
        # Single campaign should keep total budget
        assert result[0].recommended_budget == pytest.approx(500.0)

    def test_better_campaign_gets_more_budget(self, optimizer):
        bl = {
            "good": make_bl("good", "google", spend=500.0, rev=2500.0),  # ROAS=5
            "bad": make_bl("bad", "google", spend=500.0, rev=500.0),     # ROAS=1
        }
        assessments = [
            make_assessment("good", "google", spend=500.0, roas=5.0, conf=0.9, vol=0.1),
            make_assessment("bad", "google", spend=500.0, roas=1.0, conf=0.3, vol=0.8),
        ]
        result = optimizer.optimize(assessments, bl)
        rec = {r.entity_id: r.recommended_budget for r in result}
        assert rec["good"] > rec["bad"]

    def test_budget_change_clamped_to_max_change_ratio(self, optimizer):
        bl = {
            "big": make_bl("big", "google", spend=1000.0, rev=500.0),    # ROAS=0.5
            "small": make_bl("small", "google", spend=10.0, rev=100.0),  # ROAS=10
        }
        assessments = [
            make_assessment("big", "google", spend=1000.0, roas=0.5, conf=0.3, vol=0.8),
            make_assessment("small", "google", spend=10.0, roas=10.0, conf=0.9, vol=0.1),
        ]
        result = optimizer.optimize(assessments, bl)
        big_rec = [r for r in result if r.entity_id == "big"][0]
        small_rec = [r for r in result if r.entity_id == "small"][0]
        # Clamped at ±50% of current
        assert big_rec.recommended_budget >= 500.0  # 1000 - 50%
        assert small_rec.recommended_budget <= 15.0  # 10 + 50%

    def test_empty_assessments_returns_empty(self, optimizer):
        result = optimizer.optimize([], {})
        assert result == []

    def test_unknown_baseline_skipped(self, optimizer):
        bl = {"a": make_bl("a", "google")}
        assessments = [make_assessment("unknown", "google")]
        result = optimizer.optimize(assessments, bl)
        assert all(r.entity_id == "a" for r in result)

    def test_recommended_budget_non_negative(self, optimizer):
        bl = {"a": make_bl("a", "google", spend=10.0, rev=5.0)}
        assessments = [make_assessment("a", "google", spend=10.0, roas=0.5, conf=0.3, vol=0.9)]
        result = optimizer.optimize(assessments, bl)
        assert result[0].recommended_budget >= 0.0

    def test_rationale_includes_performing_well_without_flags(self, optimizer):
        bl = {"a": make_bl("a", "google", spend=100.0, rev=500.0)}
        assessments = [make_assessment("a", "google", spend=100.0, roas=5.0, conf=0.9, vol=0.1)]
        result = optimizer.optimize(assessments, bl)
        assert "performing well" in result[0].rationale.lower()

    def test_rationale_includes_maintain(self, optimizer):
        bl = {"a": make_bl("a", "google", spend=500.0)}
        # Single campaign maintains same budget
        assessments = [make_assessment("a", "google", spend=500.0)]
        result = optimizer.optimize(assessments, bl)
        if abs(result[0].recommended_budget - result[0].current_budget) < 1.0:
            assert "Maintain" in result[0].rationale

    def test_rationale_references_flags(self, optimizer):
        bl = {"a": make_bl("a", "google", spend=500.0, rev=500.0)}  # ROAS=1 < 3
        assessments = [make_assessment("a", "google", spend=500.0, roas=1.0, flags=["below_roas_target"])]
        result = optimizer.optimize(assessments, bl)
        assert "ROAS is below target" in result[0].rationale

    def test_multiple_flags_in_rationale(self, optimizer):
        bl = {"a": make_bl("a", "google", spend=500.0, rev=500.0)}
        assessments = [make_assessment("a", "google", spend=500.0, roas=1.0, conf=0.3, flags=["below_roas_target", "high_uncertainty"])]
        result = optimizer.optimize(assessments, bl)
        assert "ROAS is below target" in result[0].rationale
        assert "confidence is low" in result[0].rationale

    def test_expected_roas_matches_current_roas(self, optimizer):
        bl = {"a": make_bl("a", "google", spend=500.0, rev=2000.0)}
        assessments = [make_assessment("a", "google", spend=500.0, roas=4.0)]
        result = optimizer.optimize(assessments, bl)
        assert result[0].expected_roas == 4.0

    def test_poor_campaign_gets_decrease_rationale(self, optimizer):
        # When total budget is constant, poor campaign should get less
        bl = {
            "good": make_bl("good", "google", spend=500.0, rev=3000.0),
            "poor": make_bl("poor", "google", spend=500.0, rev=500.0),
        }
        assessments = [
            make_assessment("good", "google", spend=500.0, roas=6.0, conf=0.9, vol=0.1),
            make_assessment("poor", "google", spend=500.0, roas=1.0, conf=0.3, vol=0.8, flags=["below_roas_target"]),
        ]
        result = optimizer.optimize(assessments, bl)
        poor_rec = [r for r in result if r.entity_id == "poor"][0]
        assert "Decrease" in poor_rec.rationale
