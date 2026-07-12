import numpy as np
import pandas as pd
import pytest

from src.models.common import (
    ForecastPoint,
    ForecastSeries,
    ForecastResult,
    Granularity,
    Horizon,
    MetricType,
    QuantileValue,
)
from src.uncertainty.metrics import (
    aggregate_entities,
    apply_calibration,
    compute_calibration_error,
    compute_coverage_rate,
    compute_horizon_breakdown,
    compute_interval_widths,
    compute_nonconformity_scores,
    compute_relative_widths,
    compute_reliability,
    compute_stability_trend,
    compute_volatility,
    confidence_from_relative_width,
    find_calibration_quantile,
    pair_forecast_with_actuals,
)
from src.uncertainty.models import EntityUncertainty


def make_series(p10s, p50s, p90s, horizon=Horizon.D30):
    points = [
        ForecastPoint(
            date=type("d", (), {"date": f"2025-01-{i+1}"})(),
            values=QuantileValue(p10=float(p10), p50=float(p50), p90=float(p90)),
        )
        for i, (p10, p50, p90) in enumerate(zip(p10s, p50s, p90s))
    ]
    return ForecastSeries(
        entity_id="g_1",
        channel="google",
        granularity=Granularity.CAMPAIGN,
        metric=MetricType.REVENUE,
        horizon=horizon,
        points=points,
    )


class TestComputeIntervalWidths:
    def test_basic_widths(self):
        s = make_series([10, 20], [50, 60], [90, 100])
        widths = compute_interval_widths(s)
        np.testing.assert_array_almost_equal(widths, [80, 80])

    def test_zero_width(self):
        s = make_series([50], [50], [50])
        widths = compute_interval_widths(s)
        assert widths[0] == 0.0


class TestComputeRelativeWidths:
    def test_relative_widths(self):
        s = make_series([90, 0], [100, 50], [110, 100])
        rel = compute_relative_widths(s)
        assert rel[0] == pytest.approx(0.2)
        assert rel[1] == pytest.approx(2.0)

    def test_zero_p50_uses_epsilon(self):
        s = make_series([0], [0], [10])
        rel = compute_relative_widths(s, epsilon=1e-10)
        assert rel[0] > 0


class TestConfidenceFromRelativeWidth:
    def test_perfect_confidence(self):
        assert confidence_from_relative_width(np.array([0.0, 0.0]), threshold=2.0) == 1.0

    def test_zero_confidence(self):
        assert confidence_from_relative_width(np.array([10.0]), threshold=2.0) == 0.0

    def test_half_confidence(self):
        c = confidence_from_relative_width(np.array([1.0]), threshold=2.0)
        assert c == pytest.approx(0.5)


class TestComputeVolatility:
    def test_no_volatility(self):
        v = compute_volatility(np.array([0.5, 0.5, 0.5]))
        assert v == 0.0

    def test_high_volatility(self):
        v = compute_volatility(np.array([0.1, 0.5, 0.9]))
        assert v > 0.5

    def test_single_point(self):
        v = compute_volatility(np.array([0.5]))
        assert v == 0.0


class TestComputeStabilityTrend:
    def test_stable(self):
        s = make_series([10, 10, 10], [50, 50, 50], [90, 90, 90])
        assert compute_stability_trend(s) == "stable"

    def test_widening(self):
        s = make_series([10, 5, 0], [50, 50, 50], [90, 95, 100])
        assert compute_stability_trend(s) == "widening"

    def test_narrowing(self):
        s = make_series([0, 5, 10], [50, 50, 50], [100, 95, 90])
        assert compute_stability_trend(s) == "narrowing"

    def test_short_series_returns_stable(self):
        s = make_series([10], [50], [90])
        assert compute_stability_trend(s) == "stable"


class TestComputeHorizonBreakdown:
    def test_three_parts(self):
        s = make_series(
            [0, 0, 0, 10, 10, 10],
            [100] * 6,
            [20, 20, 20, 30, 30, 30],
        )
        breakdown = compute_horizon_breakdown(s)
        assert "early" in breakdown
        assert "mid" in breakdown
        assert "late" in breakdown

    def test_empty_series(self):
        s = make_series([], [], [])
        breakdown = compute_horizon_breakdown(s)
        assert breakdown["early"] == 0.0


class TestAggregateEntities:
    def test_empty(self):
        conf, vol, count = aggregate_entities([])
        assert conf == 1.0
        assert vol == 0.0
        assert count == 0

    def test_all_confident(self):
        entities = [
            EntityUncertainty("g_1", "google", 0.5, 0.5, 0.9, 0.1, "stable"),
            EntityUncertainty("g_2", "google", 0.5, 0.5, 0.8, 0.2, "stable"),
        ]
        conf, vol, count = aggregate_entities(entities)
        assert conf == pytest.approx(0.85)
        assert count == 0

    def test_some_uncertain(self):
        entities = [
            EntityUncertainty("g_1", "google", 0.5, 0.5, 0.3, 0.1, "widening"),
            EntityUncertainty("g_2", "google", 0.5, 0.5, 0.9, 0.1, "stable"),
        ]
        conf, vol, count = aggregate_entities(entities)
        assert count == 1


# ── Conformal calibration tests ──────────────────────────────────────────


def _make_cal_forecast() -> ForecastResult:
    """Two entities, each with 3 forecast points."""
    now = pd.Timestamp("2026-01-01")
    entities = ["camp_a", "camp_b"]
    series_list = []
    for eid in entities:
        pts = [
            ForecastPoint(
                date=now + pd.Timedelta(days=i),
                values=QuantileValue(p10=10.0, p50=20.0, p90=30.0),
            )
            for i in range(3)
        ]
        series_list.append(
            ForecastSeries(
                entity_id=eid,
                channel="google",
                granularity=Granularity.CAMPAIGN,
                metric=MetricType.REVENUE,
                horizon=Horizon.D30,
                points=pts,
            )
        )
    return ForecastResult(series=series_list, metadata={})


def _make_actuals() -> pd.DataFrame:
    rows = []
    for eid in ["camp_a", "camp_b"]:
        for d in range(3):
            rows.append({
                "campaign_id": eid,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=d),
                "revenue": 15.0 if d == 0 else 25.0 if d == 1 else 35.0,
            })
    return pd.DataFrame(rows)


class TestPairForecastWithActuals:
    def test_basic_pairing(self):
        fc = _make_cal_forecast()
        actuals = _make_actuals()
        pairs = pair_forecast_with_actuals(fc, actuals)
        assert len(pairs) == 6  # 2 entities x 3 days
        for p in pairs:
            assert "entity_id" in p
            assert "actual" in p
            assert "p10" in p
            assert "p50" in p
            assert "p90" in p
            assert "horizon" in p
            assert "channel" in p

    def test_skip_no_match(self):
        fc = _make_cal_forecast()
        # actuals for only one entity
        actuals = _make_actuals()
        actuals = actuals[actuals["campaign_id"] == "camp_a"]
        pairs = pair_forecast_with_actuals(fc, actuals)
        assert len(pairs) == 3

    def test_empty_forecast(self):
        fc = ForecastResult(series=[], metadata={})
        actuals = _make_actuals()
        pairs = pair_forecast_with_actuals(fc, actuals)
        assert pairs == []


class TestComputeNonconformityScores:
    def test_normalized_residual_inside(self):
        # actual at p50 -> score 0
        pairs = [{"actual": 20.0, "p10": 10.0, "p50": 20.0, "p90": 30.0}]
        scores = compute_nonconformity_scores(pairs, "normalized_residual")
        assert scores[0] == 0.0

    def test_normalized_residual_at_edge(self):
        # actual at p90 -> distance = (p90-p10)/2 from p50 -> score = 1.0
        pairs = [{"actual": 30.0, "p10": 10.0, "p50": 20.0, "p90": 30.0}]
        scores = compute_nonconformity_scores(pairs, "normalized_residual")
        assert scores[0] == pytest.approx(1.0)

    def test_normalized_residual_outside(self):
        # actual far outside -> score > 1
        pairs = [{"actual": 60.0, "p10": 10.0, "p50": 20.0, "p90": 30.0}]
        scores = compute_nonconformity_scores(pairs, "normalized_residual")
        assert scores[0] > 1.0

    def test_scaling_factor_inside(self):
        pairs = [{"actual": 15.0, "p10": 10.0, "p50": 20.0, "p90": 30.0}]
        scores = compute_nonconformity_scores(pairs, "scaling_factor")
        assert scores[0] == 1.0

    def test_scaling_factor_below(self):
        pairs = [{"actual": 0.0, "p10": 10.0, "p50": 20.0, "p90": 30.0}]
        scores = compute_nonconformity_scores(pairs, "scaling_factor")
        assert scores[0] == pytest.approx(2.0)  # (20-0)/(20-10) = 2


class TestFindCalibrationQuantile:
    def test_all_inside(self):
        scores = np.array([0.5, 0.6, 0.7])
        # n=3, target=0.8 → ceil(4*0.8)=ceil(3.2)=4 → idx=min(4,3)-1=2 → sorted[2]=0.7
        a = find_calibration_quantile(scores, 0.8)
        assert a == pytest.approx(0.7)

    def test_empty_fallback(self):
        a = find_calibration_quantile(np.array([]), 0.8)
        assert a == 1.0

    def test_cap_disabled(self):
        scores = np.array([1.0, 2.0, 100.0, 200.0, 300.0])
        a = find_calibration_quantile(scores, 0.8, cap_percentile=100.0)
        # n=5, target=0.8 → ceil(6*0.8)=ceil(4.8)=5 → idx=4 → sorted[4]=300
        assert a == pytest.approx(300.0)


class TestApplyCalibration:
    def test_normalized_residual_all_inside(self):
        # With alpha=1.0: interval = p50 +/- (p90-p10)/2
        # p50=20, (p90-p10)/2=10 → interval [10, 30]
        # actual=20 → inside
        pairs = [{"actual": 20.0, "p10": 10.0, "p50": 20.0, "p90": 30.0}]
        stats = apply_calibration(pairs, 1.0, "normalized_residual")
        assert stats["coverage"] == 1.0
        assert stats["n_pairs"] == 1
        assert stats["inside"] == 1

    def test_scaling_factor_all_inside(self):
        # With alpha=1.0: interval=[p10, p90]=[10, 30]
        pairs = [{"actual": 15.0, "p10": 10.0, "p50": 20.0, "p90": 30.0}]
        stats = apply_calibration(pairs, 1.0, "scaling_factor")
        assert stats["coverage"] == 1.0

    def test_alpha_zero(self):
        pairs = [{"actual": 15.0, "p10": 10.0, "p50": 20.0, "p90": 30.0}]
        stats = apply_calibration(pairs, 0.0, "normalized_residual")
        # interval collapses to p50 -> actual != p50 -> no coverage
        assert stats["coverage"] == 0.0

    def test_partial_coverage(self):
        pairs = [
            {"actual": 20.0, "p10": 10.0, "p50": 20.0, "p90": 30.0},  # inside
            {"actual": 50.0, "p10": 10.0, "p50": 20.0, "p90": 30.0},  # outside
        ]
        stats = apply_calibration(pairs, 1.0, "normalized_residual")
        assert stats["coverage"] == 0.5
        assert stats["inside"] == 1
        assert stats["n_pairs"] == 2


class TestComputeCoverageRate:
    def test_no_alpha(self):
        """Without alpha, uses raw [p10, p90]."""
        s = _make_cal_forecast().series[0]
        actuals = {
            str(pd.Timestamp("2026-01-01")): 15.0,
            str(pd.Timestamp("2026-01-02")): 25.0,
            str(pd.Timestamp("2026-01-03")): 35.0,
        }
        # Raw interval [10, 30]; 15 (inside), 25 (inside), 35 (outside)
        cov = compute_coverage_rate(s, actuals)
        assert cov == pytest.approx(2 / 3)

    def test_with_alpha(self):
        s = _make_cal_forecast().series[0]
        actuals = {
            str(pd.Timestamp("2026-01-01")): 15.0,
            str(pd.Timestamp("2026-01-02")): 25.0,
            str(pd.Timestamp("2026-01-03")): 35.0,
        }
        # alpha=2, normalized_residual: interval = p50 +/- 2*(p90-p10)/2
        #   = 20 +/- 2*10 = [0, 40]
        # All inside
        cov = compute_coverage_rate(s, actuals, alpha=2.0)
        assert cov == 1.0


class TestComputeCalibrationError:
    def test_perfect(self):
        assert compute_calibration_error(0.80, 0.80) == 0.0

    def test_under(self):
        assert compute_calibration_error(0.50, 0.80) == pytest.approx(0.30)

    def test_over(self):
        assert compute_calibration_error(0.95, 0.80) == pytest.approx(0.15)


class TestComputeReliability:
    def test_returns_structure(self):
        pairs = [
            {"actual": 15.0, "p10": 10.0, "p50": 20.0, "p90": 30.0},
            {"actual": 25.0, "p10": 10.0, "p50": 20.0, "p90": 30.0},
        ]
        result = compute_reliability(pairs, n_bins=5)
        assert "bins" in result
        assert "empirical" in result
        assert "counts" in result
        assert len(result["bins"]) == 5
        assert len(result["empirical"]) == 5
        assert len(result["counts"]) == 5
