import numpy as np
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
    compute_horizon_breakdown,
    compute_interval_widths,
    compute_relative_widths,
    compute_stability_trend,
    compute_volatility,
    confidence_from_relative_width,
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
