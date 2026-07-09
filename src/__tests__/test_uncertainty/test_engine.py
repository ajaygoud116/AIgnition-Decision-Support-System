import pytest

from src.models.common import (
    ForecastPoint,
    ForecastResult,
    ForecastSeries,
    Granularity,
    Horizon,
    MetricType,
    QuantileValue,
)
from src.uncertainty.engine import UncertaintyEngine
from src.uncertainty.models import UncertaintyReport
from src.utils.config import Config


def make_series(eid, channel, p10s, p50s, p90s, horizon=Horizon.D30):
    """Helper to build a ForecastSeries from parallel lists."""
    points = [
        ForecastPoint(
            date=type("d", (), {"date": f"2025-01-{i+1:02d}"})(),
            values=QuantileValue(p10=float(p10), p50=float(p50), p90=float(p90)),
        )
        for i, (p10, p50, p90) in enumerate(zip(p10s, p50s, p90s))
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
def engine():
    return UncertaintyEngine(Config())


@pytest.fixture
def mixed_forecast():
    """Two campaigns across 3 horizons each — one tight, one wide."""
    series = []
    tight_p10 = [95 + i for i in range(30)]
    tight_p50 = [100 + i for i in range(30)]
    tight_p90 = [105 + i for i in range(30)]

    wide_p10 = [10 + i for i in range(30)]
    wide_p50 = [100 + i for i in range(30)]
    wide_p90 = [190 + i for i in range(30)]

    for horizon in [Horizon.D30, Horizon.D60, Horizon.D90]:
        h = horizon.value
        series.append(make_series("tight_campaign", "google", tight_p10[:h], tight_p50[:h], tight_p90[:h], horizon))
        series.append(make_series("wide_campaign", "meta", wide_p10[:h], wide_p50[:h], wide_p90[:h], horizon))

    return ForecastResult(series=series)


class TestUncertaintyEngine:
    def test_compute_returns_report(self, engine, mixed_forecast):
        report = engine.compute(mixed_forecast)
        assert isinstance(report, UncertaintyReport)

    def test_two_entities(self, engine, mixed_forecast):
        report = engine.compute(mixed_forecast)
        assert len(report.entities) == 2

    def test_tight_campaign_high_confidence(self, engine, mixed_forecast):
        report = engine.compute(mixed_forecast)
        tight = [e for e in report.entities if e.entity_id == "tight_campaign"][0]
        assert tight.confidence_score > 0.8

    def test_wide_campaign_low_confidence(self, engine, mixed_forecast):
        report = engine.compute(mixed_forecast)
        wide = [e for e in report.entities if e.entity_id == "wide_campaign"][0]
        assert wide.confidence_score < 0.5

    def test_two_channels(self, engine, mixed_forecast):
        report = engine.compute(mixed_forecast)
        assert len(report.channels) == 2

    def test_google_channel(self, engine, mixed_forecast):
        report = engine.compute(mixed_forecast)
        google_ch = [c for c in report.channels if c.channel == "google"][0]
        assert google_ch.high_uncertainty_campaigns == []

    def test_meta_channel_has_high_uncertainty(self, engine, mixed_forecast):
        report = engine.compute(mixed_forecast)
        meta_ch = [c for c in report.channels if c.channel == "meta"][0]
        assert "wide_campaign" in meta_ch.high_uncertainty_campaigns

    def test_overall_confidence(self, engine, mixed_forecast):
        report = engine.compute(mixed_forecast)
        assert 0.0 <= report.overall_confidence <= 1.0

    def test_metadata_present(self, engine, mixed_forecast):
        report = engine.compute(mixed_forecast)
        assert report.metadata is not None
        assert "relative_width_threshold" in report.metadata
        assert "volatility_threshold" in report.metadata

    def test_empty_forecast(self, engine):
        empty = ForecastResult(series=[])
        report = engine.compute(empty)
        assert len(report.entities) == 0
        assert len(report.channels) == 0
        assert report.overall_confidence == 1.0
        assert report.overall_volatility == 0.0
