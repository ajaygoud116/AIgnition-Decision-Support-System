from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.common import ForecastResult
from src.uncertainty.metrics import (
    aggregate_entities,
    compute_horizon_breakdown,
    compute_relative_widths,
    compute_stability_trend,
    compute_volatility,
    confidence_from_relative_width,
)
from src.uncertainty.models import ChannelUncertainty, EntityUncertainty, UncertaintyReport
from src.utils.config import Config
from src.utils.logger import StructuredLogger


class UncertaintyEngine:
    """Computes uncertainty metrics from a ForecastResult.

    Each ForecastSeries (entity_id × horizon) produces per-point widths,
    relative widths, confidence, volatility, and stability trend.  These
    are aggregated by entity (campaign) and channel, and assembled into
    an UncertaintyReport that the decision layer consumes.

    Supports conformal calibration: call calibrate() with a held-out
    forecast + actuals to compute a scaling factor that adjusts interval
    widths to achieve target coverage.
    """

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("uncertainty.engine")
        self._threshold = config.get("uncertainty.relative_width_threshold", 2.0)
        self._volatility_threshold = config.get("decision.volatility_threshold", 0.5)
        self._calibration_factor: Optional[float] = None

    def calibrate(
        self,
        forecast_result: ForecastResult,
        actuals_df: pd.DataFrame,
    ) -> float:
        """Conformal calibration via bisection.

        Finds a scaling factor α such that the empirical coverage of
        the adjusted intervals [p50 - α*(p50-p10), p50 + α*(p90-p50)]
        matches the target coverage.

        Args:
            forecast_result: ForecastResult for the calibration period.
            actuals_df: DataFrame with columns [date, campaign_id, revenue].

        Returns:
            Calibration factor α.
        """
        target = self._config.get("uncertainty.target_coverage", 0.80)

        pairs = []
        for series in forecast_result.series:
            eid = series.entity_id
            for point in series.points:
                mask = (
                    (actuals_df["campaign_id"] == eid)
                    & (actuals_df["date"] == pd.Timestamp(point.date))
                )
                match = actuals_df.loc[mask]
                if len(match) > 0:
                    pairs.append({
                        "actual": match["revenue"].values[0],
                        "p10": point.values.p10,
                        "p50": point.values.p50,
                        "p90": point.values.p90,
                    })

        if len(pairs) < 10:
            self._logger.warning(
                "calibration_insufficient_data",
                pairs=len(pairs),
                min_required=10,
            )
            self._calibration_factor = 1.0
            return 1.0

        def _empirical_coverage(alpha: float) -> float:
            inside = 0
            for p in pairs:
                lo = p["p50"] - alpha * (p["p50"] - p["p10"])
                hi = p["p50"] + alpha * (p["p90"] - p["p50"])
                if lo <= p["actual"] <= hi:
                    inside += 1
            return inside / len(pairs)

        lo, hi = 0.1, 5.0
        for _ in range(50):
            mid = (lo + hi) / 2.0
            if _empirical_coverage(mid) < target:
                lo = mid
            else:
                hi = mid

        self._calibration_factor = (lo + hi) / 2.0
        achieved = _empirical_coverage(self._calibration_factor)
        self._logger.info(
            "calibration_complete",
            factor=round(self._calibration_factor, 4),
            target_coverage=target,
            achieved_coverage=round(achieved, 4),
            n_pairs=len(pairs),
        )
        return self._calibration_factor

    def compute(self, forecast_result: ForecastResult) -> UncertaintyReport:
        """Produce an UncertaintyReport from a ForecastResult."""
        self._logger.info(
            "uncertainty_compute_start",
            series=len(forecast_result.series),
        )

        entity_map: Dict[str, List[EntityUncertainty]] = {}

        for series in forecast_result.series:
            eid = series.entity_id
            rel_widths = compute_relative_widths(series)
            if self._calibration_factor is not None:
                rel_widths = rel_widths * self._calibration_factor
            confidence = confidence_from_relative_width(rel_widths, self._threshold)
            volatility = compute_volatility(rel_widths)
            trend = compute_stability_trend(series)
            breakdown = compute_horizon_breakdown(series)

            entity_unc = EntityUncertainty(
                entity_id=eid,
                channel=series.channel,
                avg_interval_width=float(rel_widths.mean()),
                avg_relative_width=float(rel_widths.mean()),
                confidence_score=confidence,
                volatility=volatility,
                stability_trend=trend,
                horizon_breakdown={
                    str(series.horizon.value): breakdown.get("late", 0.0),
                },
            )
            entity_map.setdefault(eid, []).append(entity_unc)

        entities = self._merge_entity_series(entity_map)
        channels = self._aggregate_channels(entities)
        overall_confidence, overall_volatility, high_count = aggregate_entities(entities)

        report = UncertaintyReport(
            entities=entities,
            channels=channels,
            overall_confidence=overall_confidence,
            overall_volatility=overall_volatility,
            high_uncertainty_count=high_count,
            metadata={
                "model": "uncertainty_engine_v1",
                "relative_width_threshold": self._threshold,
                "volatility_threshold": self._volatility_threshold,
            },
        )

        self._logger.info(
            "uncertainty_compute_complete",
            entities=len(entities),
            channels=len(channels),
            high_uncertainty=high_count,
        )
        return report

    def _merge_entity_series(self, entity_map: Dict[str, List[EntityUncertainty]]) -> List[EntityUncertainty]:
        """Average multiple horizon-series into a single EntityUncertainty."""
        merged: List[EntityUncertainty] = []
        for eid, uncs in entity_map.items():
            if not uncs:
                continue
            merged.append(
                EntityUncertainty(
                    entity_id=eid,
                    channel=uncs[0].channel,
                    avg_interval_width=float(
                        sum(u.avg_interval_width for u in uncs) / len(uncs)
                    ),
                    avg_relative_width=float(
                        sum(u.avg_relative_width for u in uncs) / len(uncs)
                    ),
                    confidence_score=float(
                        sum(u.confidence_score for u in uncs) / len(uncs)
                    ),
                    volatility=float(sum(u.volatility for u in uncs) / len(uncs)),
                    stability_trend=uncs[0].stability_trend,
                    horizon_breakdown={
                        k: v for u in uncs for k, v in u.horizon_breakdown.items()
                    },
                )
            )
        return merged

    def _aggregate_channels(self, entities: List[EntityUncertainty]) -> List[ChannelUncertainty]:
        """Group entities by channel and compute channel-level metrics."""
        channel_map: Dict[str, List[EntityUncertainty]] = {}
        for e in entities:
            channel_map.setdefault(e.channel, []).append(e)

        result: List[ChannelUncertainty] = []
        for ch, ch_entities in channel_map.items():
            avg_conf = float(sum(e.confidence_score for e in ch_entities)) / len(ch_entities)
            avg_vol = float(sum(e.volatility for e in ch_entities)) / len(ch_entities)
            high_unc = [e.entity_id for e in ch_entities if e.confidence_score < 0.5]
            result.append(
                ChannelUncertainty(
                    channel=ch,
                    avg_confidence=avg_conf,
                    avg_volatility=avg_vol,
                    campaign_count=len(ch_entities),
                    high_uncertainty_campaigns=high_unc,
                )
            )
        return result
