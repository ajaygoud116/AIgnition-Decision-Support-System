from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.models.common import ForecastResult
from src.uncertainty.metrics import (
    aggregate_entities,
    compute_horizon_breakdown,
    compute_interval_widths,
    compute_nonconformity_scores,
    compute_relative_widths,
    compute_stability_trend,
    compute_volatility,
    confidence_from_relative_width,
    find_calibration_quantile,
    pair_forecast_with_actuals,
)
from src.uncertainty.models import ChannelUncertainty, EntityUncertainty, UncertaintyReport
from src.utils.config import Config
from src.utils.logger import StructuredLogger


class UncertaintyEngine:
    """Computes uncertainty metrics from a ForecastResult.

    Each ForecastSeries (entity_id x horizon) produces per-point widths,
    relative widths, confidence, volatility, and stability trend.  These
    are aggregated by entity (campaign) and channel, and assembled into
    an UncertaintyReport that the decision layer consumes.

    Conformal calibration: call calibrate() with a held-out forecast +
    actuals to compute a scaling factor that adjusts interval widths so
    that empirical coverage matches the target (default 80%).
    """

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("uncertainty.engine")
        self._threshold = config.get("uncertainty.relative_width_threshold", 2.0)
        self._volatility_threshold = config.get("decision.volatility_threshold", 0.5)
        self._target_coverage = config.get("uncertainty.target_coverage", 0.80)
        self._calibration_alpha: Optional[float] = None

    def calibrate(
        self,
        forecast_result: ForecastResult,
        actuals_df: pd.DataFrame,
    ) -> float:
        """Split-conformal calibration of prediction intervals.

        Steps:
          1. Align every forecast point with its actual value.
          2. Compute nonconformity scores a = how much the interval
             must expand to cover the actual.
          3. Find the q-th quantile of scores where
             q = ceil((n+1) * target_coverage) / n  (finite-sample correction).
          4. Store the calibration factor alpha = q.

        The scaled interval for future predictions becomes:
          [p50 - alpha*(p50-p10),  p50 + alpha*(p90-p50)]

        Args:
            forecast_result: ForecastResult for the calibration period.
            actuals_df: DataFrame with columns [date, campaign_id, revenue].

        Returns:
            Calibration factor alpha.
        """
        pairs = pair_forecast_with_actuals(forecast_result, actuals_df)

        if len(pairs) < 10:
            self._logger.warning(
                "calibration_insufficient_data",
                pairs=len(pairs),
                min_required=10,
            )
            self._calibration_alpha = 1.0
            return 1.0

        scores = compute_nonconformity_scores(pairs)
        self._calibration_alpha = find_calibration_quantile(scores, self._target_coverage)

        # Compute achieved coverage on calibration set
        from src.uncertainty.metrics import apply_calibration
        stats = apply_calibration(pairs, self._calibration_alpha)

        self._logger.info(
            "calibration_complete",
            alpha=round(self._calibration_alpha, 4),
            target_coverage=self._target_coverage,
            achieved_coverage=round(stats["coverage"], 4),
            n_pairs=stats["n_pairs"],
            n_inside=stats["inside"],
        )
        return self._calibration_alpha

    def compute(self, forecast_result: ForecastResult) -> UncertaintyReport:
        """Produce an UncertaintyReport from a ForecastResult.

        If calibrate() has been called, the calibration factor is applied
        to scale interval widths before computing confidence scores.
        """
        self._logger.info(
            "uncertainty_compute_start",
            series=len(forecast_result.series),
        )

        entity_map: Dict[str, List[EntityUncertainty]] = {}
        alpha = self._calibration_alpha if self._calibration_alpha is not None else 1.0

        for series in forecast_result.series:
            eid = series.entity_id
            raw_widths = compute_interval_widths(series)
            rel_widths = compute_relative_widths(series)

            # Apply calibration: scale interval widths
            calibrated_widths = raw_widths * alpha
            calibrated_rel = rel_widths * alpha

            confidence = confidence_from_relative_width(calibrated_rel, self._threshold)
            volatility = compute_volatility(calibrated_rel)
            trend = compute_stability_trend(series)
            breakdown = compute_horizon_breakdown(series)

            entity_unc = EntityUncertainty(
                entity_id=eid,
                channel=series.channel,
                avg_interval_width=float(calibrated_widths.mean()),
                avg_relative_width=float(calibrated_rel.mean()),
                confidence_score=confidence,
                volatility=volatility,
                stability_trend=trend,
                calibrated_coverage=self._target_coverage,
                calibration_alpha=alpha,
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
                "model": "uncertainty_engine_v2",
                "relative_width_threshold": self._threshold,
                "volatility_threshold": self._volatility_threshold,
                "target_coverage": self._target_coverage,
                "calibration_alpha": alpha,
            },
        )

        self._logger.info(
            "uncertainty_compute_complete",
            entities=len(entities),
            channels=len(channels),
            high_uncertainty=high_count,
            calibration_alpha=round(alpha, 4),
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
                    calibrated_coverage=float(
                        sum(u.calibrated_coverage for u in uncs) / len(uncs)
                    ),
                    calibration_alpha=float(
                        sum(u.calibration_alpha for u in uncs) / len(uncs)
                    ),
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
