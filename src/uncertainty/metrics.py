from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.common import ForecastSeries, ForecastResult
from src.uncertainty.models import EntityUncertainty


# ── Existing width-based metrics (unchanged) ──────────────────────────


def compute_interval_widths(series: ForecastSeries) -> np.ndarray:
    """Return p90 - p10 for each point in the series."""
    return np.array([p.values.p90 - p.values.p10 for p in series.points])


def compute_relative_widths(series: ForecastSeries, epsilon: float = 1e-10) -> np.ndarray:
    """Return (p90 - p10) / p50 for each point.

    Clamps p50 to epsilon to avoid division by zero.
    """
    widths = compute_interval_widths(series)
    mid = np.array([max(p.values.p50, epsilon) for p in series.points])
    return widths / mid


def confidence_from_relative_width(
    relative_widths: np.ndarray,
    threshold: float = 2.0,
) -> float:
    """Score in [0, 1]; 1 = very confident, 0 = very uncertain.

    confidence = 1 - min(mean(relative_widths) / threshold, 1.0)
    """
    avg = float(np.mean(relative_widths))
    return max(0.0, 1.0 - min(avg / threshold, 1.0))


def compute_volatility(relative_widths: np.ndarray) -> float:
    """Coefficient of variation of relative widths as a volatility measure."""
    if len(relative_widths) < 2 or float(np.mean(relative_widths)) == 0:
        return 0.0
    return float(np.std(relative_widths) / np.mean(relative_widths))


def compute_stability_trend(series: ForecastSeries) -> str:
    """Determine if intervals are narrowing, stable, or widening.

    Uses linear regression slope of interval widths over time.
    """
    widths = compute_interval_widths(series)
    n = len(widths)
    if n < 3:
        return "stable"

    x = np.arange(n)
    slope = np.polyfit(x, widths, 1)[0]
    avg_width = float(np.mean(widths))

    if avg_width == 0:
        return "stable"

    relative_slope = slope / avg_width
    if relative_slope > 0.02:
        return "widening"
    elif relative_slope < -0.02:
        return "narrowing"
    return "stable"


def compute_horizon_breakdown(series: ForecastSeries) -> Dict[str, float]:
    """Average relative width in first third, middle third, last third."""
    widths = compute_relative_widths(series)
    n = len(widths)
    if n == 0:
        return {"early": 0.0, "mid": 0.0, "late": 0.0}

    third = max(n // 3, 1)
    return {
        "early": float(np.mean(widths[:third])),
        "mid": float(np.mean(widths[third : 2 * third])),
        "late": float(np.mean(widths[2 * third :])),
    }


def aggregate_entities(
    entities: List[EntityUncertainty],
) -> Tuple[float, float, int]:
    """Overall confidence, overall volatility, high-uncertainty count."""
    if not entities:
        return 1.0, 0.0, 0.0

    overall_confidence = float(np.mean([e.confidence_score for e in entities]))
    overall_volatility = float(np.mean([e.volatility for e in entities]))
    high_count = sum(1 for e in entities if e.confidence_score < 0.5)

    return overall_confidence, overall_volatility, high_count


# ── Conformal calibration metrics ─────────────────────────────────────


def pair_forecast_with_actuals(
    forecast_result: ForecastResult,
    actuals_df: pd.DataFrame,
) -> List[dict]:
    """Align every forecast point with its corresponding actual value.

    Returns a list of dicts with keys:
      actual, p10, p50, p90, entity_id, horizon, channel.
    Skips points where no actual is available.
    """
    pairs = []
    for series in forecast_result.series:
        eid = series.entity_id
        for point in series.points:
            mask = (
                (actuals_df["campaign_id"] == eid)
                & (actuals_df["date"] == pd.Timestamp(point.date))
            )
            match = actuals_df.loc[mask]
            if len(match) == 0:
                continue
            pairs.append({
                "entity_id": eid,
                "channel": series.channel,
                "horizon": series.horizon.value,
                "actual": float(match["revenue"].values[0]),
                "p10": point.values.p10,
                "p50": point.values.p50,
                "p90": point.values.p90,
            })
    return pairs


def compute_nonconformity_scores(
    pairs: List[dict],
    score_type: str = "normalized_residual",
) -> np.ndarray:
    """Compute nonconformity scores for conformal calibration.

    score_type='scaling_factor' (original):
        alpha needed to expand [p10, p90] to cover the actual.
        alpha = 1.0 if actual in [p10, p90].
        alpha > 1.0 if actual outside.
        This can be unstable when (p50-p10) or (p90-p50) is near zero.

    score_type='normalized_residual' (default, recommended):
        |actual - p50| / ((p90-p10)/2)
        Measures how many half-interval-widths the actual is from p50.
        When alpha=1, the calibrated interval spans p50 +/- (p90-p10)/2,
        which equals one raw [p10, p90] width.
        Heavily penalises confident-but-wrong predictions.
    """
    eps = 1e-10
    scores = []
    for p in pairs:
        a, lo, mid, hi = (p["actual"], p["p10"], p["p50"], p["p90"])

        if score_type == "scaling_factor":
            if a < lo:
                denom = max(mid - lo, eps)
                score = (mid - a) / denom
            elif a > hi:
                denom = max(hi - mid, eps)
                score = (a - mid) / denom
            else:
                score = 1.0
        else:
            # normalized_residual: |actual - p50| / ((p90-p10)/2)
            # This way alpha=1 means interval spans p50 +/- (p90-p10)/2,
            # which is exactly symmetric around the raw [p10, p90] width.
            denom = max((hi - lo) / 2.0, eps)
            score = abs(a - mid) / denom

        scores.append(max(score, 0.0))
    return np.array(scores)


def find_calibration_quantile(
    scores: np.ndarray,
    target_coverage: float,
    cap_percentile: float = 100.0,
) -> float:
    """Find the q-th quantile of nonconformity scores such that
    empirical coverage of adjusted intervals >= target_coverage.

    Uses the conformal prediction finite-sample correction:
      q = ceil((n+1) * target_coverage) / n

    To guard against extreme outliers, scores can optionally be capped at the
    *cap_percentile* percentile before computing the quantile (default: no cap).
    Capping trades theoretical coverage guarantee for narrower intervals.

    Uncap scores guarantee coverage >= target_coverage in expectation
    under exchangeability.
    """
    n = len(scores)
    if n == 0:
        return 1.0

    # Cap extreme outliers
    if cap_percentile < 100.0:
        cap = float(np.percentile(scores, cap_percentile))
        scores = np.clip(scores, 0.0, cap)

    q_idx = int(np.ceil((n + 1) * target_coverage))
    q_idx = min(q_idx, n) - 1  # zero-indexed, clamp at n-1
    sorted_scores = np.sort(scores)
    return float(sorted_scores[q_idx])


def apply_calibration(
    pairs: List[dict],
    alpha: float,
    score_type: str = "normalized_residual",
) -> Dict:
    """Apply calibration scaling factor *alpha* to a list of prediction
    pairs and return coverage statistics.

    For score_type='scaling_factor':
        Adjusted interval: [p50 - alpha*(p50-p10), p50 + alpha*(p90-p50)]
        When alpha=1: interval = [p10, p90]
    For score_type='normalized_residual' (default):
        Adjusted interval: [p50 - alpha*(p90-p10)/2, p50 + alpha*(p90-p10)/2]
        When alpha=1: interval spans p50 +/- (p90-p10)/2 (one width each side)
    """
    inside = 0
    n = len(pairs)
    for p in pairs:
        a, lo, mid, hi = (p["actual"], p["p10"], p["p50"], p["p90"])
        if score_type == "scaling_factor":
            adj_lo = mid - alpha * (mid - lo)
            adj_hi = mid + alpha * (hi - mid)
        else:
            half_width = (hi - lo) / 2.0
            adj_lo = mid - alpha * half_width
            adj_hi = mid + alpha * half_width
        if adj_lo <= a <= adj_hi:
            inside += 1

    coverage = inside / n if n > 0 else 0.0
    return {
        "alpha": alpha,
        "n_pairs": n,
        "inside": inside,
        "coverage": coverage,
    }


def compute_coverage_rate(
    series: ForecastSeries,
    actuals: Dict[str, float],
    alpha: Optional[float] = None,
    score_type: str = "normalized_residual",
) -> float:
    """Compute what fraction of actuals fall inside the forecast intervals.

    If *alpha* is provided (calibration factor), intervals are scaled
    before checking coverage.
    """
    inside = 0
    total = 0
    for point in series.points:
        date_str = str(point.date)
        if date_str not in actuals:
            continue
        actual = actuals[date_str]
        lo = point.values.p10
        hi = point.values.p90
        mid = point.values.p50
        if alpha is not None:
            if score_type == "scaling_factor":
                lo = mid - alpha * (mid - lo)
                hi = mid + alpha * (hi - mid)
            else:
                half_w = (hi - lo) / 2.0
                lo = mid - alpha * half_w
                hi = mid + alpha * half_w
        if lo <= actual <= hi:
            inside += 1
        total += 1
    return inside / total if total > 0 else 0.0


def compute_calibration_error(
    coverage: float,
    target_coverage: float,
) -> float:
    """Absolute difference between achieved and target coverage."""
    return abs(coverage - target_coverage)


def compute_reliability(
    pairs: List[dict],
    n_bins: int = 10,
) -> Dict:
    """Reliability diagram: bin predicted coverage vs empirical coverage.

    For conformal prediction, the predicted coverage for each interval
    is the target coverage (all intervals have the same nominal coverage).
    This measures whether the calibration holds uniformly.

    Returns:
        bins: array of bin centers
        empirical: array of empirical coverage per bin
        counts: array of counts per bin
    """
    eps = 1e-10
    scores = compute_nonconformity_scores(pairs)
    # Normalize scores to [0, 1] for binning (alpha of 1 = nominal interval)
    # We clip to a reasonable range for visualization
    alphas = np.clip(scores, 0.0, 5.0)

    bins = np.linspace(0, 5.0, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    empirical = []
    counts = []

    for i in range(n_bins):
        mask = (alphas >= bins[i]) & (alphas < bins[i + 1])
        count = int(mask.sum())
        counts.append(count)
        if count > 0:
            covered = 0
            for idx in np.where(mask)[0]:
                p = pairs[int(idx)]
                a, lo, mid, hi = p["actual"], p["p10"], p["p50"], p["p90"]
                half_w = (hi - lo) / 2.0
                adj_lo = mid - alphas[idx] * half_w
                adj_hi = mid + alphas[idx] * half_w
                if adj_lo <= a <= adj_hi:
                    covered += 1
            empirical.append(covered / count)
        else:
            empirical.append(0.0)

    return {
        "bins": bin_centers.tolist(),
        "empirical": empirical,
        "counts": counts,
    }
