from typing import Dict, List, Tuple

import numpy as np

from src.models.common import ForecastSeries
from src.uncertainty.models import EntityUncertainty


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
