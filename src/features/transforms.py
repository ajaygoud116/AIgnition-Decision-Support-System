from typing import List, Optional

import numpy as np
import pandas as pd

METRICS_FOR_FEATURES = [
    "spend",
    "revenue",
    "clicks",
    "impressions",
    "conversions",
    "daily_budget",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based time features from the date column."""
    result = df.copy()
    d = result["date"]
    result["dow"] = d.dt.dayofweek
    result["month"] = d.dt.month
    result["quarter"] = d.dt.quarter
    result["doy"] = d.dt.dayofyear
    result["woy"] = d.dt.isocalendar().week.astype(int)
    result["is_weekend"] = (d.dt.dayofweek >= 5).astype(int)
    return result


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio-based derived metrics."""
    result = df.copy()

    epsilon = 1e-10

    result["roas"] = np.where(
        result["spend"] > epsilon,
        result["revenue"] / result["spend"],
        0.0,
    )

    result["ctr"] = np.where(
        result["impressions"] > epsilon,
        result["clicks"] / result["impressions"],
        0.0,
    )

    result["conv_rate"] = np.where(
        result["clicks"] > epsilon,
        result["conversions"] / result["clicks"],
        0.0,
    )

    result["spend_per_click"] = np.where(
        result["clicks"] > epsilon,
        result["spend"] / result["clicks"],
        0.0,
    )

    result["revenue_per_impression"] = np.where(
        result["impressions"] > epsilon,
        result["revenue"] / result["impressions"],
        0.0,
    )

    return result


def add_rolling_features(
    df: pd.DataFrame,
    windows: List[int],
    metrics: Optional[List[str]] = None,
    min_periods: int = 2,
) -> pd.DataFrame:
    """Add rolling window statistics (mean, std) for each metric.

    Operates on a single-campaign DataFrame already sorted by date.
    Collapses after the first window row to avoid expanding-window bias.
    """
    if metrics is None:
        metrics = METRICS_FOR_FEATURES

    result = df.copy()

    for m in metrics:
        if m not in result.columns:
            continue
        for w in windows:
            result[f"{m}_rolling_mean_{w}"] = (
                result[m].shift(1).rolling(window=w, min_periods=min_periods).mean()
            )
            result[f"{m}_rolling_std_{w}"] = (
                result[m].shift(1).rolling(window=w, min_periods=min_periods).std().fillna(0.0)
            )

    return result


def add_lag_features(
    df: pd.DataFrame,
    lags: List[int],
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Add lagged values for each metric.

    Operates on a single-campaign DataFrame already sorted by date.
    """
    if metrics is None:
        metrics = METRICS_FOR_FEATURES

    result = df.copy()

    for m in metrics:
        if m not in result.columns:
            continue
        for lag in lags:
            result[f"{m}_lag_{lag}"] = result[m].shift(lag)

    return result


def drop_high_na_rows(df: pd.DataFrame, max_na_ratio: float = 0.3) -> pd.DataFrame:
    """Drop rows where the fraction of NaN values exceeds max_na_ratio."""
    if max_na_ratio >= 1.0:
        return df
    na_fraction = df.isna().mean(axis=1)
    return df.loc[na_fraction <= max_na_ratio].copy()
