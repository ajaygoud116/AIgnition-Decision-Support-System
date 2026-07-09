import pandas as pd

from src.features.transforms import (
    add_lag_features,
    add_ratio_features,
    add_rolling_features,
    add_time_features,
    drop_high_na_rows,
)
from src.utils.config import Config
from src.utils.logger import StructuredLogger


class FeatureBuilder:
    """Orchestrates feature engineering on a cleaned unified DataFrame.

    Groups by campaign_id, sorts chronologically, applies time/ratio/rolling/lag
    transforms per campaign, drops high-NA rows, and returns the enriched DataFrame.
    """

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("features.builder")
        self._rolling_windows = config.get("features.rolling_windows", [7, 14, 30])
        self._lag_windows = config.get("features.lag_windows", [1, 7, 14, 30])
        self._max_na_ratio = config.get("features.max_na_ratio", 0.3)

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature transforms per campaign and return enriched DataFrame."""
        self._logger.info(
            "feature_build_start",
            rows=len(df),
            campaigns=df["campaign_id"].nunique(),
        )

        df = add_time_features(df)
        df = add_ratio_features(df)

        df = self._apply_per_campaign(df, add_rolling_features, windows=self._rolling_windows)
        df = self._apply_per_campaign(df, add_lag_features, lags=self._lag_windows)

        before = len(df)
        df = drop_high_na_rows(df, max_na_ratio=self._max_na_ratio)
        dropped = before - len(df)

        n_features = len([c for c in df.columns if c not in self._base_columns(df)])
        self._logger.info(
            "feature_build_complete",
            rows=len(df),
            features=n_features,
            rows_dropped=dropped,
        )

        return df

    def _apply_per_campaign(self, df: pd.DataFrame, fn, **kwargs) -> pd.DataFrame:
        """Apply a transform function group-wise per campaign and reassemble."""
        if "campaign_id" not in df.columns:
            return fn(df, **kwargs)

        groups = []
        for _, group in df.sort_values(["campaign_id", "date"]).groupby("campaign_id", sort=False):
            groups.append(fn(group, **kwargs))
        return pd.concat(groups, ignore_index=True)

    @staticmethod
    def _base_columns(df: pd.DataFrame) -> set:
        """Identify the original unified schema columns present in the DataFrame."""
        base = {
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        }
        return {c for c in base if c in df.columns}
