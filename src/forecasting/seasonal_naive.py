from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logger import StructuredLogger


class SeasonalNaiveForecaster:
    """Per-campaign seasonal naive model.

    For each campaign, the forecast is the average of the last N days
    of actual revenue.  Optionally uses day-of-week patterns from the
    last complete week.
    """

    def __init__(self, window: int = 7):
        self._window = window
        self._baselines: Dict[str, float] = {}
        self._dow_patterns: Dict[str, Dict[int, float]] = {}
        self._logger = StructuredLogger("forecasting.seasonal_naive")

    def fit(self, df: pd.DataFrame, target_col: str = "revenue") -> "SeasonalNaiveForecaster":
        """Compute per-campaign baselines from the last window days."""
        if "campaign_id" not in df.columns:
            return self

        df_sorted = df.sort_values(["campaign_id", "date"])

        for cid, group in df_sorted.groupby("campaign_id", sort=False):
            tail = group.tail(self._window)
            self._baselines[cid] = float(tail[target_col].mean())

            last_week = group.tail(7).copy()
            if len(last_week) >= 7:
                last_week["_dow"] = last_week["date"].dt.dayofweek
                self._dow_patterns[cid] = {
                    int(dow): float(grp[target_col].mean())
                    for dow, grp in last_week.groupby("_dow")
                }

        self._logger.info(
            "sn_fit_complete",
            campaigns=len(self._baselines),
        )
        return self

    def predict(
        self,
        entity_id: str,
        future_dates: pd.DatetimeIndex,
    ) -> np.ndarray:
        """Return a flat array of predicted revenue for each future date."""
        baseline = self._baselines.get(entity_id, 0.0)
        dow_pattern = self._dow_patterns.get(entity_id, {})

        if dow_pattern:
            values = np.array([dow_pattern.get(d.dayofweek, baseline) for d in future_dates])
        else:
            values = np.full(len(future_dates), baseline)

        return values
