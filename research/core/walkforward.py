from typing import Generator, Tuple
import pandas as pd
import numpy as np


class ExpandingWindowCV:
    """Expanding-window cross-validation for time series.

    Each fold: train on [0:train_end], test on [train_end:train_end+horizon].
    The train window expands by step_days each fold.
    """

    def __init__(self, n_splits: int = 3, initial_train_days: int = 120,
                 step_days: int = 60, forecast_horizon: int = 30):
        self.n_splits = n_splits
        self.initial_train_days = initial_train_days
        self.step_days = step_days
        self.forecast_horizon = forecast_horizon

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """Yield (train_df, test_df) for each fold.

        The DataFrame must have a 'date' column and be sorted chronologically.
        Splits are based on sorted unique dates to preserve temporal ordering
        across all campaigns.
        """
        dates = sorted(df["date"].unique())
        if len(dates) < self.initial_train_days + self.forecast_horizon:
            raise ValueError(
                f"Need at least {self.initial_train_days + self.forecast_horizon} days, "
                f"have {len(dates)}"
            )

        for i in range(self.n_splits):
            train_end_idx = self.initial_train_days + i * self.step_days
            test_end_idx = train_end_idx + self.forecast_horizon

            if test_end_idx > len(dates):
                break

            train_end_date = dates[train_end_idx - 1]
            test_start_date = dates[train_end_idx]
            test_end_date = dates[test_end_idx - 1]

            train_df = df[df["date"] <= train_end_date].copy()
            test_df = df[(df["date"] >= test_start_date) & (df["date"] <= test_end_date)].copy()

            if train_df.empty or test_df.empty:
                continue

            yield train_df, test_df


class SlidingWindowCV:
    """Sliding-window CV: fixed window size, slides forward."""

    def __init__(self, window_days: int = 120, step_days: int = 60, forecast_horizon: int = 30):
        self.window_days = window_days
        self.step_days = step_days
        self.forecast_horizon = forecast_horizon

    def split(self, df: pd.DataFrame) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        dates = sorted(df["date"].unique())
        if len(dates) < self.window_days + self.forecast_horizon:
            return

        start = 0
        while start + self.window_days + self.forecast_horizon <= len(dates):
            train_end = start + self.window_days
            test_end = train_end + self.forecast_horizon

            train_df = df[df["date"] <= dates[train_end - 1]].copy()
            test_df = df[(df["date"] >= dates[train_end]) & (df["date"] <= dates[test_end - 1])].copy()

            if not train_df.empty and not test_df.empty:
                yield train_df, test_df

            start += self.step_days
