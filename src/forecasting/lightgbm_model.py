from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.utils.logger import StructuredLogger


class LightGBMQuantileForecaster:
    """Three LightGBM quantile regressors (p10, p50, p90) trained on the full
    feature set to predict revenue."""

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        params: Optional[Dict[str, Any]] = None,
        random_seed: int = 42,
    ):
        self._quantiles = quantiles or [0.1, 0.5, 0.9]
        self._models: Dict[float, lgb.LGBMRegressor] = {}
        self._feature_columns: List[str] = []
        self._logger = StructuredLogger("forecasting.lgb")

        default_params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
            "random_state": random_seed,
        }
        if params:
            default_params.update(params)
        self._params = default_params

    def fit(self, df: pd.DataFrame, target_col: str = "revenue") -> "LightGBMQuantileForecaster":
        """Train three quantile regressors on the feature DataFrame."""
        self._feature_columns = [c for c in df.columns if c not in self._base_columns(df)]
        self._feature_columns = [c for c in self._feature_columns if c != target_col]

        X = df[self._feature_columns].values
        y = df[target_col].values

        self._logger.info(
            "lgb_training_start",
            samples=len(X),
            features=len(self._feature_columns),
        )

        for q in self._quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                **self._params,
            )
            model.fit(X, y)
            self._models[q] = model

        self._logger.info("lgb_training_complete", models=len(self._models))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return predictions of shape (n, 3) — one column per quantile.

        The caller must ensure df contains self._feature_columns.
        """
        X = df[self._feature_columns].values
        results = np.column_stack([self._models[q].predict(X) for q in self._quantiles])
        return results

    @staticmethod
    def _base_columns(df: pd.DataFrame) -> set:
        base = {
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        }
        return {c for c in base if c in df.columns}
