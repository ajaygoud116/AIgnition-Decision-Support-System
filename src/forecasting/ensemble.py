import numpy as np

from src.models.common import QuantileValue


class EnsembleForecaster:
    """Weighted combination of LightGBM and Seasonal Naive forecasts.

    Both models predict the same quantiles; the ensemble is a convex
    combination per quantile.
    """

    def __init__(self, lgb_weight: float = 0.5, sn_weight: float = 0.5):
        self._lgb_weight = lgb_weight
        self._sn_weight = sn_weight

    def combine(
        self,
        lgb_preds: np.ndarray,
        sn_preds: np.ndarray,
    ) -> np.ndarray:
        """Combine two forecast arrays (both shape (n_dates, 3) or (n_dates,))."""
        if lgb_preds.ndim == 1:
            lgb_preds = lgb_preds.reshape(-1, 1)
        if sn_preds.ndim == 1:
            sn_preds = sn_preds.reshape(-1, 1)

        return self._lgb_weight * lgb_preds + self._sn_weight * sn_preds

    def to_quantile_value(self, ensemble_pred: np.ndarray) -> QuantileValue:
        """Convert a single 3-element ensemble prediction to QuantileValue."""
        return QuantileValue(
            p10=float(ensemble_pred[0]),
            p50=float(ensemble_pred[1]),
            p90=float(ensemble_pred[2]),
        )
