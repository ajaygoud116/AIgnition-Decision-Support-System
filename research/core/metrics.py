import numpy as np


class ForecastMetrics:
    """All evaluation metrics for forecasting experiments."""

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
        mask = np.abs(y_true) > eps
        if not mask.any():
            return 0.0
        return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-10) -> float:
        denom = np.abs(y_true) + np.abs(y_pred)
        denom = np.maximum(denom, eps)
        return float(np.mean(2.0 * np.abs(y_true - y_pred) / denom) * 100)

    @staticmethod
    def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float = 0.5) -> float:
        error = y_true - y_pred
        return float(np.mean(np.maximum(tau * error, (tau - 1.0) * error)))

    @staticmethod
    def coverage_90(y_true: np.ndarray, y_p10: np.ndarray, y_p90: np.ndarray) -> float:
        inside = np.logical_and(y_true >= y_p10, y_true <= y_p90)
        return float(np.mean(inside))

    @staticmethod
    def coverage(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray) -> float:
        inside = np.logical_and(y_true >= y_lower, y_true <= y_upper)
        return float(np.mean(inside))

    @staticmethod
    def interval_width(y_p10: np.ndarray, y_p90: np.ndarray) -> float:
        return float(np.mean(y_p90 - y_p10))

    @staticmethod
    def winkler_score(y_true: np.ndarray, y_lower: np.ndarray, y_upper: np.ndarray, alpha: float = 0.1) -> float:
        """Winkler score for prediction interval evaluation. Lower is better."""
        delta = y_upper - y_lower
        below = y_true < y_lower
        above = y_true > y_upper
        penalty = np.zeros_like(y_true)
        penalty[below] = (y_lower[below] - y_true[below]) / alpha
        penalty[above] = (y_true[above] - y_upper[above]) / alpha
        return float(np.mean(delta + penalty))

    @staticmethod
    def calibration_error(y_true: np.ndarray, y_pred: np.ndarray, quantile: float) -> float:
        """Absolute difference between empirical and nominal quantile coverage."""
        empirical = float(np.mean(y_true <= y_pred))
        return abs(empirical - quantile)

    def all_metrics(self, y_true: np.ndarray, y_pred_p50: np.ndarray,
                    y_pred_p10: np.ndarray = None, y_pred_p90: np.ndarray = None) -> dict:
        result = {
            "rmse": self.rmse(y_true, y_pred_p50),
            "mae": self.mae(y_true, y_pred_p50),
            "mape": self.mape(y_true, y_pred_p50),
            "smape": self.smape(y_true, y_pred_p50),
            "pinball_p50": self.pinball_loss(y_true, y_pred_p50, 0.5),
        }
        if y_pred_p10 is not None and y_pred_p90 is not None:
            result["pinball_p10"] = self.pinball_loss(y_true, y_pred_p10, 0.1)
            result["pinball_p90"] = self.pinball_loss(y_true, y_pred_p90, 0.9)
            result["coverage_90"] = self.coverage(y_true, y_pred_p10, y_pred_p90)
            result["interval_width"] = self.interval_width(y_pred_p10, y_pred_p90)
            result["winkler"] = self.winkler_score(y_true, y_pred_p10, y_pred_p90)
            result["calibration_error_p10"] = self.calibration_error(y_true, y_pred_p10, 0.1)
            result["calibration_error_p50"] = self.calibration_error(y_true, y_pred_p50, 0.5)
            result["calibration_error_p90"] = self.calibration_error(y_true, y_pred_p90, 0.9)
        return result
