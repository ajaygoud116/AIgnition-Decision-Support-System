from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.transforms import (
    add_lag_features,
    add_ratio_features,
    add_rolling_features,
    add_time_features,
)
from src.forecasting.ensemble import EnsembleForecaster
from src.forecasting.lightgbm_model import LightGBMQuantileForecaster
from src.forecasting.seasonal_naive import SeasonalNaiveForecaster
from src.models.common import (
    ForecastPoint,
    ForecastResult,
    ForecastSeries,
    Granularity,
    Horizon,
    MetricType,
    QuantileValue,
)
from src.utils.config import Config
from src.utils.logger import StructuredLogger


class Forecaster:
    """End-to-end forecast orchestrator.

    Trains LightGBM quantile + Seasonal Naive models, builds future
    feature frames per campaign, generates ensemble predictions, and
    assembles a ForecastResult.
    """

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("forecasting.forecaster")
        self._lgb = LightGBMQuantileForecaster(
            quantiles=config.get("forecast.quantiles", [0.1, 0.5, 0.9]),
            random_seed=config.get("project.random_seed", 42),
        )
        self._sn = SeasonalNaiveForecaster(window=7)
        self._ensemble = EnsembleForecaster(lgb_weight=0.5, sn_weight=0.5)
        self._horizons: List[int] = config.get("forecast.horizons", [30, 60, 90])
        self._min_history = config.get("forecast.min_history_days", 30)
        self._lookback = config.get("forecast.lookback_window", 90)
        self._fitted: bool = False

    def fit(self, feature_df: pd.DataFrame) -> "Forecaster":
        """Train both sub-models on the full feature DataFrame."""
        self._logger.info(
            "fit_start",
            rows=len(feature_df),
            campaigns=feature_df["campaign_id"].nunique(),
        )
        self._lgb.fit(feature_df)
        self._sn.fit(feature_df)
        self._fitted = True
        self._logger.info("fit_complete")
        return self

    def predict(self, feature_df: pd.DataFrame) -> ForecastResult:
        """Generate forecasts for all campaigns with sufficient history.

        Returns a ForecastResult with one ForecastSeries per
        (campaign_id, horizon, MetricType.REVENUE) combination.
        """
        if not self._fitted:
            raise RuntimeError("Forecaster must be fitted before predict()")

        series_list: List[ForecastSeries] = []

        last_train_date = feature_df["date"].max()

        for (cid, ch), group in (
            feature_df.sort_values(["campaign_id", "date"])
            .groupby(["campaign_id", "channel"], sort=False)
        ):
            if len(group) < self._min_history:
                self._logger.info("skip_campaign", campaign=cid, reason="insufficient_history")
                continue

            group = group.tail(self._lookback).reset_index(drop=True)

            for h in self._horizons:
                horizon_dates = pd.date_range(
                    start=last_train_date + pd.Timedelta(days=1),
                    periods=h,
                    freq="D",
                )

                try:
                    future_df = self._build_future_features(group, horizon_dates)
                except Exception:
                    self._logger.error("future_features_failed", campaign=cid, horizon=h)
                    continue

                lgb_preds = self._lgb.predict(future_df)
                sn_preds = self._sn.predict(cid, horizon_dates)
                sn_array = np.tile(sn_preds.reshape(-1, 1), (1, 3))
                combined = self._ensemble.combine(lgb_preds, sn_array)

                points = []
                for i in range(h):
                    row = combined[i]
                    sorted_ = np.sort(row)
                    sorted_ = np.maximum(sorted_, 0.0)
                    points.append(
                        ForecastPoint(
                            date=horizon_dates[i].date(),
                            values=QuantileValue(
                                p10=float(sorted_[0]),
                                p50=float(sorted_[1]),
                                p90=float(sorted_[2]),
                            ),
                        )
                    )

                series_list.append(
                    ForecastSeries(
                        entity_id=cid,
                        channel=ch,
                        granularity=Granularity.CAMPAIGN,
                        metric=MetricType.REVENUE,
                        horizon=Horizon(h),
                        points=points,
                    )
                )

        result = ForecastResult(
            series=series_list,
            metadata={
                "model": "lightgbm_quantile + seasonal_naive_ensemble",
                "horizons": self._horizons,
                "last_train_date": str(last_train_date.date()),
                "campaigns_forecasted": len(set(s.entity_id for s in series_list)),
            },
        )

        self._logger.info(
            "predict_complete",
            series=len(series_list),
            campaigns=result.metadata["campaigns_forecasted"],
        )
        return result

    def _build_future_features(
        self,
        campaign_group: pd.DataFrame,
        future_dates: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """Build a feature DataFrame for future dates by recomputing
        features over the extended history (historical + flat future).

        Raw metrics are forward-filled from the last observation, then
        all transforms (time, ratio, rolling, lag) are reapplied so
        that derived features evolve across the forecast horizon.
        """
        base_cols = [
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        ]
        present_cols = [c for c in base_cols if c in campaign_group.columns]

        last_row = campaign_group.iloc[-1:][present_cols].copy()

        future = pd.DataFrame({"date": future_dates})
        for col in present_cols:
            if col == "date":
                continue
            future[col] = last_row[col].iloc[0]

        combined = pd.concat([campaign_group[present_cols], future], ignore_index=True)

        combined = add_time_features(combined)
        combined = add_ratio_features(combined)
        combined = add_rolling_features(
            combined, windows=self._config.get("features.rolling_windows", [7, 14, 30])
        )
        combined = add_lag_features(
            combined, lags=self._config.get("features.lag_windows", [1, 7, 14, 30])
        )

        future_start = len(campaign_group)
        return combined.iloc[future_start:].reset_index(drop=True)

    def get_lgb_model(self) -> LightGBMQuantileForecaster:
        return self._lgb

    def get_sn_model(self) -> SeasonalNaiveForecaster:
        return self._sn

    def get_feature_columns(self) -> List[str]:
        return list(self._lgb._feature_columns) if self._lgb._feature_columns else []
