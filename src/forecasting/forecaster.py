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
        """Generate baseline forecasts for all campaigns.

        Returns a ForecastResult with one ForecastSeries per
        (campaign_id, horizon, MetricType.REVENUE) combination.
        """
        if not self._fitted:
            raise RuntimeError("Forecaster must be fitted before predict()")
        last_train_date = feature_df["date"].max()
        series_list = self._predict_all(feature_df, last_train_date, scenario_overrides=None)

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

    def predict_scenario(
        self,
        feature_df: pd.DataFrame,
        budget_map: Dict[str, float],
    ) -> ForecastResult:
        """Generate budget-conditioned forecasts.

        For each campaign whose daily_budget is overridden in *budget_map*,
        the future feature frame is rebuilt with the scenario budget and
        a proportional spend multiplier.  Campaigns not in *budget_map*
        use their baseline (forward-filled) budget.

        Returns a ForecastResult (same structure as predict()).
        """
        if not self._fitted:
            raise RuntimeError("Forecaster must be fitted before predict_scenario()")
        last_train_date = feature_df["date"].max()

        # Build a per-campaign scenario overrides dict
        scenario_map: Dict[str, Optional[Dict[str, float]]] = {}
        for (cid, ch), group in (
            feature_df.sort_values(["campaign_id", "date"])
            .groupby(["campaign_id", "channel"], sort=False)
        ):
            if cid in budget_map:
                new_budget = budget_map[cid]
                last_budget = group["daily_budget"].iloc[-1]
                mult = new_budget / last_budget if last_budget > 0 else 1.0
                mult = max(0.1, min(5.0, mult))  # clamp at reasonable bounds
                scenario_map[cid] = {"daily_budget": new_budget, "spend_multiplier": mult}
            else:
                scenario_map[cid] = None

        series_list = self._predict_all(
            feature_df, last_train_date, scenario_map=scenario_map,
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
            "predict_scenario_complete",
            series=len(series_list),
            overrides=len(budget_map),
        )
        return result

    def _predict_all(
        self,
        feature_df: pd.DataFrame,
        last_train_date: pd.Timestamp,
        scenario_overrides: Optional[Dict[str, float]] = None,
        scenario_map: Optional[Dict[str, Optional[Dict[str, float]]]] = None,
    ) -> List[ForecastSeries]:
        """Shared predict logic used by predict() and predict_scenario()."""
        series_list: List[ForecastSeries] = []

        for (cid, ch), group in (
            feature_df.sort_values(["campaign_id", "date"])
            .groupby(["campaign_id", "channel"], sort=False)
        ):
            if len(group) < self._min_history:
                self._logger.info("skip_campaign", campaign=cid, reason="insufficient_history")
                continue

            group = group.tail(self._lookback).reset_index(drop=True)

            # Resolve per-campaign overrides if scenario_map is provided
            overrides = scenario_overrides
            if scenario_map is not None:
                overrides = scenario_map.get(cid, scenario_overrides)

            for h in self._horizons:
                horizon_dates = pd.date_range(
                    start=last_train_date + pd.Timedelta(days=1),
                    periods=h,
                    freq="D",
                )

                try:
                    future_df = self._build_future_features(group, horizon_dates, overrides)
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

        return series_list

    def _build_future_features(
        self,
        campaign_group: pd.DataFrame,
        future_dates: pd.DatetimeIndex,
        scenario_overrides: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """Build a feature DataFrame for future dates by recomputing
        features over the extended history (historical + expected future).

        Raw metrics for future dates are set to their expected day-of-week
        value (mean over the last N weeks of actual data), not forward-filled
        from the last observation.  This preserves temporal diversity in
        rolling, lag, and ratio features across the forecast horizon.

        Static identifiers (channel, campaign_id, campaign_name, campaign_type)
        are forward-filled.  daily_budget is forward-filled from the last
        observed value (it is a controlled variable).

        Optional scenario_overrides can supply alternate spend / budget
        values for budget-conditioned forecasting.

        All feature transforms (time, ratio, rolling, lag) are recomputed
        on the combined historical + future DataFrame.
        """
        base_cols = [
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        ]
        present_cols = [c for c in base_cols if c in campaign_group.columns]

        future = pd.DataFrame({"date": future_dates})
        future["dow"] = future["date"].dt.dayofweek

        # ── 1. Metrics that are forward-filled (static identifiers) ──────────
        last_row = campaign_group.iloc[-1:][present_cols].copy()
        for col in ["channel", "campaign_id", "campaign_name", "campaign_type"]:
            if col in present_cols:
                future[col] = last_row[col].iloc[0]

        # daily_budget is a controlled variable — forward-fill (or override)
        if "daily_budget" in present_cols:
            future["daily_budget"] = last_row["daily_budget"].iloc[0]

        # ── 2. Metrics that use expected weekday values ─────────────────────
        dynamic_metrics = ["spend", "revenue", "clicks", "impressions", "conversions"]
        dynamic_metrics = [m for m in dynamic_metrics if m in present_cols]

        # Compute DOW averages from the last N weeks of the campaign_group
        # Use at least 14 days of history, falling back to campaign_group tail
        min_history = max(14, len(campaign_group))
        tail_rows = campaign_group.tail(min_history).copy()
        tail_rows["_dow"] = tail_rows["date"].dt.dayofweek
        dow_means = tail_rows.groupby("_dow")[dynamic_metrics].mean().add_suffix("_dow_mean")

        for metric in dynamic_metrics:
            col_mean = f"{metric}_dow_mean"
            if col_mean in dow_means.columns:
                # Map future DOW to the expected value
                dow_map = dow_means[col_mean].to_dict()
                future[metric] = future["dow"].map(dow_map).fillna(dow_means[col_mean].mean())
            else:
                # Fallback: forward-fill if DOW computation fails
                future[metric] = last_row[metric].iloc[0]

        # ── 3. Apply scenario overrides (budget-conditioned forecasting) ─────
        if scenario_overrides:
            spend_mult = scenario_overrides.get("spend_multiplier")
            for key, val in scenario_overrides.items():
                if key == "spend_multiplier":
                    continue
                if key in future.columns:
                    future[key] = val
            if spend_mult is not None and "spend" in future.columns:
                future["spend"] = future["spend"] * spend_mult

        future = future.drop(columns=["dow"])

        # ── 4. Concat with historical, recompute all transforms ──────────────
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
