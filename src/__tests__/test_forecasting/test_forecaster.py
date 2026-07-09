import numpy as np
import pandas as pd
import pytest

from src.forecasting.forecaster import Forecaster
from src.models.common import ForecastResult, Granularity, Horizon, MetricType
from src.utils.config import Config


@pytest.fixture
def multi_campaign_feature_df():
    """Two campaigns with 100 days of feature-rich data."""
    np.random.seed(42)
    rows = []
    for cid, ch, ctype in [("g_1", "google", "SEARCH"), ("g_2", "google", "DISPLAY")]:
        for i in range(100):
            d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
            rows.append({
                "date": d,
                "channel": ch,
                "campaign_id": cid,
                "campaign_name": f"{ctype}_Campaign",
                "campaign_type": ctype,
                "spend": 50.0 + i * 1.5 + np.random.normal(0, 3),
                "revenue": 150.0 + i * 4.5 + np.random.normal(0, 10),
                "clicks": int(20 + i * 0.8 + np.random.normal(0, 2)),
                "impressions": int(200 + i * 6 + np.random.normal(0, 20)),
                "conversions": 2.0 + i * 0.1 + np.random.normal(0, 0.3),
                "daily_budget": 100.0,
                "dow": d.dayofweek,
                "month": d.month,
                "quarter": d.quarter,
                "doy": d.dayofyear,
                "woy": int(d.isocalendar().week),
                "is_weekend": int(d.dayofweek >= 5),
                "roas": 3.0 + i * 0.01,
                "ctr": 0.1 + i * 0.0005,
                "conv_rate": 0.1 + i * 0.0002,
                "spend_per_click": 2.5 + i * 0.01,
                "revenue_per_impression": 0.75 + i * 0.002,
            })
    df = pd.DataFrame(rows)
    for m in ["spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"]:
        for w in [7, 14, 30]:
            df[f"{m}_rolling_mean_{w}"] = df.groupby("campaign_id")[m].transform(
                lambda x: x.shift(1).rolling(w, min_periods=2).mean()
            )
            df[f"{m}_rolling_std_{w}"] = df.groupby("campaign_id")[m].transform(
                lambda x: x.shift(1).rolling(w, min_periods=2).std().fillna(0.0)
            )
        for lag in [1, 7, 14, 30]:
            df[f"{m}_lag_{lag}"] = df.groupby("campaign_id")[m].transform(
                lambda x: x.shift(lag)
            )
    return df


@pytest.fixture
def forecaster():
    return Forecaster(Config())


class TestForecaster:
    def test_fit_returns_self(self, forecaster, multi_campaign_feature_df):
        result = forecaster.fit(multi_campaign_feature_df)
        assert result is forecaster
        assert forecaster._fitted

    def test_predict_returns_forecast_result(self, forecaster, multi_campaign_feature_df):
        forecaster.fit(multi_campaign_feature_df)
        result = forecaster.predict(multi_campaign_feature_df)
        assert isinstance(result, ForecastResult)

    def test_predict_has_series_for_each_campaign_horizon(self, forecaster, multi_campaign_feature_df):
        forecaster.fit(multi_campaign_feature_df)
        result = forecaster.predict(multi_campaign_feature_df)
        expected_series = 2 * 3  # 2 campaigns × 3 horizons
        assert len(result.series) == expected_series

    def test_series_have_correct_structure(self, forecaster, multi_campaign_feature_df):
        forecaster.fit(multi_campaign_feature_df)
        result = forecaster.predict(multi_campaign_feature_df)
        s = result.series[0]
        assert isinstance(s.entity_id, str)
        assert isinstance(s.channel, str)
        assert s.granularity == Granularity.CAMPAIGN
        assert s.metric == MetricType.REVENUE
        assert isinstance(s.horizon, Horizon)

    def test_series_points_count_matches_horizon(self, forecaster, multi_campaign_feature_df):
        forecaster.fit(multi_campaign_feature_df)
        result = forecaster.predict(multi_campaign_feature_df)
        for s in result.series:
            assert len(s.points) == s.horizon.value

    def test_quantile_order_in_each_point(self, forecaster, multi_campaign_feature_df):
        forecaster.fit(multi_campaign_feature_df)
        result = forecaster.predict(multi_campaign_feature_df)
        for s in result.series:
            for p in s.points:
                assert p.values.p10 <= p.values.p50 <= p.values.p90

    def test_metadata_present(self, forecaster, multi_campaign_feature_df):
        forecaster.fit(multi_campaign_feature_df)
        result = forecaster.predict(multi_campaign_feature_df)
        assert result.metadata is not None
        assert "model" in result.metadata
        assert "horizons" in result.metadata
        assert "campaigns_forecasted" in result.metadata

    def test_predict_without_fit_raises(self, forecaster):
        df = pd.DataFrame({"campaign_id": ["g_1"], "date": [pd.Timestamp("2025-01-01")]})
        with pytest.raises(RuntimeError, match="must be fitted"):
            forecaster.predict(df)

    def test_insufficient_history_skipped(self):
        cfg = Config()
        fc = Forecaster(cfg)
        rows = []
        for i in range(100):
            d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
            rows.append({
                "date": d,
                "channel": "google",
                "campaign_id": "g_full" if i < 90 else "g_short",
                "campaign_name": "Test",
                "campaign_type": "SEARCH",
                "spend": 50.0 + i * 0.5,
                "revenue": 150.0 + i * 1.5,
                "clicks": 20 + i,
                "impressions": 200 + i * 2,
                "conversions": 2.0 + i * 0.1,
                "daily_budget": 100.0,
                "dow": d.dayofweek,
                "month": d.month,
                "quarter": d.quarter,
                "doy": d.dayofyear,
                "woy": int(d.isocalendar().week),
                "is_weekend": int(d.dayofweek >= 5),
                "roas": 3.0,
                "ctr": 0.1,
                "conv_rate": 0.1,
                "spend_per_click": 2.5,
                "revenue_per_impression": 0.75,
            })
        df = pd.DataFrame(rows)
        for m in ["spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"]:
            for w in [7, 14, 30]:
                df[f"{m}_rolling_mean_{w}"] = df.groupby("campaign_id")[m].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=2).mean()
                )
                df[f"{m}_rolling_std_{w}"] = df.groupby("campaign_id")[m].transform(
                    lambda x: x.shift(1).rolling(w, min_periods=2).std().fillna(0.0)
                )
            for lag in [1, 7, 14, 30]:
                df[f"{m}_lag_{lag}"] = df.groupby("campaign_id")[m].transform(
                    lambda x: x.shift(lag)
                )
        fc.fit(df)
        result = fc.predict(df)
        # Only g_full (90 days) meets min_history_days=30; g_short (10 days) is skipped
        forecasted_campaigns = set(s.entity_id for s in result.series)
        assert "g_full" in forecasted_campaigns
        assert "g_short" not in forecasted_campaigns

    def test_get_lgb_model(self, forecaster, multi_campaign_feature_df):
        forecaster.fit(multi_campaign_feature_df)
        from src.forecasting.lightgbm_model import LightGBMQuantileForecaster
        assert isinstance(forecaster.get_lgb_model(), LightGBMQuantileForecaster)

    def test_get_sn_model(self, forecaster, multi_campaign_feature_df):
        forecaster.fit(multi_campaign_feature_df)
        from src.forecasting.seasonal_naive import SeasonalNaiveForecaster
        assert isinstance(forecaster.get_sn_model(), SeasonalNaiveForecaster)

    def test_get_feature_columns(self, forecaster, multi_campaign_feature_df):
        forecaster.fit(multi_campaign_feature_df)
        cols = forecaster.get_feature_columns()
        assert len(cols) > 0
        assert "dow" in cols
        assert "roas" in cols
