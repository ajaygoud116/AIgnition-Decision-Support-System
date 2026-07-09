import numpy as np
import pandas as pd
import pytest

from src.forecasting.lightgbm_model import LightGBMQuantileForecaster


@pytest.fixture
def feature_df():
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=n, freq="D"),
        "channel": ["google"] * n,
        "campaign_id": ["g_1"] * n,
        "campaign_name": ["Campaign_A"] * n,
        "campaign_type": ["SEARCH"] * n,
        "spend": np.linspace(50, 200, n) + np.random.normal(0, 5, n),
        "revenue": np.linspace(150, 600, n) + np.random.normal(0, 15, n),
        "clicks": (np.linspace(20, 80, n) + np.random.normal(0, 3, n)).astype(int),
        "impressions": (np.linspace(200, 800, n) + np.random.normal(0, 30, n)).astype(int),
        "conversions": np.linspace(2, 8, n) + np.random.normal(0, 0.5, n),
        "daily_budget": [100.0] * n,
        "dow": pd.date_range("2025-01-01", periods=n, freq="D").dayofweek,
        "month": pd.date_range("2025-01-01", periods=n, freq="D").month,
        "quarter": pd.date_range("2025-01-01", periods=n, freq="D").quarter,
        "doy": pd.date_range("2025-01-01", periods=n, freq="D").dayofyear,
        "woy": pd.date_range("2025-01-01", periods=n, freq="D").isocalendar().week.astype(int),
        "is_weekend": (pd.date_range("2025-01-01", periods=n, freq="D").dayofweek >= 5).astype(int),
        "roas": np.linspace(2.5, 3.5, n),
        "ctr": np.linspace(0.08, 0.12, n),
        "conv_rate": np.linspace(0.08, 0.12, n),
        "spend_per_click": np.linspace(2.0, 3.0, n),
        "revenue_per_impression": np.linspace(0.6, 0.9, n),
    })


class TestLightGBMQuantileForecaster:
    def test_fit_stores_models(self, feature_df):
        model = LightGBMQuantileForecaster()
        model.fit(feature_df)
        assert len(model._models) == 3
        for q in [0.1, 0.5, 0.9]:
            assert q in model._models

    def test_predict_returns_correct_shape(self, feature_df):
        model = LightGBMQuantileForecaster()
        model.fit(feature_df)
        preds = model.predict(feature_df.head(10))
        assert preds.shape == (10, 3)

    def test_predict_p50_between_p10_and_p90(self, feature_df):
        model = LightGBMQuantileForecaster()
        model.fit(feature_df)
        preds = model.predict(feature_df.head(10))
        assert (preds[:, 1] >= preds[:, 0]).all()
        assert (preds[:, 1] <= preds[:, 2]).all()

    def test_predict_returns_non_nan(self, feature_df):
        model = LightGBMQuantileForecaster()
        model.fit(feature_df)
        preds = model.predict(feature_df.head(10))
        assert not np.isnan(preds).any()

    def test_custom_params(self):
        model = LightGBMQuantileForecaster(
            params={"n_estimators": 50, "num_leaves": 15},
        )
        assert model._params["n_estimators"] == 50
        assert model._params["num_leaves"] == 15

    def test_custom_quantiles(self, feature_df):
        model = LightGBMQuantileForecaster(quantiles=[0.25, 0.5, 0.75])
        model.fit(feature_df)
        assert list(model._models.keys()) == [0.25, 0.5, 0.75]
