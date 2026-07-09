import numpy as np
import pandas as pd
import pytest

from src.forecasting.seasonal_naive import SeasonalNaiveForecaster


@pytest.fixture
def single_campaign_df():
    """10 days — enough for window=7, but <14 days so DOW pattern won't be used."""
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "date": dates,
        "campaign_id": ["g_1"] * 10,
        "revenue": [100 + i * 2 for i in range(10)],
    })


@pytest.fixture
def full_two_week_df():
    """14 days (2 complete weeks) to trigger DOW pattern."""
    dates = pd.date_range("2025-01-01", periods=14, freq="D")
    return pd.DataFrame({
        "date": dates,
        "campaign_id": ["g_1"] * 14,
        "revenue": [100 + i * 2 for i in range(14)],
    })


@pytest.fixture
def multi_campaign_df():
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=14, freq="D")
    rows = []
    for cid in ["g_1", "m_1"]:
        for i, d in enumerate(dates):
            rows.append({
                "date": d,
                "campaign_id": cid,
                "revenue": 100.0 + i * 2 + (10.0 if cid == "g_1" else 0.0),
            })
    return pd.DataFrame(rows)


class TestSeasonalNaiveForecaster:
    def test_fit_stores_all_campaigns(self, multi_campaign_df):
        sn = SeasonalNaiveForecaster(window=7)
        sn.fit(multi_campaign_df)
        assert len(sn._baselines) == 2
        assert "g_1" in sn._baselines
        assert "m_1" in sn._baselines

    def test_fit_baseline_is_last_7_day_average(self, single_campaign_df):
        sn = SeasonalNaiveForecaster(window=7)
        sn.fit(single_campaign_df)
        expected = single_campaign_df["revenue"].tail(7).mean()
        assert sn._baselines["g_1"] == pytest.approx(expected)

    def test_predict_uses_dow_pattern_when_available(self, single_campaign_df):
        sn = SeasonalNaiveForecaster(window=7)
        sn.fit(single_campaign_df)
        # 8 days covers at least 2 of each DOW
        future = pd.date_range("2025-01-15", periods=8, freq="D")
        preds = sn.predict("g_1", future)
        assert len(preds) == 8
        # Same DOW should give same prediction
        wed_preds = [preds[i] for i, d in enumerate(future) if d.dayofweek == 2]
        assert len(wed_preds) >= 2
        assert all(w == wed_preds[0] for w in wed_preds)

    def test_predict_dow_value_matches_last_observed(self, single_campaign_df):
        sn = SeasonalNaiveForecaster(window=7)
        sn.fit(single_campaign_df)
        # Jan 15, 2025 is Wednesday (dow=2); last Wed in data is Jan 8 → revenue=114
        future = pd.date_range("2025-01-15", periods=1, freq="D")
        preds = sn.predict("g_1", future)
        assert preds[0] == pytest.approx(114.0, abs=0.1)

    def test_dow_pattern_used_with_full_week(self, full_two_week_df):
        sn = SeasonalNaiveForecaster(window=7)
        sn.fit(full_two_week_df)
        assert len(sn._dow_patterns.get("g_1", {})) == 7

    def test_predict_missing_campaign_returns_zero(self):
        sn = SeasonalNaiveForecaster(window=7)
        future = pd.date_range("2025-01-15", periods=3, freq="D")
        preds = sn.predict("unknown", future)
        assert (preds == 0.0).all()

    def test_window_respected(self, single_campaign_df):
        sn = SeasonalNaiveForecaster(window=3)
        sn.fit(single_campaign_df)
        expected = single_campaign_df["revenue"].tail(3).mean()
        assert sn._baselines["g_1"] == pytest.approx(expected)

    def test_init_with_default_window(self):
        sn = SeasonalNaiveForecaster()
        assert sn._window == 7
