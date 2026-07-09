import pandas as pd
import pytest

from src.features.builder import FeatureBuilder
from src.utils.config import Config


@pytest.fixture
def campaign_df():
    """Two campaigns with 15 days of history each."""
    rows = []
    for cid, ch, ctype in [("g_1", "google", "SEARCH"), ("g_2", "google", "DISPLAY")]:
        for day_offset in range(15):
            d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day_offset)
            rows.append({
                "date": d,
                "channel": ch,
                "campaign_id": cid,
                "campaign_name": f"{ctype}_Campaign",
                "campaign_type": ctype,
                "spend": 50.0 + day_offset * 1.0,
                "revenue": 150.0 + day_offset * 3.0,
                "clicks": 20 + day_offset * 2,
                "impressions": 200 + day_offset * 10,
                "conversions": 2.0 + day_offset * 0.2,
                "daily_budget": 100.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def builder():
    return FeatureBuilder(Config())


class TestFeatureBuilder:
    def test_returns_dataframe(self, builder, campaign_df):
        result = builder.build(campaign_df)
        assert isinstance(result, pd.DataFrame)

    def test_feature_columns_added(self, builder, campaign_df):
        result = builder.build(campaign_df)
        base = {
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        }
        feature_cols = set(result.columns) - base
        assert len(feature_cols) > 0

    def test_time_features_present(self, builder, campaign_df):
        result = builder.build(campaign_df)
        for col in ["dow", "month", "quarter", "doy", "woy", "is_weekend"]:
            assert col in result.columns

    def test_ratio_features_present(self, builder, campaign_df):
        result = builder.build(campaign_df)
        for col in ["roas", "ctr", "conv_rate", "spend_per_click", "revenue_per_impression"]:
            assert col in result.columns

    def test_rolling_features_present(self, builder, campaign_df):
        result = builder.build(campaign_df)
        for m in ["spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"]:
            assert f"{m}_rolling_mean_7" in result.columns
            assert f"{m}_rolling_mean_14" in result.columns
            assert f"{m}_rolling_std_7" in result.columns

    def test_lag_features_present(self, builder, campaign_df):
        result = builder.build(campaign_df)
        for m in ["spend", "revenue", "clicks", "impressions", "conversions", "daily_budget"]:
            assert f"{m}_lag_1" in result.columns
            assert f"{m}_lag_7" in result.columns
            assert f"{m}_lag_14" in result.columns

    def test_rows_dropped_for_na(self, builder, campaign_df):
        result = builder.build(campaign_df)
        # First rows of each campaign will have NaN in lag/rolling features
        # so they get dropped by the max_na_ratio filter
        assert len(result) < len(campaign_df)

    def test_both_campaigns_retained(self, builder, campaign_df):
        result = builder.build(campaign_df)
        assert result["campaign_id"].nunique() == 2

    def test_original_columns_preserved(self, builder, campaign_df):
        result = builder.build(campaign_df)
        for col in campaign_df.columns:
            assert col in result.columns

    def test_config_controls_windows(self):
        config = Config()
        builder = FeatureBuilder(config)
        assert builder._rolling_windows == [7, 14, 30]
        assert builder._lag_windows == [1, 7, 14, 30]
        assert builder._max_na_ratio == 0.3

    def test_single_row_per_campaign(self, builder):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01")],
            "channel": ["google", "meta"],
            "campaign_id": ["g_1", "m_1"],
            "campaign_name": ["Campaign_A", "Campaign_B"],
            "campaign_type": ["SEARCH", "Generic"],
            "spend": [50.0, 30.0],
            "revenue": [150.0, 90.0],
            "clicks": [20, 15],
            "impressions": [200, 150],
            "conversions": [2.0, 1.5],
            "daily_budget": [100.0, 100.0],
        })
        result = builder.build(df)
        assert len(result) == 0  # All rows dropped due to NaN from lag/rolling
