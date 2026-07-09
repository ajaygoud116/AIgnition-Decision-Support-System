import numpy as np
import pandas as pd
import pytest

from src.features.transforms import (
    METRICS_FOR_FEATURES,
    add_lag_features,
    add_ratio_features,
    add_rolling_features,
    add_time_features,
    drop_high_na_rows,
)


@pytest.fixture
def single_campaign_df():
    return pd.DataFrame({
        "date": pd.date_range("2025-01-01", periods=10, freq="D"),
        "spend": [10.0, 20.0, 15.0, 30.0, 25.0, 40.0, 35.0, 50.0, 45.0, 60.0],
        "revenue": [30.0, 60.0, 45.0, 90.0, 75.0, 120.0, 105.0, 150.0, 135.0, 180.0],
        "clicks": [10, 20, 15, 30, 25, 40, 35, 50, 45, 60],
        "impressions": [100, 200, 150, 300, 250, 400, 350, 500, 450, 600],
        "conversions": [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 3.5, 5.0, 4.5, 6.0],
        "daily_budget": [100.0] * 10,
    })


class TestAddTimeFeatures:
    def test_adds_time_columns(self, single_campaign_df):
        result = add_time_features(single_campaign_df)
        assert "dow" in result.columns
        assert "month" in result.columns
        assert "quarter" in result.columns
        assert "doy" in result.columns
        assert "woy" in result.columns
        assert "is_weekend" in result.columns

    def test_is_weekend_jan_4_2025_is_saturday(self):
        df = pd.DataFrame({"date": [pd.Timestamp("2025-01-04")]})
        result = add_time_features(df)
        assert result["is_weekend"].iloc[0] == 1

    def test_is_weekend_jan_6_2025_is_monday(self):
        df = pd.DataFrame({"date": [pd.Timestamp("2025-01-06")]})
        result = add_time_features(df)
        assert result["is_weekend"].iloc[0] == 0

    def test_dow_range(self, single_campaign_df):
        result = add_time_features(single_campaign_df)
        assert result["dow"].between(0, 6).all()

    def test_month_range(self, single_campaign_df):
        result = add_time_features(single_campaign_df)
        assert result["month"].between(1, 12).all()

    def test_woy_is_int(self, single_campaign_df):
        result = add_time_features(single_campaign_df)
        assert result["woy"].dtype == int


class TestAddRatioFeatures:
    def test_roas_correct(self, single_campaign_df):
        result = add_ratio_features(single_campaign_df)
        expected = single_campaign_df["revenue"] / single_campaign_df["spend"]
        pd.testing.assert_series_equal(result["roas"], expected, check_names=False)

    def test_ctr_correct(self, single_campaign_df):
        result = add_ratio_features(single_campaign_df)
        expected = single_campaign_df["clicks"] / single_campaign_df["impressions"]
        pd.testing.assert_series_equal(result["ctr"], expected, check_names=False)

    def test_conv_rate_correct(self, single_campaign_df):
        result = add_ratio_features(single_campaign_df)
        expected = single_campaign_df["conversions"] / single_campaign_df["clicks"]
        pd.testing.assert_series_equal(result["conv_rate"], expected, check_names=False)

    def test_spend_per_click_correct(self, single_campaign_df):
        result = add_ratio_features(single_campaign_df)
        expected = single_campaign_df["spend"] / single_campaign_df["clicks"]
        pd.testing.assert_series_equal(result["spend_per_click"], expected, check_names=False)

    def test_revenue_per_impression_correct(self, single_campaign_df):
        result = add_ratio_features(single_campaign_df)
        expected = single_campaign_df["revenue"] / single_campaign_df["impressions"]
        pd.testing.assert_series_equal(result["revenue_per_impression"], expected, check_names=False)

    def test_zero_spend_roas_is_zero(self):
        df = pd.DataFrame({
            "spend": [0.0],
            "revenue": [100.0],
            "clicks": [10],
            "impressions": [100],
            "conversions": [1.0],
        })
        result = add_ratio_features(df)
        assert result["roas"].iloc[0] == 0.0

    def test_zero_impressions_ctr_is_zero(self):
        df = pd.DataFrame({
            "spend": [10.0],
            "revenue": [100.0],
            "clicks": [0],
            "impressions": [0],
            "conversions": [0.0],
        })
        result = add_ratio_features(df)
        assert result["ctr"].iloc[0] == 0.0
        assert result["conv_rate"].iloc[0] == 0.0


class TestAddRollingFeatures:
    def test_rolling_mean_added_for_each_metric_and_window(self, single_campaign_df):
        result = add_rolling_features(single_campaign_df, windows=[3, 7])
        for m in METRICS_FOR_FEATURES:
            assert f"{m}_rolling_mean_3" in result.columns
            assert f"{m}_rolling_mean_7" in result.columns

    def test_rolling_std_added(self, single_campaign_df):
        result = add_rolling_features(single_campaign_df, windows=[3])
        for m in METRICS_FOR_FEATURES:
            assert f"{m}_rolling_std_3" in result.columns

    def test_first_row_is_nan_for_rolling(self, single_campaign_df):
        result = add_rolling_features(single_campaign_df, windows=[3], min_periods=2)
        assert pd.isna(result["spend_rolling_mean_3"].iloc[0])

    def test_rolling_uses_shifted_values(self, single_campaign_df):
        result = add_rolling_features(single_campaign_df, windows=[2], min_periods=1)
        # Row 2: mean of row 0 and row 1 values (shifted by 1)
        expected = (single_campaign_df["spend"].iloc[0] + single_campaign_df["spend"].iloc[1]) / 2
        assert result["spend_rolling_mean_2"].iloc[2] == pytest.approx(expected)

    def test_missing_metric_skipped(self, single_campaign_df):
        df = single_campaign_df.drop(columns=["daily_budget"])
        result = add_rolling_features(df, windows=[3])
        assert "daily_budget_rolling_mean_3" not in result.columns


class TestAddLagFeatures:
    def test_lag_added_for_each_metric_and_lag(self, single_campaign_df):
        result = add_lag_features(single_campaign_df, lags=[1, 7])
        for m in METRICS_FOR_FEATURES:
            assert f"{m}_lag_1" in result.columns
            assert f"{m}_lag_7" in result.columns

    def test_lag_1_equals_previous_value(self, single_campaign_df):
        result = add_lag_features(single_campaign_df, lags=[1])
        assert result["spend_lag_1"].iloc[1] == single_campaign_df["spend"].iloc[0]
        assert result["spend_lag_1"].iloc[5] == single_campaign_df["spend"].iloc[4]

    def test_lag_1_first_row_is_nan(self, single_campaign_df):
        result = add_lag_features(single_campaign_df, lags=[1])
        assert pd.isna(result["spend_lag_1"].iloc[0])

    def test_lag_7_first_seven_rows_are_nan(self, single_campaign_df):
        result = add_lag_features(single_campaign_df, lags=[7])
        for i in range(7):
            assert pd.isna(result["spend_lag_7"].iloc[i])
        assert not pd.isna(result["spend_lag_7"].iloc[7])

    def test_missing_metric_skipped(self, single_campaign_df):
        df = single_campaign_df.drop(columns=["daily_budget"])
        result = add_lag_features(df, lags=[1])
        assert "daily_budget_lag_1" not in result.columns


class TestDropHighNARows:
    def test_no_na_rows_unchanged(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = drop_high_na_rows(df, max_na_ratio=0.5)
        assert len(result) == 3

    def test_high_na_row_dropped(self):
        df = pd.DataFrame({
            "a": [1.0, None, None],
            "b": [4.0, None, 6.0],
        })
        result = drop_high_na_rows(df, max_na_ratio=0.3)
        assert len(result) == 1

    def test_max_na_ratio_one_keeps_all(self):
        df = pd.DataFrame({"a": [None, None], "b": [1.0, 2.0]})
        result = drop_high_na_rows(df, max_na_ratio=1.0)
        assert len(result) == 2

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        result = drop_high_na_rows(df)
        assert len(result) == 0
