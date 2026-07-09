import pandas as pd
import pytest

from src.validation.checks import (
    check_budget_consistency,
    check_channel_coverage,
    check_duplicate_rows,
    check_future_dates,
    check_invalid_campaign_types,
    check_missing_values,
    check_negative_spend,
    check_zero_activity,
)


class TestCheckMissingValues:
    def test_no_missing_values(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        issues, result = check_missing_values(df)
        assert len(issues) == 0

    def test_warning_below_threshold(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [4.0, 5.0, 6.0]})
        issues, result = check_missing_values(df, max_ratio=0.5)
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].count == 1

    def test_error_above_threshold(self):
        df = pd.DataFrame({"a": [1.0, None, None], "b": [4.0, 5.0, 6.0]})
        issues, result = check_missing_values(df, max_ratio=0.3)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert issues[0].count == 2

    def test_multiple_columns_missing(self):
        df = pd.DataFrame({"a": [None, None], "b": [1.0, None]})
        issues, result = check_missing_values(df, max_ratio=0.1)
        assert len(issues) == 2


class TestCheckNegativeSpend:
    def test_no_negative_spend(self):
        df = pd.DataFrame({"spend": [1.0, 2.0, 3.0]})
        issues, result = check_negative_spend(df)
        assert len(issues) == 0

    def test_negative_spend_clamped(self):
        df = pd.DataFrame({"spend": [1.0, -5.0, 3.0, -2.0]})
        issues, result = check_negative_spend(df)
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].count == 2
        assert (result["spend"] >= 0).all()
        assert result["spend"].iloc[1] == 0.0
        assert result["spend"].iloc[3] == 0.0

    def test_no_spend_column(self):
        df = pd.DataFrame({"revenue": [1.0, 2.0]})
        issues, result = check_negative_spend(df)
        assert len(issues) == 0


class TestCheckDuplicateRows:
    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        issues, result = check_duplicate_rows(df)
        assert len(issues) == 0
        assert len(result) == 2

    def test_duplicates_removed_warning(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        issues, result = check_duplicate_rows(df, max_duplicate_ratio=0.5)
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].count == 1
        assert len(result) == 2

    def test_duplicates_removed_error(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
        issues, result = check_duplicate_rows(df, max_duplicate_ratio=0.3)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert len(result) == 1

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        issues, result = check_duplicate_rows(df)
        assert len(issues) == 0


class TestCheckFutureDates:
    def test_no_future_dates(self):
        df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]})
        issues, result = check_future_dates(df)
        assert len(issues) == 0

    def test_future_dates_removed(self):
        df = pd.DataFrame({
            "date": [
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2025-01-01"),
            ]
        })
        issues, result = check_future_dates(
            df, reference_date=pd.Timestamp("2024-06-01")
        )
        assert len(issues) == 1
        assert issues[0].count == 1
        assert len(result) == 1

    def test_no_date_column(self):
        df = pd.DataFrame({"a": [1, 2]})
        issues, result = check_future_dates(df)
        assert len(issues) == 0


class TestCheckInvalidCampaignTypes:
    def test_all_valid(self):
        df = pd.DataFrame({
            "channel": ["google", "google"],
            "campaign_type": ["SEARCH", "DISPLAY"],
        })
        issues, result = check_invalid_campaign_types(
            df, valid_types={"google": ["SEARCH", "DISPLAY"]}
        )
        assert len(issues) == 0

    def test_invalid_types_found(self):
        df = pd.DataFrame({
            "channel": ["google", "google"],
            "campaign_type": ["SEARCH", "UNKNOWN_TYPE"],
        })
        issues, result = check_invalid_campaign_types(
            df, valid_types={"google": ["SEARCH", "DISPLAY"]}
        )
        assert len(issues) == 1
        assert issues[0].count == 1
        assert "UNKNOWN_TYPE" in issues[0].message

    def test_no_channel_column(self):
        df = pd.DataFrame({"campaign_type": ["SEARCH"]})
        issues, result = check_invalid_campaign_types(df)
        assert len(issues) == 0

    def test_no_types_dict(self):
        df = pd.DataFrame({
            "channel": ["google"],
            "campaign_type": ["SEARCH"],
        })
        issues, result = check_invalid_campaign_types(df)
        assert len(issues) == 0


class TestCheckBudgetConsistency:
    def test_spend_within_budget(self):
        df = pd.DataFrame({"spend": [50.0, 80.0], "daily_budget": [100.0, 100.0]})
        issues, result = check_budget_consistency(df)
        assert len(issues) == 0

    def test_spend_exceeds_budget(self):
        df = pd.DataFrame({"spend": [150.0, 80.0], "daily_budget": [100.0, 100.0]})
        issues, result = check_budget_consistency(df)
        assert len(issues) == 1
        assert issues[0].severity == "warning"
        assert issues[0].count == 1

    def test_zero_budget_ignored(self):
        df = pd.DataFrame({"spend": [50.0, 0.0], "daily_budget": [0.0, 0.0]})
        issues, result = check_budget_consistency(df)
        assert len(issues) == 0


class TestCheckZeroActivity:
    def test_all_active(self):
        df = pd.DataFrame({
            "spend": [10.0, 20.0],
            "impressions": [100, 200],
        })
        issues, result = check_zero_activity(df)
        assert len(issues) == 0

    def test_zero_activity_detected(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            "campaign_id": ["g_1", "g_2"],
            "channel": ["google", "google"],
            "spend": [0.0, 20.0],
            "impressions": [0, 200],
        })
        issues, result = check_zero_activity(df)
        assert len(issues) == 1
        assert issues[0].count == 1


class TestCheckChannelCoverage:
    def test_all_channels_present(self):
        df = pd.DataFrame({
            "channel": ["google", "meta", "bing"],
        })
        issues, result = check_channel_coverage(df)
        assert len(issues) == 1
        assert issues[0].severity == "warning"

    def test_missing_channel(self):
        df = pd.DataFrame({
            "channel": ["google", "meta"],
        })
        issues, result = check_channel_coverage(df)
        assert len(issues) == 1
        assert issues[0].severity == "error"
        assert "bing" in issues[0].message

    def test_multiple_missing_channels(self):
        df = pd.DataFrame({"channel": ["google"]})
        issues, result = check_channel_coverage(
            df, expected_channels=["google", "meta", "bing"]
        )
        assert len(issues) == 2
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 2
        assert all("bing" in i.message or "meta" in i.message for i in errors)

    def test_no_channel_column(self):
        df = pd.DataFrame({"a": [1]})
        issues, result = check_channel_coverage(df)
        assert len(issues) == 1
        assert issues[0].severity == "error"
