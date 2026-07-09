import pandas as pd
import pytest

from src.utils.config import Config
from src.validation.validator import ValidationEngine


@pytest.fixture
def clean_unified_df():
    """A clean unified DataFrame representing 2 campaigns across all channels."""
    rows = []
    for channel, cid, ctype in [
        ("google", "g_1", "SEARCH"),
        ("google", "g_2", "DISPLAY"),
        ("meta", "m_1", "Generic"),
        ("meta", "m_2", "Prospecting"),
        ("bing", "b_1", "Search"),
        ("bing", "b_2", "Shopping"),
    ]:
        for day_offset in range(60):
            d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=day_offset)
            rows.append({
                "date": d,
                "channel": channel,
                "campaign_id": cid,
                "campaign_name": f"{channel}_{ctype}_Campaign",
                "campaign_type": ctype,
                "spend": 50.0 + (day_offset * 0.5),
                "revenue": 150.0 + (day_offset * 1.5),
                "clicks": 20 + day_offset,
                "impressions": 200 + day_offset * 2,
                "conversions": 2.0 + (day_offset * 0.1),
                "daily_budget": 100.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def dirty_df():
    """DataFrame with various validation issues."""
    df = pd.DataFrame({
        "date": [
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-01-02"),
            pd.Timestamp("2026-12-01"),
        ],
        "channel": ["google", "google", "meta", "google"],
        "campaign_id": ["g_1", "g_1", "m_1", "g_1"],
        "campaign_name": ["Campaign_A", "Campaign_A", "Campaign_B", "Campaign_A"],
        "campaign_type": ["SEARCH", "SEARCH", "Generic", "SEARCH"],
        "spend": [100.0, 100.0, -50.0, 200.0],
        "revenue": [300.0, 300.0, 150.0, 600.0],
        "clicks": [30, 30, 10, 60],
        "impressions": [300, 300, 0, 600],
        "conversions": [3.0, 3.0, 0.0, 6.0],
        "daily_budget": [100.0, 100.0, 100.0, 100.0],
    })
    return df


class TestValidationEngine:
    def test_init_with_default_config(self):
        engine = ValidationEngine()
        assert engine._config is not None

    def test_init_with_custom_config(self):
        config = Config()
        engine = ValidationEngine(config=config)
        assert engine._config is config

    def test_validate_clean_dataframe(self, clean_unified_df):
        engine = ValidationEngine()
        report = engine.validate(clean_unified_df)
        assert report.passed
        assert len(report.errors) == 0
        assert report.original_shape == clean_unified_df.shape
        assert report.final_shape == clean_unified_df.shape

    def test_validate_dirty_dataframe(self, dirty_df):
        engine = ValidationEngine()
        report = engine.validate(dirty_df)
        # Should have duplicate row error, negative spend, future date
        assert len(report.errors) > 0
        assert len(report.warnings) > 0
        assert not report.passed
        assert report.original_shape[0] == 4
        assert report.final_shape[0] < 4

    def test_report_stats_structure(self, clean_unified_df):
        engine = ValidationEngine()
        report = engine.validate(clean_unified_df)
        assert "original_rows" in report.stats
        assert "final_rows" in report.stats
        assert "rows_removed" in report.stats
        assert "total_errors" in report.stats
        assert "total_warnings" in report.stats
        assert "channels_present" in report.stats
        assert "date_range" in report.stats

    def test_report_has_cleaned_df(self, clean_unified_df):
        engine = ValidationEngine()
        report = engine.validate(clean_unified_df)
        assert isinstance(report.cleaned_df, pd.DataFrame)
        assert len(report.cleaned_df) > 0

    def test_empty_dataframe(self):
        engine = ValidationEngine()
        df = pd.DataFrame(columns=["date", "channel", "campaign_id", "campaign_name",
                                    "campaign_type", "spend", "revenue", "clicks",
                                    "impressions", "conversions", "daily_budget"])
        report = engine.validate(df)
        assert not report.passed

    def test_missing_channel_column(self):
        engine = ValidationEngine()
        df = pd.DataFrame({"a": [1], "b": [2]})
        report = engine.validate(df)
        assert not report.passed

    def test_all_checks_run_independently(self):
        """Each check should not crash if a column is missing."""
        engine = ValidationEngine()
        df = pd.DataFrame({"date": [pd.Timestamp("2024-01-01")]})
        report = engine.validate(df)
        # Should complete without crashing; no pass guarantee
        assert isinstance(report.cleaned_df, pd.DataFrame)
