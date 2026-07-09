import pandas as pd
import pytest

from src.ingestion.google_ads import GoogleAdsIngestor


class TestGoogleAdsIngestor:
    def setup_method(self):
        self.ingestor = GoogleAdsIngestor()

    def test_can_handle_with_valid_columns(self, sample_google_raw):
        assert self.ingestor.can_handle(sample_google_raw) is True

    def test_can_handle_with_missing_columns(self):
        df = pd.DataFrame({"segments_date": [], "campaign_id": []})
        assert self.ingestor.can_handle(df) is False

    def test_can_handle_rejects_meta_columns(self, sample_meta_raw):
        assert self.ingestor.can_handle(sample_meta_raw) is False

    def test_can_handle_rejects_bing_columns(self, sample_bing_raw):
        assert self.ingestor.can_handle(sample_bing_raw) is False

    def test_normalize_output_schema(self, sample_google_raw):
        result = self.ingestor.normalize(sample_google_raw)

        expected_columns = {
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        }
        assert set(result.columns) == expected_columns
        assert len(result) == 2

    def test_normalize_cost_micros_conversion(self, sample_google_raw):
        result = self.ingestor.normalize(sample_google_raw)
        # 46,980,000 microdollars = $46.98
        assert result["spend"].iloc[0] == pytest.approx(46.98, rel=1e-2)
        # 53,558,564 microdollars = $53.56
        assert result["spend"].iloc[1] == pytest.approx(53.56, rel=1e-2)

    def test_normalize_whitespace_stripped(self, sample_google_raw):
        result = self.ingestor.normalize(sample_google_raw)
        assert result["campaign_name"].iloc[0] == "Search_TM_Campaign_01"
        assert result["campaign_type"].iloc[0] == "SEARCH"

    def test_normalize_channel_is_google(self, sample_google_raw):
        result = self.ingestor.normalize(sample_google_raw)
        assert (result["channel"] == "google").all()

    def test_normalize_campaign_id_namespaced(self, sample_google_raw):
        result = self.ingestor.normalize(sample_google_raw)
        assert result["campaign_id"].iloc[0] == "google_9988712287"

    def test_normalize_revenue_mapped(self, sample_google_raw):
        result = self.ingestor.normalize(sample_google_raw)
        assert result["revenue"].iloc[0] == 549.99

    def test_normalize_budget_parsed(self, sample_google_raw):
        result = self.ingestor.normalize(sample_google_raw)
        assert result["daily_budget"].iloc[0] == 90.0

    def test_normalize_missing_columns_raises(self):
        df = pd.DataFrame({"date": [], "spend": []})
        with pytest.raises(ValueError, match="missing required columns"):
            self.ingestor.normalize(df)

    def test_normalize_empty_dataframe(self):
        df = pd.DataFrame(
            {
                "campaign_id": pd.Series(dtype="int64"),
                "segments_date": pd.Series(dtype="object"),
                "metrics_clicks": pd.Series(dtype="int64"),
                "metrics_conversions": pd.Series(dtype="float64"),
                "metrics_cost_micros": pd.Series(dtype="int64"),
                "metrics_impressions": pd.Series(dtype="int64"),
                "metrics_video_views": pd.Series(dtype="int64"),
                "metrics_conversions_value": pd.Series(dtype="float64"),
                "campaign_advertising_channel_type": pd.Series(dtype="object"),
                "campaign_budget_amount": pd.Series(dtype="object"),
                "campaign_name": pd.Series(dtype="object"),
            }
        )
        result = self.ingestor.normalize(df)
        assert len(result) == 0
        assert set(result.columns) == {
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        }
