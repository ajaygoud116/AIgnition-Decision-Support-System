import pandas as pd
import pytest

from src.ingestion.bing_ads import BingAdsIngestor


class TestBingAdsIngestor:
    def setup_method(self):
        self.ingestor = BingAdsIngestor()

    def test_can_handle_with_valid_columns(self, sample_bing_raw):
        assert self.ingestor.can_handle(sample_bing_raw) is True

    def test_can_handle_with_missing_columns(self):
        df = pd.DataFrame({"TimePeriod": [], "Spend": []})
        assert self.ingestor.can_handle(df) is False

    def test_can_handle_rejects_google_columns(self, sample_google_raw):
        assert self.ingestor.can_handle(sample_google_raw) is False

    def test_can_handle_rejects_meta_columns(self, sample_meta_raw):
        assert self.ingestor.can_handle(sample_meta_raw) is False

    def test_normalize_output_schema(self, sample_bing_raw):
        result = self.ingestor.normalize(sample_bing_raw)

        expected_columns = {
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        }
        assert set(result.columns) == expected_columns
        assert len(result) == 2

    def test_normalize_channel_is_bing(self, sample_bing_raw):
        result = self.ingestor.normalize(sample_bing_raw)
        assert (result["channel"] == "bing").all()

    def test_normalize_campaign_id_namespaced(self, sample_bing_raw):
        result = self.ingestor.normalize(sample_bing_raw)
        assert result["campaign_id"].iloc[0] == "bing_566560838"

    def test_normalize_whitespace_stripped(self, sample_bing_raw):
        result = self.ingestor.normalize(sample_bing_raw)
        assert result["campaign_name"].iloc[0] == "Search_TM_Campaign_02"
        assert result["campaign_type"].iloc[0] == "Search"

    def test_normalize_direct_spend(self, sample_bing_raw):
        result = self.ingestor.normalize(sample_bing_raw)
        assert result["spend"].iloc[0] == 4.70
        assert result["spend"].iloc[1] == 4.30

    def test_normalize_revenue_mapped(self, sample_bing_raw):
        result = self.ingestor.normalize(sample_bing_raw)
        assert result["revenue"].iloc[0] == 0.0

    def test_normalize_conversions_float(self, sample_bing_raw):
        result = self.ingestor.normalize(sample_bing_raw)
        assert result["conversions"].iloc[0] == 0.0

    def test_normalize_budget_mapped(self, sample_bing_raw):
        result = self.ingestor.normalize(sample_bing_raw)
        assert result["daily_budget"].iloc[0] == 10.0

    def test_normalize_clicks_impressions_int(self, sample_bing_raw):
        result = self.ingestor.normalize(sample_bing_raw)
        assert result["clicks"].dtype == int
        assert result["impressions"].dtype == int
        assert result["clicks"].iloc[0] == 22
        assert result["impressions"].iloc[0] == 140

    def test_normalize_missing_columns_raises(self):
        df = pd.DataFrame({"date": [], "spend": []})
        with pytest.raises(ValueError, match="missing required columns"):
            self.ingestor.normalize(df)
