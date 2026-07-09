import pandas as pd
import pytest

from src.ingestion.meta_ads import MetaAdsIngestor


class TestMetaAdsIngestor:
    def setup_method(self):
        self.ingestor = MetaAdsIngestor()

    def test_can_handle_with_valid_columns(self, sample_meta_raw):
        assert self.ingestor.can_handle(sample_meta_raw) is True

    def test_can_handle_with_missing_columns(self):
        df = pd.DataFrame({"date_start": [], "spend": []})
        assert self.ingestor.can_handle(df) is False

    def test_can_handle_rejects_google_columns(self, sample_google_raw):
        assert self.ingestor.can_handle(sample_google_raw) is False

    def test_can_handle_rejects_bing_columns(self, sample_bing_raw):
        assert self.ingestor.can_handle(sample_bing_raw) is False

    def test_normalize_output_schema(self, sample_meta_raw):
        result = self.ingestor.normalize(sample_meta_raw)

        expected_columns = {
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        }
        assert set(result.columns) == expected_columns
        assert len(result) == 2

    def test_normalize_channel_is_meta(self, sample_meta_raw):
        result = self.ingestor.normalize(sample_meta_raw)
        assert (result["channel"] == "meta").all()

    def test_normalize_campaign_id_namespaced(self, sample_meta_raw):
        result = self.ingestor.normalize(sample_meta_raw)
        assert result["campaign_id"].iloc[0] == "meta_120210921616440533"

    def test_normalize_whitespace_stripped(self, sample_meta_raw):
        result = self.ingestor.normalize(sample_meta_raw)
        assert result["campaign_name"].iloc[0] == "Generic_Campaign_02"
        assert result["campaign_type"].iloc[0] == "Generic"

    def test_normalize_type_from_campaign_name(self, sample_meta_raw):
        result = self.ingestor.normalize(sample_meta_raw)
        assert result["campaign_type"].iloc[0] == "Generic"

    def test_normalize_empty_budget_becomes_zero(self, sample_meta_raw):
        result = self.ingestor.normalize(sample_meta_raw)
        assert result["daily_budget"].iloc[0] == 0.0

    def test_normalize_conversions_default(self, sample_meta_raw):
        result = self.ingestor.normalize(sample_meta_raw)
        assert (result["conversions"] == 0.0).all()

    def test_normalize_revenue_mapped(self, sample_meta_raw):
        result = self.ingestor.normalize(sample_meta_raw)
        assert result["revenue"].iloc[0] == 0.0
        assert result["revenue"].iloc[1] == 183.0

    def test_extract_campaign_type_prospecting(self):
        assert (
            MetaAdsIngestor._extract_campaign_type("Prospecting_DPA_Campaign_04")
            == "Prospecting"
        )

    def test_extract_campaign_type_remarketing(self):
        assert (
            MetaAdsIngestor._extract_campaign_type("Remarketing_Brand_Campaign_03")
            == "Remarketing"
        )

    def test_extract_campaign_type_generic_brand(self):
        assert (
            MetaAdsIngestor._extract_campaign_type("Generic_Brand_Campaign_01")
            == "Generic"
        )

    def test_extract_campaign_type_no_underscore(self):
        assert MetaAdsIngestor._extract_campaign_type("SimpleName") == "SimpleName"

    def test_extract_campaign_type_empty(self):
        assert MetaAdsIngestor._extract_campaign_type("") == "unknown"

    def test_normalize_missing_columns_raises(self):
        df = pd.DataFrame({"date": [], "spend": []})
        with pytest.raises(ValueError, match="missing required columns"):
            self.ingestor.normalize(df)
