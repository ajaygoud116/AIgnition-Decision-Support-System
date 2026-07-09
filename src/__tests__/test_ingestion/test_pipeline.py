from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.pipeline import IngestionPipeline
from src.utils.errors import IngestionError


class TestIngestionPipeline:
    def setup_method(self):
        self.pipeline = IngestionPipeline()

    def test_ingest_all_returns_unified_dataframe(self, sample_data_dir):
        result = self.pipeline.ingest_all(sample_data_dir)

        expected_columns = {
            "date", "channel", "campaign_id", "campaign_name",
            "campaign_type", "spend", "revenue", "clicks",
            "impressions", "conversions", "daily_budget",
        }
        assert set(result.columns) == expected_columns
        assert len(result) == 3  # 1 row per platform

    def test_ingest_all_includes_all_channels(self, sample_data_dir):
        result = self.pipeline.ingest_all(sample_data_dir)
        channels = result["channel"].unique()
        assert sorted(channels) == ["bing", "google", "meta"]

    def test_ingest_all_no_nan_in_channel(self, sample_data_dir):
        result = self.pipeline.ingest_all(sample_data_dir)
        assert result["channel"].isna().sum() == 0

    def test_ingest_all_dates_are_datetime(self, sample_data_dir):
        result = self.pipeline.ingest_all(sample_data_dir)
        assert result["date"].dtype == "datetime64[ns]"

    def test_ingest_all_no_duplicate_campaign_ids(self, sample_data_dir):
        """Campaign IDs are namespaced by channel, so no collisions."""
        result = self.pipeline.ingest_all(sample_data_dir)
        assert result["campaign_id"].is_unique

    def test_ingest_all_spend_is_positive(self, sample_data_dir):
        result = self.pipeline.ingest_all(sample_data_dir)
        assert (result["spend"] >= 0).all()

    def test_ingest_all_empty_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No CSV files"):
            self.pipeline.ingest_all(tmp_path)

    def test_ingest_all_nonexistent_directory_raises(self):
        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            self.pipeline.ingest_all(Path("/nonexistent/path"))

    def test_ingest_all_unrecognized_csv_raises(self, tmp_path):
        """A CSV with no matching ingestor should raise IngestionError."""
        df = pd.DataFrame({"unknown_col": [1, 2]})
        df.to_csv(tmp_path / "unknown_platform.csv", index=False)
        with pytest.raises(IngestionError, match="No ingestor found"):
            self.pipeline.ingest_all(tmp_path)

    def test_pipeline_idempotent(self, sample_data_dir):
        """Running ingest_all twice on the same data should return same shape."""
        result1 = self.pipeline.ingest_all(sample_data_dir)
        result2 = self.pipeline.ingest_all(sample_data_dir)
        assert len(result1) == len(result2)
        assert list(result1["channel"]) == list(result2["channel"])
