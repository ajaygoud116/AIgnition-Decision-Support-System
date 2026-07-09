import pandas as pd

from src.models.ingestion import Channel
from src.ingestion.base import AbstractIngestor


class GoogleAdsIngestor(AbstractIngestor):
    """Ingests Google Ads campaign statistics.

    Detects by: campaign_id, segments_date, metrics_cost_micros columns.
    Converts metrics_cost_micros (int) to spend (float) by dividing by 1,000,000.
    Budget column is parsed from whitespace-padded string format.
    """

    REQUIRED_COLUMNS = {
        "campaign_id",
        "segments_date",
        "metrics_cost_micros",
        "metrics_conversions_value",
        "campaign_name",
        "campaign_advertising_channel_type",
    }

    COLUMN_MAP = {
        "segments_date": "date",
        "campaign_id": "campaign_id_raw",
        "metrics_clicks": "clicks",
        "metrics_impressions": "impressions",
        "metrics_cost_micros": "spend_raw",
        "metrics_conversions_value": "revenue",
        "metrics_conversions": "conversions",
        "campaign_advertising_channel_type": "campaign_type",
        "campaign_budget_amount": "budget_raw",
        "campaign_name": "campaign_name",
    }

    def can_handle(self, raw: pd.DataFrame) -> bool:
        present = set(raw.columns.str.strip())
        return self.REQUIRED_COLUMNS.issubset(present)

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df.columns = df.columns.str.strip()

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Google Ads CSV is missing required columns: {missing}"
            )

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(df["segments_date"].astype(str).str.strip())
        result["channel"] = Channel.GOOGLE.value
        result["campaign_id_raw"] = df["campaign_id"].astype(str).str.strip()
        result["campaign_name"] = df["campaign_name"].astype(str).str.strip()
        result["campaign_type"] = (
            df["campaign_advertising_channel_type"].astype(str).str.strip()
        )
        result["spend"] = pd.to_numeric(df["metrics_cost_micros"], errors="coerce") / 1_000_000.0
        result["revenue"] = pd.to_numeric(df["metrics_conversions_value"], errors="coerce").fillna(0.0)
        result["clicks"] = pd.to_numeric(df["metrics_clicks"], errors="coerce").fillna(0).astype(int)
        result["impressions"] = pd.to_numeric(df["metrics_impressions"], errors="coerce").fillna(0).astype(int)
        result["conversions"] = pd.to_numeric(df["metrics_conversions"], errors="coerce").fillna(0.0)

        budget_clean = df["campaign_budget_amount"].astype(str).str.strip()
        result["daily_budget"] = pd.to_numeric(budget_clean, errors="coerce").fillna(0.0)

        result["campaign_id"] = result.apply(
            lambda r: f"{Channel.GOOGLE.value}_{r['campaign_id_raw']}", axis=1
        )

        return result[
            [
                "date",
                "channel",
                "campaign_id",
                "campaign_name",
                "campaign_type",
                "spend",
                "revenue",
                "clicks",
                "impressions",
                "conversions",
                "daily_budget",
            ]
        ]
