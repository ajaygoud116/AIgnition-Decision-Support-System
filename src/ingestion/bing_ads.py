import pandas as pd

from src.models.ingestion import Channel
from src.ingestion.base import AbstractIngestor


class BingAdsIngestor(AbstractIngestor):
    """Ingests Microsoft Ads (Bing) campaign statistics.

    Detects by: CampaignId, TimePeriod, Revenue, Spend columns.
    Column names in the raw CSV have leading/trailing whitespace which is stripped.
    CampaignType values also contain padding whitespace which is stripped.
    """

    REQUIRED_COLUMNS = {
        "CampaignId",
        "TimePeriod",
        "Revenue",
        "Spend",
        "CampaignName",
        "CampaignType",
    }

    COLUMN_MAP = {
        "TimePeriod": "date",
        "CampaignId": "campaign_id_raw",
        "Clicks": "clicks",
        "Impressions": "impressions",
        "Spend": "spend",
        "Revenue": "revenue",
        "Conversions": "conversions",
        "CampaignType": "campaign_type",
        "DailyBudget": "budget_raw",
        "CampaignName": "campaign_name",
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
                f"Bing Ads CSV is missing required columns: {missing}"
            )

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(df["TimePeriod"].astype(str).str.strip())
        result["channel"] = Channel.BING.value
        result["campaign_id_raw"] = df["CampaignId"].astype(str).str.strip()
        result["campaign_name"] = df["CampaignName"].astype(str).str.strip()
        result["campaign_type"] = df["CampaignType"].astype(str).str.strip()
        result["spend"] = pd.to_numeric(df["Spend"], errors="coerce").fillna(0.0)
        result["revenue"] = pd.to_numeric(df["Revenue"], errors="coerce").fillna(0.0)
        result["clicks"] = pd.to_numeric(df["Clicks"], errors="coerce").fillna(0).astype(int)
        result["impressions"] = pd.to_numeric(df["Impressions"], errors="coerce").fillna(0).astype(int)
        result["conversions"] = pd.to_numeric(df["Conversions"], errors="coerce").fillna(0.0)

        budget_clean = df["DailyBudget"].astype(str).str.strip()
        result["daily_budget"] = pd.to_numeric(budget_clean, errors="coerce").fillna(0.0)

        result["campaign_id"] = result.apply(
            lambda r: f"{Channel.BING.value}_{r['campaign_id_raw']}", axis=1
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
