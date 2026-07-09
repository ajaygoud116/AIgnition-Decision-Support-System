import pandas as pd

from src.models.ingestion import Channel
from src.ingestion.base import AbstractIngestor


class MetaAdsIngestor(AbstractIngestor):
    """Ingests Meta Ads campaign statistics.

    Detects by: campaign_id, date_start, cpc, cpm columns.
    Meta has no explicit campaign_type column — it is derived from the campaign name
    by extracting the first underscore-delimited segment (e.g. 'Prospecting_DPA_...' → 'Prospecting').
    Daily budget is often blank; missing values are filled with 0.0.
    """

    REQUIRED_COLUMNS = {
        "campaign_id",
        "date_start",
        "spend",
        "conversion",
        "campaign_name",
    }

    COLUMN_MAP = {
        "date_start": "date",
        "campaign_id": "campaign_id_raw",
        "spend": "spend",
        "conversion": "revenue",
        "clicks": "clicks",
        "impressions": "impressions",
        "campaign_name": "campaign_name",
        "daily_budget": "budget_raw",
    }

    def can_handle(self, raw: pd.DataFrame) -> bool:
        present = set(raw.columns.str.strip())
        return self.REQUIRED_COLUMNS.issubset(present)

    @staticmethod
    def _extract_campaign_type(name: str) -> str:
        """Derive campaign type from the first segment of the campaign name.

        Examples:
          'Generic_Campaign_02'       → 'Generic'
          'Prospecting_DPA_Campaign_04' → 'Prospecting'
          'Remarketing_Brand_Campaign_03' → 'Remarketing'
        """
        if not name or "_" not in name:
            return name or "unknown"
        return name.split("_")[0]

    def normalize(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        df.columns = df.columns.str.strip()

        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Meta Ads CSV is missing required columns: {missing}"
            )

        result = pd.DataFrame()
        result["date"] = pd.to_datetime(df["date_start"].astype(str).str.strip())
        result["channel"] = Channel.META.value
        result["campaign_id_raw"] = df["campaign_id"].astype(str).str.strip()
        result["campaign_name"] = df["campaign_name"].astype(str).str.strip()
        result["campaign_type"] = result["campaign_name"].apply(self._extract_campaign_type)
        result["spend"] = pd.to_numeric(df["spend"], errors="coerce").fillna(0.0)
        result["revenue"] = pd.to_numeric(df["conversion"], errors="coerce").fillna(0.0)
        result["clicks"] = pd.to_numeric(df["clicks"], errors="coerce").fillna(0).astype(int)
        result["impressions"] = pd.to_numeric(df["impressions"], errors="coerce").fillna(0).astype(int)
        result["conversions"] = 0.0  # Meta data does not provide conversions count

        budget_raw = df["daily_budget"].astype(str).str.strip()
        result["daily_budget"] = pd.to_numeric(budget_raw, errors="coerce").fillna(0.0)

        result["campaign_id"] = result.apply(
            lambda r: f"{Channel.META.value}_{r['campaign_id_raw']}", axis=1
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
