from dataclasses import dataclass
from datetime import date
from enum import Enum
from typing import Optional


class Channel(str, Enum):
    GOOGLE = "google"
    META = "meta"
    BING = "bing"


@dataclass(frozen=True)
class CampaignId:
    channel: Channel
    platform_id: str

    def __str__(self) -> str:
        return f"{self.channel.value}_{self.platform_id}"


@dataclass
class UnifiedRecord:
    date: date
    channel: Channel
    campaign_id: CampaignId
    campaign_name: str
    campaign_type: str
    spend: float
    revenue: float
    clicks: int
    impressions: int
    conversions: float
    daily_budget: float
