from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class CampaignBaseline:
    entity_id: str
    channel: str
    campaign_type: str
    total_spend: float
    total_revenue: float
    last_daily_spend: float
    last_daily_budget: float
    historical_roas: float


def extract_baselines(feature_df: pd.DataFrame) -> Dict[str, CampaignBaseline]:
    baselines: Dict[str, CampaignBaseline] = {}
    for (cid, ch), group in feature_df.groupby(["campaign_id", "channel"], sort=False):
        group = group.sort_values("date")
        total_rev = float(group["revenue"].sum())
        total_spend = float(group["spend"].sum())
        last_row = group.iloc[-1]
        baselines[cid] = CampaignBaseline(
            entity_id=cid,
            channel=ch,
            campaign_type=str(last_row.get("campaign_type", "")),
            total_spend=total_spend,
            total_revenue=total_rev,
            last_daily_spend=float(last_row["spend"]),
            last_daily_budget=float(last_row.get("daily_budget", 0.0)),
            historical_roas=float(total_rev / total_spend) if total_spend > 0 else 0.0,
        )
    return baselines
