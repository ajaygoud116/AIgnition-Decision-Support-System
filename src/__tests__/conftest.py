from datetime import date
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_google_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "campaign_id": [9988712287, 9988712287],
            "segments_date": [" 2024-01-01   ", " 2024-01-02   "],
            "metrics_clicks": [158, 244],
            "metrics_conversions": [4.19, 4.08],
            "metrics_cost_micros": [46980000, 53558564],
            "metrics_impressions": [481, 818],
            "metrics_video_views": [0, 0],
            "metrics_conversions_value": [549.99, 670.82],
            "campaign_advertising_channel_type": [" SEARCH   ", " SEARCH   "],
            "campaign_budget_amount": [" 90.0 ", " 90.0 "],
            "campaign_name": [" Search_TM_Campaign_01", " Search_TM_Campaign_01"],
        }
    )


@pytest.fixture
def sample_meta_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "campaign_id": [120210921616440533, 120210921616440533],
            "date_start": [" 2024-05-23", " 2024-05-24"],
            "cpc": [12.10, 17.18],
            "cpm": [55.68, 63.25],
            "ctr": [1.62, 2.02],
            "reach": [0.0, 0.0],
            "spend": [85.0, 85.0],
            "clicks": [37.0, 38.0],
            "impressions": [5188.0, 5080.0],
            "conversion": [0.0, 183.0],
            "daily_budget": ["", ""],
            "campaign_name": [" Generic_Campaign_02", " Generic_Campaign_02"],
        }
    )


@pytest.fixture
def sample_bing_raw() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CampaignId": [566560838, 566560838],
            "TimePeriod": [" 2024-05-25", " 2024-05-26"],
            "Revenue": [0.0, 0.0],
            "Spend": [4.70, 4.30],
            "Clicks": [22, 14],
            "Impressions": [140, 120],
            "Conversions": [0.0, 0.0],
            "CampaignType": [" Search   ", " Search   "],
            "DailyBudget": [10.0, 10.0],
            "CampaignName": [" Search_TM_Campaign_02", " Search_TM_Campaign_02"],
        }
    )


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory with sample CSV files for pipeline testing."""
    google_df = pd.DataFrame(
        {
            "campaign_id": [9988712287],
            "segments_date": [" 2024-01-01   "],
            "metrics_clicks": [158],
            "metrics_conversions": [4.19],
            "metrics_cost_micros": [46980000],
            "metrics_impressions": [481],
            "metrics_video_views": [0],
            "metrics_conversions_value": [549.99],
            "campaign_advertising_channel_type": [" SEARCH   "],
            "campaign_budget_amount": [" 90.0 "],
            "campaign_name": [" Search_TM_Campaign_01"],
        }
    )
    google_df.to_csv(tmp_path / "google_ads_campaign_stats.csv", index=False)

    meta_df = pd.DataFrame(
        {
            "campaign_id": [120210921616440533],
            "date_start": [" 2024-05-23"],
            "cpc": [12.10],
            "cpm": [55.68],
            "ctr": [1.62],
            "reach": [0.0],
            "spend": [85.0],
            "clicks": [37.0],
            "impressions": [5188.0],
            "conversion": [0.0],
            "daily_budget": [""],
            "campaign_name": [" Generic_Campaign_02"],
        }
    )
    meta_df.to_csv(tmp_path / "meta_ads_campaign_stats.csv", index=False)

    bing_df = pd.DataFrame(
        {
            "CampaignId": [566560838],
            "TimePeriod": [" 2024-05-25"],
            "Revenue": [0.0],
            "Spend": [4.70],
            "Clicks": [22],
            "Impressions": [140],
            "Conversions": [0.0],
            "CampaignType": [" Search   "],
            "DailyBudget": [10.0],
            "CampaignName": [" Search_TM_Campaign_02"],
        }
    )
    bing_df.to_csv(tmp_path / "bing_campaign_stats.csv", index=False)

    return tmp_path
