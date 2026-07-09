import numpy as np
import pandas as pd
import pytest

from src.simulation.baselines import CampaignBaseline, extract_baselines


@pytest.fixture
def sample_feature_df():
    np.random.seed(42)
    rows = []
    for cid, ch, ctype in [("camp_a", "google", "SEARCH"), ("camp_b", "meta", "Generic")]:
        for i in range(10):
            d = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
            rows.append({
                "date": d,
                "channel": ch,
                "campaign_id": cid,
                "campaign_type": ctype,
                "campaign_name": f"{ctype}_Test",
                "spend": 50.0 + i,
                "revenue": 150.0 + i * 3,
                "clicks": 20 + i,
                "impressions": 200 + i * 5,
                "conversions": 2.0 + i * 0.1,
                "daily_budget": 100.0,
            })
    return pd.DataFrame(rows)


class TestExtractBaselines:
    def test_returns_dict_keyed_by_campaign_id(self, sample_feature_df):
        baselines = extract_baselines(sample_feature_df)
        assert set(baselines.keys()) == {"camp_a", "camp_b"}

    def test_baseline_has_correct_entity_id(self, sample_feature_df):
        baselines = extract_baselines(sample_feature_df)
        assert baselines["camp_a"].entity_id == "camp_a"

    def test_baseline_has_correct_channel(self, sample_feature_df):
        baselines = extract_baselines(sample_feature_df)
        assert baselines["camp_a"].channel == "google"
        assert baselines["camp_b"].channel == "meta"

    def test_total_spend_is_summed(self, sample_feature_df):
        baselines = extract_baselines(sample_feature_df)
        total_spend = sample_feature_df[sample_feature_df["campaign_id"] == "camp_a"]["spend"].sum()
        assert baselines["camp_a"].total_spend == float(total_spend)

    def test_total_revenue_is_summed(self, sample_feature_df):
        baselines = extract_baselines(sample_feature_df)
        total_rev = sample_feature_df[sample_feature_df["campaign_id"] == "camp_a"]["revenue"].sum()
        assert baselines["camp_a"].total_revenue == float(total_rev)

    def test_historical_roas(self, sample_feature_df):
        baselines = extract_baselines(sample_feature_df)
        total_rev = sample_feature_df[sample_feature_df["campaign_id"] == "camp_a"]["revenue"].sum()
        total_spend = sample_feature_df[sample_feature_df["campaign_id"] == "camp_a"]["spend"].sum()
        expected_roas = total_rev / total_spend
        assert baselines["camp_a"].historical_roas == pytest.approx(expected_roas)

    def test_roas_zero_for_zero_spend(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2025-01-01")],
            "channel": ["google"],
            "campaign_id": ["zero_spend"],
            "campaign_type": ["SEARCH"],
            "campaign_name": ["Zero"],
            "spend": [0.0],
            "revenue": [100.0],
            "clicks": [0],
            "impressions": [0],
            "conversions": [0.0],
            "daily_budget": [0.0],
        })
        baselines = extract_baselines(df)
        assert baselines["zero_spend"].historical_roas == 0.0

    def test_last_daily_spend_from_last_row(self, sample_feature_df):
        baselines = extract_baselines(sample_feature_df)
        last_row = sample_feature_df[sample_feature_df["campaign_id"] == "camp_a"].sort_values("date").iloc[-1]
        assert baselines["camp_a"].last_daily_spend == float(last_row["spend"])

    def test_empty_dataframe_returns_empty_dict(self):
        df = pd.DataFrame(columns=["date", "campaign_id", "channel", "spend", "revenue", "campaign_type", "campaign_name", "clicks", "impressions", "conversions", "daily_budget"])
        baselines = extract_baselines(df)
        assert baselines == {}

    def test_campaign_type_is_set(self, sample_feature_df):
        baselines = extract_baselines(sample_feature_df)
        assert baselines["camp_a"].campaign_type == "SEARCH"
        assert baselines["camp_b"].campaign_type == "Generic"

    def test_last_daily_budget(self, sample_feature_df):
        baselines = extract_baselines(sample_feature_df)
        assert baselines["camp_a"].last_daily_budget == 100.0

    def test_missing_daily_budget_defaults_zero(self):
        df = pd.DataFrame({
            "date": [pd.Timestamp("2025-01-01")],
            "channel": ["google"],
            "campaign_id": ["no_budget"],
            "campaign_type": ["SEARCH"],
            "campaign_name": ["Test"],
            "spend": [50.0],
            "revenue": [150.0],
            "clicks": [20],
            "impressions": [200],
            "conversions": [1.0],
        })
        baselines = extract_baselines(df)
        assert baselines["no_budget"].last_daily_budget == 0.0
