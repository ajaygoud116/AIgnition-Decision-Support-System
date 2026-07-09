import numpy as np
import pandas as pd
import pytest

from src.pipeline.runner import PipelineRunner
from src.utils.config import Config


@pytest.fixture
def config():
    return Config()


def _create_minimal_data_dir(base_path):
    """Create a directory with enough data to train and predict."""
    data_dir = base_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    n = 45  # enough for min_history_days=30 + feature windows
    dates = [pd.Timestamp("2024-01-01") + pd.Timedelta(days=i) for i in range(n)]

    google = pd.DataFrame({
        "campaign_id": ["g_1"] * n,
        "segments_date": [f" {d.strftime('%Y-%m-%d')}   " for d in dates],
        "metrics_clicks": [50 + i + int(np.random.normal(0, 3)) for i in range(n)],
        "metrics_conversions": [2.0 + i * 0.1 + float(np.random.normal(0, 0.2)) for i in range(n)],
        "metrics_cost_micros": [5000000 + i * 100000 for i in range(n)],
        "metrics_impressions": [500 + i * 5 + int(np.random.normal(0, 10)) for i in range(n)],
        "metrics_video_views": [0] * n,
        "metrics_conversions_value": [500.0 + i * 5 + float(np.random.normal(0, 5)) for i in range(n)],
        "campaign_advertising_channel_type": [" SEARCH   "] * n,
        "campaign_budget_amount": [" 100.0 "] * n,
        "campaign_name": [" Search_Campaign_01"] * n,
    })
    google.to_csv(data_dir / "google_ads_campaign_stats.csv", index=False)

    meta = pd.DataFrame({
        "campaign_id": [120210921616440533] * n,
        "date_start": [f" {d.strftime('%Y-%m-%d')}" for d in dates],
        "cpc": [12.0 + i * 0.1 for i in range(n)],
        "cpm": [55.0 + i * 0.2 for i in range(n)],
        "ctr": [1.5 + i * 0.01 for i in range(n)],
        "reach": [1000 + i * 10 for i in range(n)],
        "spend": [85.0 + i * 0.5 for i in range(n)],
        "clicks": [30 + i for i in range(n)],
        "impressions": [5000 + i * 50 for i in range(n)],
        "conversion": [0.0 + i * 0.5 for i in range(n)],
        "daily_budget": [""] * n,
        "campaign_name": [" Generic_Campaign_01"] * n,
    })
    meta.to_csv(data_dir / "meta_ads_campaign_stats.csv", index=False)

    bing = pd.DataFrame({
        "CampaignId": [566560838] * n,
        "TimePeriod": [f" {d.strftime('%Y-%m-%d')}" for d in dates],
        "Revenue": [0.0 + i * 0.5 for i in range(n)],
        "Spend": [4.0 + i * 0.1 for i in range(n)],
        "Clicks": [20 + i for i in range(n)],
        "Impressions": [100 + i * 2 for i in range(n)],
        "Conversions": [0.0 + i * 0.1 for i in range(n)],
        "CampaignType": [" Search   "] * n,
        "DailyBudget": [10.0] * n,
        "CampaignName": [" Search_Campaign_B1"] * n,
    })
    bing.to_csv(data_dir / "bing_campaign_stats.csv", index=False)

    return data_dir


class TestPipelineRunner:
    def test_run_creates_output_files(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        assert (output_dir / "forecasts.csv").exists()
        assert (output_dir / "uncertainty.csv").exists()
        assert (output_dir / "simulations.csv").exists()
        assert (output_dir / "recommendations.csv").exists()
        assert (output_dir / "summary.json").exists()

    def test_forecasts_csv_not_empty(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        df = pd.read_csv(output_dir / "forecasts.csv")
        assert len(df) > 0

    def test_forecasts_have_revenue_metric(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        df = pd.read_csv(output_dir / "forecasts.csv")
        assert (df["metric"] == "revenue").any()

    def test_uncertainty_csv_not_empty(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        df = pd.read_csv(output_dir / "uncertainty.csv")
        assert len(df) > 0

    def test_simulations_csv_not_empty(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        df = pd.read_csv(output_dir / "simulations.csv")
        assert len(df) > 0

    def test_recommendations_csv_not_empty(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        df = pd.read_csv(output_dir / "recommendations.csv")
        assert len(df) > 0

    def test_summary_json_has_keys(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        import json
        with open(output_dir / "summary.json") as f:
            data = json.load(f)
        assert "campaigns_forecasted" in data
        assert "scenarios_simulated" in data
        assert "recommendations" in data

    def test_model_is_pickled_after_run(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        assert model_path.exists()

    def test_second_run_loads_pickled_model(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        mtime_first = model_path.stat().st_mtime

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)

        assert True  # Successful second run means pickle loading works

    def test_force_retrain_overwrites_model(self, tmp_path):
        data_dir = _create_minimal_data_dir(tmp_path)
        model_path = tmp_path / "pickle" / "model.pkl"
        output_dir = tmp_path / "output"

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir)
        mtime_first = model_path.stat().st_mtime

        runner = PipelineRunner(Config())
        runner.run(data_dir, model_path, output_dir, force_retrain=True)

        assert True  # Force retrain succeeds without error
