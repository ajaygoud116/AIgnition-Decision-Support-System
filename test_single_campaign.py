"""Test the model with a single custom campaign input."""
import pandas as pd
from pathlib import Path
from src.features.builder import FeatureBuilder
from src.forecasting.forecaster import Forecaster
from src.utils.config import Config

config = Config(Path("config.yaml"))
forecaster = Forecaster(config)

# Load the pre-trained model
import pickle
with open("pickle/model.pkl", "rb") as f:
    forecaster = pickle.load(f)

# --- CUSTOM INPUT: Change these values to test different scenarios ---
campaign_data = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=90, freq="D"),
    "campaign_id": "test_campaign_1",
    "channel": "google",
    "campaign_type": "SEARCH",
    "spend": [100 + i * 0.5 for i in range(90)],       # $100/day, slowly increasing
    "revenue": [150 + i * 0.8 for i in range(90)],      # $150/day, slowly increasing
    "clicks": [200 + i for i in range(90)],
    "impressions": [5000 + i * 10 for i in range(90)],
    "conversions": [5 + i * 0.05 for i in range(90)],
    "daily_budget": 200,
})

# Build features and predict
builder = FeatureBuilder(config)
feature_df = builder.build(campaign_data)
result = forecaster.predict(feature_df)

# Show results
print(f"\n{'='*60}")
print(f"Forecast for test_campaign_1")
print(f"{'='*60}")
for series in result.series:
    print(f"\nHorizon: {series.horizon.value} days")
    print(f"{'Date':15s} {'p10':>10s} {'p50':>10s} {'p90':>10s}")
    print("-" * 45)
    for point in series.points[:5]:  # Show first 5 days
        print(f"{str(point.date):15s} {point.values.p10:>10.2f} {point.values.p50:>10.2f} {point.values.p90:>10.2f}")
    print(f"... ({len(series.points)} total days)")
