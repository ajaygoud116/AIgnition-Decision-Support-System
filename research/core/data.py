from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.ingestion.pipeline import IngestionPipeline
from src.validation.validator import ValidationEngine
from src.features.builder import FeatureBuilder
from src.utils.config import Config


class ExperimentData:
    """Loads, validates, and featurizes data for experiments."""

    def __init__(self, config: Config):
        self._config = config
        self._data_dir = Path("data")

    def load_real_data(self) -> pd.DataFrame:
        """Load real CSV data through the full ingestion pipeline."""
        pipeline = IngestionPipeline()
        df = pipeline.ingest_all(self._data_dir)
        validator = ValidationEngine(self._config)
        report = validator.validate(df)
        return report.cleaned_df

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build features from a cleaned unified DataFrame."""
        builder = FeatureBuilder(self._config)
        return builder.build(df)

    def get_feature_data(self) -> pd.DataFrame:
        """One-shot: load real data, validate, build features, return."""
        df = self.load_real_data()
        return self.build_features(df)

    def synthetic_campaign_data(
        self,
        n_campaigns: int = 5,
        n_days: int = 200,
        seed: int = 42,
        trend: float = 0.5,
        noise_scale: float = 0.15,
        roas_base: float = 3.0,
    ) -> pd.DataFrame:
        """Generate synthetic multi-campaign data with known properties."""
        rng = np.random.RandomState(seed)
        rows = []
        channels = ["google", "meta", "bing"]
        types = {"google": "SEARCH", "meta": "Generic", "bing": "Search"}

        for c in range(n_campaigns):
            ch = channels[c % len(channels)]
            cid = f"syn_{ch}_{c}"
            base_spend = rng.uniform(50, 500)
            base_revenue = base_spend * roas_base * rng.uniform(0.7, 1.3)
            spend_noise = rng.uniform(0.05, 0.15)
            rev_noise = rng.uniform(0.1, 0.2)
            weekly_pattern = rng.uniform(0.7, 1.3, 7)

            for i in range(n_days):
                d = pd.Timestamp("2024-01-01") + pd.Timedelta(days=i)
                dow = d.dayofweek
                trend_factor = 1.0 + trend * i / n_days
                weekly = weekly_pattern[dow]
                seasonal = 1.0 + 0.2 * np.sin(2 * np.pi * i / 365)

                spend = base_spend * trend_factor * seasonal * weekly
                spend *= 1.0 + rng.normal(0, spend_noise)
                spend = max(spend, 1.0)

                actual_roas = roas_base * (1.0 + rng.normal(0, 0.1))
                revenue = spend * actual_roas * (1.0 + rng.normal(0, rev_noise))
                revenue = max(revenue, 0.0)

                clicks = int(spend / rng.uniform(0.5, 2.0) + rng.normal(0, 5))
                impressions = int(clicks / rng.uniform(0.01, 0.05) + rng.normal(0, 50))
                conversions = revenue / rng.uniform(50, 200)

                rows.append({
                    "date": d,
                    "channel": ch,
                    "campaign_id": cid,
                    "campaign_name": f"Syn_{ch}_{c}",
                    "campaign_type": types[ch],
                    "spend": round(spend, 2),
                    "revenue": round(revenue, 2),
                    "clicks": max(clicks, 0),
                    "impressions": max(impressions, 0),
                    "conversions": max(conversions, 0),
                    "daily_budget": round(base_spend * 1.2, 2),
                })

        df = pd.DataFrame(rows)
        return df

    def synthetic_quantile_data(
        self, n_samples: int = 1000, n_features: int = 10, seed: int = 42
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """Generate data with known quantile structure for calibration validation.

        Returns (feature_df, y_true, y_p10, y_p90) where the true quantiles
        are known analytically.
        """
        rng = np.random.RandomState(seed)
        n = n_samples
        p = n_features

        X = rng.randn(n, p)
        beta = rng.randn(p) * 0.5
        mu = X @ beta
        sigma = 0.5 + 0.5 * np.abs(X[:, 0])  # heteroskedastic
        y = mu + rng.normal(0, sigma)

        true_p10 = mu - 1.2816 * sigma
        true_p50 = mu
        true_p90 = mu + 1.2816 * sigma

        col_names = [f"feat_{i}" for i in range(p)]
        base_cols = {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "channel": ["google"] * n,
            "campaign_id": ["syn_calib"] * n,
            "campaign_name": ["Calib_Campaign"] * n,
            "campaign_type": ["SEARCH"] * n,
            "spend": np.abs(rng.randn(n) * 100 + 200),
            "revenue": y.copy(),
            "clicks": np.abs((rng.randn(n) * 20 + 50).astype(int)),
            "impressions": np.abs((rng.randn(n) * 200 + 500).astype(int)),
            "conversions": np.abs(y / 100),
            "daily_budget": np.full(n, 500.0),
        }

        feat_df = pd.DataFrame(base_cols)
        for i, name in enumerate(col_names):
            feat_df[name] = X[:, i]

        return feat_df, y, true_p10, true_p50, true_p90
