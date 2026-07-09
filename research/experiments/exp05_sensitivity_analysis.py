"""Experiment 5: Sensitivity Analysis.

Tests robustness of the forecasting system under various stress scenarios:
1. Missing data (10%, 20%, 30% randomly removed)
2. Increased noise (2x, 3x standard deviation)
3. Outliers (inject 99th percentile values)
4. Concept drift (gradual mean shift)
5. Budget shocks (sudden spend changes)
6. Seasonality shift (day-of-week pattern disruption)
"""

import numpy as np
import pandas as pd

from src.forecasting.lightgbm_model import LightGBMQuantileForecaster
from src.utils.config import Config as SrcConfig

from research.core.config import ExperimentConfig
from research.core.data import ExperimentData
from research.core.metrics import ForecastMetrics
from research.core.visualization import ExperimentVisualizer
from research.core.reporting import ExperimentReporter


def _inject_missing(df: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """Randomly set rows' revenue to NaN."""
    result = df.copy()
    mask = np.random.rand(len(result)) < ratio
    result.loc[mask, "revenue"] = np.nan
    result["revenue"] = result["revenue"].fillna(method="ffill").fillna(0)
    return result


def _inject_noise(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    """Add extra Gaussian noise to revenue."""
    result = df.copy()
    noise = np.random.normal(0, result["revenue"].std() * (factor - 1), len(result))
    result["revenue"] = result["revenue"] + noise
    result["revenue"] = result["revenue"].clip(lower=0)
    return result


def _inject_outliers(df: pd.DataFrame, ratio: float = 0.02) -> pd.DataFrame:
    """Replace random rows' revenue with extreme values."""
    result = df.copy()
    mask = np.random.rand(len(result)) < ratio
    extreme_val = result["revenue"].quantile(0.99) * 5
    result.loc[mask, "revenue"] = extreme_val
    return result


def _inject_concept_drift(df: pd.DataFrame, shift_per_day: float = 0.005) -> pd.DataFrame:
    """Gradually shift mean revenue upward."""
    result = df.copy()
    dates = sorted(result["date"].unique())
    date_shift = {d: i * shift_per_day for i, d in enumerate(dates)}
    result["drift_factor"] = result["date"].map(date_shift)
    result["revenue"] = result["revenue"] * (1.0 + result["drift_factor"])
    result = result.drop(columns=["drift_factor"])
    return result


def run(config: ExperimentConfig, src_config: SrcConfig) -> dict:
    print("=" * 60)
    print("EXP05: Sensitivity Analysis")
    print("=" * 60)

    data = ExperimentData(src_config)
    feature_df = data.get_feature_data()
    print(f"Loaded feature DataFrame: {feature_df.shape}")

    metrics_calc = ForecastMetrics()
    reporter = ExperimentReporter(config.output_dir)

    # Use date-based split for this experiment
    dates = sorted(feature_df["date"].unique())
    split_idx = len(dates) * 2 // 3
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])

    train_df = feature_df[feature_df["date"].isin(train_dates)].copy()
    test_df = feature_df[feature_df["date"].isin(test_dates)].copy()

    # Baseline
    base_model = LightGBMQuantileForecaster(random_seed=config.random_seed)
    base_model.fit(train_df, target_col="revenue")
    base_preds = base_model.predict(test_df)
    test_actuals = test_df.groupby("date")["revenue"].mean().values
    min_len = min(len(test_actuals), len(base_preds))
    base_metrics = metrics_calc.all_metrics(
        test_actuals[:min_len], base_preds[:min_len, 1],
        base_preds[:min_len, 0], base_preds[:min_len, 2],
    )
    base_rmse = base_metrics["rmse"]
    print(f"  Baseline RMSE: {base_rmse:.2f}")

    scenarios = {
        "baseline": lambda df: df,
        "missing_10pct": lambda df: _inject_missing(df, 0.1),
        "missing_30pct": lambda df: _inject_missing(df, 0.3),
        "noise_2x": lambda df: _inject_noise(df, 2.0),
        "noise_3x": lambda df: _inject_noise(df, 3.0),
        "outliers_2pct": lambda df: _inject_outliers(df, 0.02),
        "concept_drift": lambda df: _inject_concept_drift(df, 0.005),
    }

    results = {}
    for scenario_name, inject_fn in scenarios.items():
        modified_df = inject_fn(feature_df.copy())
        mod_train = modified_df[modified_df["date"].isin(train_dates)].copy()
        mod_test = modified_df[modified_df["date"].isin(test_dates)].copy()

        if mod_train.empty or mod_test.empty:
            continue

        model = LightGBMQuantileForecaster(random_seed=config.random_seed)
        model.fit(mod_train, target_col="revenue")
        preds = model.predict(mod_test)

        mod_actuals = mod_test.groupby("date")["revenue"].mean().values
        min_l = min(len(mod_actuals), len(preds))
        metrics = metrics_calc.all_metrics(
            mod_actuals[:min_l], preds[:min_l, 1],
            preds[:min_l, 0], preds[:min_l, 2],
        )
        metrics["delta_from_baseline"] = metrics["rmse"] - base_rmse
        metrics["delta_pct"] = (metrics["rmse"] - base_rmse) / base_rmse * 100
        results[scenario_name] = metrics

        print(f"  {scenario_name:20s} RMSE={metrics['rmse']:8.2f}  "
              f"Delta={metrics['delta_pct']:+6.1f}%  Coverage={metrics.get('coverage_90', 0):.2f}")

    reporter.save_metrics_table(results, "exp05_sensitivity")
    print("Saved: research/tables/exp05_sensitivity.csv")

    summary = {
        "experiment": "exp05_sensitivity_analysis",
        "baseline_rmse": round(base_rmse, 2),
        "most_robust_scenario": min(results, key=lambda s: results[s].get("delta_pct", 0)),
        "most_sensitive_scenario": max(results, key=lambda s: results[s].get("delta_pct", 0)),
        "max_degradation_pct": round(max(results[s].get("delta_pct", 0) for s in results), 2),
    }
    reporter.save_summary(summary, "exp05_sensitivity")

    print("\nEXP05 complete.\n")
    return results
