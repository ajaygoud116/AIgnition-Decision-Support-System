"""Experiment 2: Feature Ablation.

Quantifies the contribution of each feature group by removing it
and measuring the degradation in forecast accuracy.
"""

from copy import deepcopy
from typing import Dict, List

import numpy as np
import pandas as pd

from src.features.builder import FeatureBuilder
from src.features.transforms import (
    add_lag_features, add_ratio_features, add_rolling_features,
    add_time_features, drop_high_na_rows,
)
from src.forecasting.lightgbm_model import LightGBMQuantileForecaster
from src.utils.config import Config as SrcConfig

from research.core.config import ExperimentConfig
from research.core.data import ExperimentData
from research.core.metrics import ForecastMetrics
from research.core.visualization import ExperimentVisualizer
from research.core.reporting import ExperimentReporter
from research.core.walkforward import ExpandingWindowCV


def run(config: ExperimentConfig, src_config: SrcConfig) -> Dict[str, Dict[str, float]]:
    print("=" * 60)
    print("EXP02: Feature Ablation")
    print("=" * 60)

    data = ExperimentData(src_config)
    feature_df = data.get_feature_data()
    print(f"Loaded feature DataFrame: {feature_df.shape}")

    metrics_calc = ForecastMetrics()
    viz = ExperimentVisualizer(config.output_dir)
    reporter = ExperimentReporter(config.output_dir)
    cv = ExpandingWindowCV(
        n_splits=config.cv_n_splits,
        initial_train_days=config.cv_initial_train_days,
        step_days=config.cv_step_days,
        forecast_horizon=config.cv_forecast_horizon,
    )

    base_columns = {
        "date", "channel", "campaign_id", "campaign_name",
        "campaign_type", "spend", "revenue", "clicks",
        "impressions", "conversions", "daily_budget",
    }

    def _build_without_group(df: pd.DataFrame, group: str) -> pd.DataFrame:
        """Build features excluding a specific group."""
        result = df.copy()
        result = add_time_features(result)
        result = add_ratio_features(result)

        if group != "rolling":
            result = _apply_per_campaign(result, add_rolling_features,
                                         windows=config.rolling_windows)
        if group != "lag":
            result = _apply_per_campaign(result, add_lag_features,
                                         lags=config.lag_windows)
        if group == "ratio":
            ratio_cols = [c for c in result.columns
                          if c.endswith("_per_") or c in ("roas", "ctr", "conv_rate",
                                                          "spend_per_click", "revenue_per_impression")]
            result = result.drop(columns=[c for c in ratio_cols if c in result.columns])
        if group == "time":
            time_cols = ["dow", "month", "quarter", "doy", "woy", "is_weekend"]
            result = result.drop(columns=[c for c in time_cols if c in result.columns])

        result = drop_high_na_rows(result, max_na_ratio=0.3)
        return result

    ablation_groups = ["all", "rolling", "lag", "ratio", "time"]
    all_results: Dict[str, Dict[str, float]] = {}

    for group in ablation_groups:
        print(f"  Ablation: exclude '{group}' features")
        fold_metrics: Dict[str, List[float]] = {}

        for fold_idx, (train_df, test_df) in enumerate(cv.split(feature_df)):
            train_abl = _build_without_group(train_df, group)
            test_abl = _build_without_group(test_df, group)

            if train_abl.empty or test_abl.empty:
                continue

            model = LightGBMQuantileForecaster(random_seed=config.random_seed)
            model.fit(train_abl, target_col="revenue")
            preds = model.predict(test_abl)

            test_actuals = test_abl.groupby("date")["revenue"].mean().values
            min_len = min(len(preds), len(test_actuals))

            metrics = metrics_calc.all_metrics(
                test_actuals[:min_len], preds[:min_len, 1],
                preds[:min_len, 0], preds[:min_len, 2],
            )

            for k, v in metrics.items():
                if k not in fold_metrics:
                    fold_metrics[k] = []
                fold_metrics[k].append(v)

            print(f"    Fold {fold_idx + 1}: RMSE={metrics['rmse']:.2f}")

        all_results[group] = {k: float(np.mean(v)) for k, v in fold_metrics.items()}

    # Compute delta vs "all" baseline
    baseline = all_results.get("all", {})
    for group in ablation_groups:
        if group == "all":
            continue
        if group in all_results:
            delta = all_results[group].get("rmse", 0) - baseline.get("rmse", 0)
            all_results[group]["delta_rmse_vs_all"] = delta
            all_results[group]["delta_pct"] = delta / baseline.get("rmse", 1) * 100

    reporter.save_metrics_table(all_results, "exp02_ablation")
    print("Saved: research/tables/exp02_ablation.csv")

    summary = {
        "experiment": "exp02_feature_ablation",
        "baseline_rmse": all_results.get("all", {}).get("rmse", 0),
        "worst_ablation": max(ablation_groups[1:],
                              key=lambda g: all_results.get(g, {}).get("delta_pct", 0)),
        "ablation_results": {g: {"rmse": all_results[g]["rmse"],
                                  "delta_pct": all_results[g].get("delta_pct", 0)}
                             for g in ablation_groups if g in all_results},
    }
    reporter.save_summary(summary, "exp02_ablation")

    print(f"\n  Baseline RMSE: {all_results.get('all', {}).get('rmse', 0):.2f}")
    for group in ablation_groups[1:]:
        if group in all_results:
            print(f"  Without {group}: RMSE={all_results[group]['rmse']:.2f} "
                  f"(delta={all_results[group].get('delta_pct', 0):.1f}%)")

    print("\nEXP02 complete.\n")
    return all_results


def _apply_per_campaign(df: pd.DataFrame, fn, **kwargs) -> pd.DataFrame:
    if "campaign_id" not in df.columns:
        return fn(df, **kwargs)
    groups = []
    for _, group in df.sort_values(["campaign_id", "date"]).groupby("campaign_id", sort=False):
        groups.append(fn(group, **kwargs))
    return pd.concat(groups, ignore_index=True)
