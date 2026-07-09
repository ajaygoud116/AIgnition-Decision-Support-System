"""Experiment 1: Forecast Benchmark.

Compares 8 models (Naive, Historical Mean, Seasonal Naive, Linear Regression,
Random Forest, XGBoost, LightGBM, Ensemble) using walk-forward validation.
"""

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.utils.config import Config as SrcConfig

from research.core.config import ExperimentConfig
from research.core.data import ExperimentData
from research.core.metrics import ForecastMetrics
from research.core.significance import compute_ranking, diebold_mariano
from research.core.visualization import ExperimentVisualizer
from research.core.reporting import ExperimentReporter
from research.core.walkforward import ExpandingWindowCV


def _naive_forecast(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Naive: last observed value repeated."""
    last_val = train["revenue"].iloc[-1]
    return np.full(horizon, last_val)


def _historical_mean(train: pd.DataFrame, horizon: int) -> np.ndarray:
    """Historical mean of all training revenue."""
    mean_val = train["revenue"].mean()
    return np.full(horizon, mean_val)


def _seasonal_naive(train: pd.DataFrame, horizon: int, window: int = 7) -> np.ndarray:
    """Seasonal naive: average of last {window} days."""
    last_n = train["revenue"].tail(window).mean()
    return np.full(horizon, last_n)


def _linear_regression(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       target_col: str = "revenue") -> np.ndarray:
    """Linear regression on available features."""
    from sklearn.linear_model import LinearRegression

    feature_cols = [c for c in train_df.columns
                    if c not in ["date", "campaign_id", "campaign_name", "channel",
                                 "campaign_type", target_col]]
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values

    # Handle NaN
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _random_forest(train_df: pd.DataFrame, test_df: pd.DataFrame,
                   target_col: str = "revenue", seed: int = 42) -> np.ndarray:
    from sklearn.ensemble import RandomForestRegressor

    feature_cols = [c for c in train_df.columns
                    if c not in ["date", "campaign_id", "campaign_name", "channel",
                                 "campaign_type", target_col]]
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
    y_train = np.nan_to_num(train_df[target_col].values, nan=0.0)
    X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0)

    model = RandomForestRegressor(n_estimators=300, max_depth=15,
                                  random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _xgboost(train_df: pd.DataFrame, test_df: pd.DataFrame,
             target_col: str = "revenue", seed: int = 42) -> np.ndarray:
    import xgboost as xgb

    feature_cols = [c for c in train_df.columns
                    if c not in ["date", "campaign_id", "campaign_name", "channel",
                                 "campaign_type", target_col]]
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
    y_train = np.nan_to_num(train_df[target_col].values, nan=0.0)
    X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0)

    model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                             max_depth=6, random_state=seed, verbosity=0)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame,
              target_col: str = "revenue", seed: int = 42) -> np.ndarray:
    import lightgbm as lgb

    feature_cols = [c for c in train_df.columns
                    if c not in ["date", "campaign_id", "campaign_name", "channel",
                                 "campaign_type", target_col]]
    feature_cols = [c for c in feature_cols if c in train_df.columns]

    X_train = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
    y_train = np.nan_to_num(train_df[target_col].values, nan=0.0)
    X_test = np.nan_to_num(test_df[feature_cols].values, nan=0.0)

    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                              num_leaves=31, min_child_samples=20,
                              subsample=0.8, colsample_bytree=0.8,
                              verbosity=-1, random_state=seed)
    model.fit(X_train, y_train)
    return model.predict(X_test)


def _ensemble_forecast(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       target_col: str = "revenue", seed: int = 42) -> np.ndarray:
    """50/50 average of LightGBM and Seasonal Naive."""
    lgb_preds = _lightgbm(train_df, test_df, target_col, seed)
    sn_val = train_df[target_col].tail(7).mean()
    sn_preds = np.full(len(lgb_preds), sn_val)
    return 0.5 * lgb_preds + 0.5 * sn_preds


def run(config: ExperimentConfig, src_config: SrcConfig) -> Dict[str, Dict[str, float]]:
    print("=" * 60)
    print("EXP01: Forecast Benchmark")
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

    all_results: Dict[str, Dict[str, float]] = {}
    all_raw_p50 = {}
    all_raw_true = {}
    training_times: Dict[str, List[float]] = {m: [] for m in config.benchmark_models}
    inference_times: Dict[str, List[float]] = {m: [] for m in config.benchmark_models}

    for fold_idx, (train_df, test_df) in enumerate(cv.split(feature_df)):
        horizon = len(test_df["date"].unique())
        print(f"  Fold {fold_idx + 1}: train={train_df['date'].min().date()}..{train_df['date'].max().date()}, "
              f"test={test_df['date'].min().date()}..{test_df['date'].max().date()}")

        # Pre-train ML models on ALL campaigns' data (once per fold for efficiency)
        ml_models = {}
        _needs_ml = any(m in config.benchmark_models
                        for m in ["linear_regression", "random_forest", "xgboost", "lightgbm", "ensemble"])
        if _needs_ml:
            feature_cols = [c for c in train_df.columns
                            if c not in ["date", "campaign_id", "campaign_name", "channel",
                                         "campaign_type", "revenue"]]
            feature_cols = [c for c in feature_cols if c in train_df.columns]
            X_train_all = np.nan_to_num(train_df[feature_cols].values, nan=0.0)
            y_train_all = np.nan_to_num(train_df["revenue"].values, nan=0.0)

            if "linear_regression" in config.benchmark_models:
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(X_train_all, y_train_all)
                ml_models["linear_regression"] = (lr, feature_cols)

            if "random_forest" in config.benchmark_models:
                from sklearn.ensemble import RandomForestRegressor
                rf = RandomForestRegressor(n_estimators=300, max_depth=15,
                                           random_state=config.random_seed, n_jobs=-1)
                rf.fit(X_train_all, y_train_all)
                ml_models["random_forest"] = (rf, feature_cols)

            if "xgboost" in config.benchmark_models:
                import xgboost as xgb
                xg = xgb.XGBRegressor(n_estimators=500, learning_rate=0.05,
                                      max_depth=6, random_state=config.random_seed, verbosity=0)
                xg.fit(X_train_all, y_train_all)
                ml_models["xgboost"] = (xg, feature_cols)

            if "lightgbm" in config.benchmark_models or "ensemble" in config.benchmark_models:
                import lightgbm as lgb
                lgb_m = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                          num_leaves=31, min_child_samples=20,
                                          subsample=0.8, colsample_bytree=0.8,
                                          verbosity=-1, random_state=config.random_seed)
                lgb_m.fit(X_train_all, y_train_all)
                ml_models["lightgbm"] = (lgb_m, feature_cols)

        for model_name in config.benchmark_models:
            t0 = time.time()
            camp_metrics = []
            camp_true_list = []
            camp_preds_list = []

            for cid in test_df["campaign_id"].unique():
                c_train = train_df[train_df["campaign_id"] == cid].sort_values("date")
                c_test = test_df[test_df["campaign_id"] == cid].sort_values("date")
                if len(c_train) < 7 or len(c_test) < 3:
                    continue

                if model_name == "naive":
                    preds = _naive_forecast(c_train, len(c_test))
                elif model_name == "historical_mean":
                    preds = _historical_mean(c_train, len(c_test))
                elif model_name == "seasonal_naive":
                    preds = _seasonal_naive(c_train, len(c_test))
                elif model_name in ml_models:
                    model, feat_cols = ml_models[model_name]
                    X_test = np.nan_to_num(c_test[feat_cols].values, nan=0.0)
                    preds = model.predict(X_test)
                elif model_name == "ensemble":
                    lgb_model, feat_cols = ml_models["lightgbm"]
                    X_test = np.nan_to_num(c_test[feat_cols].values, nan=0.0)
                    lgb_preds = lgb_model.predict(X_test)
                    sn_val = c_train["revenue"].tail(7).mean()
                    sn_preds = np.full(len(lgb_preds), sn_val)
                    preds = 0.5 * lgb_preds + 0.5 * sn_preds
                else:
                    raise ValueError(f"Unknown model: {model_name}")

                actuals = c_test["revenue"].values[:len(preds)]
                preds_aligned = preds[:len(actuals)]

                m = metrics_calc.all_metrics(
                    actuals, preds_aligned,
                    preds_aligned * 0.5,
                    preds_aligned * 1.5,
                )
                camp_metrics.append(m)
                camp_true_list.extend(actuals.tolist())
                camp_preds_list.extend(preds_aligned.tolist())

            t1 = time.time()

            if not camp_metrics:
                continue

            avg_metrics = {k: float(np.mean([r[k] for r in camp_metrics]))
                           for k in camp_metrics[0].keys()}
            avg_metrics["train_time_s"] = round(t1 - t0, 4)
            avg_metrics["infer_time_ms"] = round((t1 - t0) * 1000 / len(camp_metrics), 4)

            if model_name not in all_results:
                all_results[model_name] = {}
                all_raw_p50[model_name] = []
                all_raw_true[model_name] = []

            for k, v in avg_metrics.items():
                if k not in all_results[model_name]:
                    all_results[model_name][k] = []
                all_results[model_name][k].append(v)

            all_raw_p50[model_name].extend(camp_preds_list)
            all_raw_true[model_name].extend(camp_true_list)

            print(f"    {model_name:20s} RMSE={avg_metrics['rmse']:8.2f}  "
                  f"MAE={avg_metrics['mae']:8.2f}  Time={avg_metrics['train_time_s']:.3f}s")

    # Aggregate across folds
    final_results: Dict[str, Dict[str, float]] = {}
    for model_name in config.benchmark_models:
        final_results[model_name] = {}
        for k, v in all_results[model_name].items():
            final_results[model_name][k] = float(np.mean(v))

    # Save tables
    reporter.save_metrics_table(final_results, "exp01_benchmark")
    print("\nSaved: research/tables/exp01_benchmark.csv")

    rankings = compute_ranking(final_results)
    reporter.save_ranking_table(rankings, "exp01_benchmark")
    print("Saved: research/tables/exp01_benchmark_ranking.csv")

    # Significance tests (Diebold-Mariano)
    pairs = []
    models_list = config.benchmark_models
    for i, m1 in enumerate(models_list):
        for m2 in models_list[i + 1:]:
            e1 = np.array(all_raw_true[m1]) - np.array(all_raw_p50[m1])
            e2 = np.array(all_raw_true[m2]) - np.array(all_raw_p50[m2])
            dm_stat, p_val = diebold_mariano(e1, e2)
            pairs.append({
                "model_a": m1, "model_b": m2,
                "dm_stat": round(dm_stat, 4),
                "p_value": round(p_val, 6),
                "significant_005": p_val < 0.05,
                "better": m1 if dm_stat < 0 else m2,
            })
    reporter.save_significance_table(pairs, "exp01_benchmark")
    print("Saved: research/tables/exp01_benchmark_significance.csv")

    # Figures
    viz.benchmark_bar_chart(final_results, "rmse", "Forecast Benchmark: RMSE Comparison")
    viz.benchmark_bar_chart(final_results, "mae", "Forecast Benchmark: MAE Comparison")
    viz.benchmark_bar_chart(final_results, "mape", "Forecast Benchmark: MAPE Comparison")
    print("Saved: research/figures/benchmark_*.png")

    # Summary
    summary = {
        "experiment": "exp01_forecast_benchmark",
        "n_folds": config.cv_n_splits,
        "n_campaigns": int(feature_df["campaign_id"].nunique()),
        "total_rows": len(feature_df),
        "best_model_by_rmse": min(final_results, key=lambda m: final_results[m]["rmse"]),
        "best_model_by_mae": min(final_results, key=lambda m: final_results[m]["mae"]),
        "best_model_by_mape": min(final_results, key=lambda m: final_results[m]["mape"]),
        "ensemble_rmse": final_results.get("ensemble", {}).get("rmse", 0),
        "lightgbm_rmse": final_results.get("lightgbm", {}).get("rmse", 0),
    }
    reporter.save_summary(summary, "exp01_benchmark")
    print("Saved: research/reports/exp01_benchmark_summary.json")

    print("\nEXP01 complete.\n")
    return final_results
