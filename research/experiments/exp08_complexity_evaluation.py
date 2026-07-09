"""Experiment 8: Complexity Evaluation.

Measures training time, inference time, and memory usage
for each model using a consistent 90-day future date range.
"""

import time
import tracemalloc
from typing import Dict

import numpy as np
import pandas as pd

from src.forecasting.lightgbm_model import LightGBMQuantileForecaster
from src.utils.config import Config as SrcConfig

from research.core.config import ExperimentConfig
from research.core.data import ExperimentData
from research.core.reporting import ExperimentReporter


def run(config: ExperimentConfig, src_config: SrcConfig) -> dict:
    print("=" * 60)
    print("EXP08: Complexity Evaluation")
    print("=" * 60)

    data = ExperimentData(src_config)
    feature_df = data.get_feature_data()
    print(f"Loaded feature DataFrame: {feature_df.shape}")

    reporter = ExperimentReporter(config.output_dir)
    n_features = len(feature_df.columns)
    n_campaigns = int(feature_df["campaign_id"].nunique())

    np.random.seed(config.random_seed)
    all_campaigns = feature_df["campaign_id"].unique()
    n_test_campaigns = min(3, len(all_campaigns))
    test_campaigns = np.random.choice(all_campaigns, n_test_campaigns, replace=False)
    small_df = feature_df[feature_df["campaign_id"].isin(test_campaigns)].copy()

    last_date = small_df["date"].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=90)
    future_di = pd.DatetimeIndex(future_dates)

    future_rows = []
    for cid in test_campaigns:
        camp = small_df[small_df["campaign_id"] == cid]
        last_row = camp.iloc[-1]
        for d in future_dates:
            row = last_row.copy()
            row["date"] = d
            future_rows.append(row)
    future_df = pd.DataFrame(future_rows)
    n_predict_rows = len(future_df)

    results: Dict[str, Dict[str, float]] = {}

    # -- LightGBM --
    print("  Benchmarking LightGBM...")
    lgb = LightGBMQuantileForecaster(random_seed=config.random_seed)

    tracemalloc.start()
    t0 = time.time()
    lgb.fit(small_df, target_col="revenue")
    t1 = time.time()
    _, peak_mem_lgb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    t2 = time.time()
    lgb.predict(future_df)
    t3 = time.time()
    _, peak_mem_lgb_pred = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    lgb_train = t1 - t0
    lgb_infer = (t3 - t2)
    lgb_infer_per = lgb_infer / n_predict_rows * 1000

    results["lightgbm"] = {
        "train_time_s": round(lgb_train, 4),
        "infer_time_s": round(lgb_infer, 4),
        "infer_time_ms_per_row": round(lgb_infer_per, 4),
        "peak_memory_fit_mb": round(peak_mem_lgb / 1e6, 2),
        "peak_memory_pred_mb": round(peak_mem_lgb_pred / 1e6, 2),
        "n_features": n_features,
        "n_rows_train": len(small_df),
        "n_rows_predict": n_predict_rows,
    }
    print(f"    Train: {lgb_train:.3f}s, Infer: {lgb_infer_per:.4f}ms/row, "
          f"Mem: {peak_mem_lgb / 1e6:.1f}MB | {n_predict_rows} rows")

    # -- Seasonal Naive --
    print("  Benchmarking Seasonal Naive...")
    from src.forecasting.seasonal_naive import SeasonalNaiveForecaster
    sn = SeasonalNaiveForecaster(window=7)

    tracemalloc.start()
    t0 = time.time()
    sn.fit(small_df, target_col="revenue")
    t1 = time.time()
    _, peak_mem_sn = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Warm-up predict (not timed)
    for cid in test_campaigns:
        sn.predict(cid, future_di)

    t2 = time.time()
    for cid in test_campaigns:
        sn.predict(cid, future_di)
    t3 = time.time()

    sn_train = t1 - t0
    sn_infer = (t3 - t2)
    sn_infer_per = sn_infer / (len(test_campaigns) * 90) * 1000

    results["seasonal_naive"] = {
        "train_time_s": round(sn_train, 4),
        "infer_time_s": round(sn_infer, 4),
        "infer_time_ms_per_row": round(sn_infer_per, 4),
        "peak_memory_fit_mb": round(peak_mem_sn / 1e6, 2),
        "peak_memory_pred_mb": 0.0,
        "n_features": 0,
        "n_rows_train": len(small_df),
        "n_rows_predict": n_predict_rows,
    }
    print(f"    Train: {sn_train:.4f}s, Infer: {sn_infer_per:.4f}ms/row, "
          f"Mem: {peak_mem_sn / 1e6:.1f}MB")

    # -- Ensemble (timed as lightweight combine, no re-fit) --
    print("  Benchmarking Ensemble (LGB + SN)...")
    from src.forecasting.ensemble import EnsembleForecaster
    ensemble = EnsembleForecaster(lgb_weight=0.5, sn_weight=0.5)

    # Build SN array matching LGB predict output shape
    sn_all = []
    for cid in test_campaigns:
        sp = sn.predict(cid, future_di)
        sn_all.extend(sp)
    sn_array = np.array(sn_all).reshape(-1, 1)
    sn_array = np.tile(sn_array, (1, 3))

    # Get LGB predictions for the same data
    t4 = time.time()
    lgb_preds = lgb.predict(future_df)
    combined = ensemble.combine(lgb_preds, sn_array)
    t5 = time.time()

    ens_infer = t5 - t4
    ens_infer_per = ens_infer / n_predict_rows * 1000

    results["ensemble"] = {
        "train_time_s": round(lgb_train + sn_train, 4),
        "infer_time_s": round(ens_infer, 4),
        "infer_time_ms_per_row": round(ens_infer_per, 4),
        "peak_memory_fit_mb": round((peak_mem_lgb + peak_mem_sn) / 1e6, 2),
        "peak_memory_pred_mb": 0.0,
        "n_features": n_features,
        "n_rows_train": len(small_df),
        "n_rows_predict": n_predict_rows,
    }
    print(f"    Train: {lgb_train + sn_train:.3f}s, Infer: {ens_infer_per:.4f}ms/row")

    reporter.save_metrics_table(results, "exp08_complexity")
    print("Saved: research/tables/exp08_complexity.csv")

    summary = {
        "experiment": "exp08_complexity_evaluation",
        "lightgbm_train_s": results["lightgbm"]["train_time_s"],
        "lightgbm_infer_ms_per_row": results["lightgbm"]["infer_time_ms_per_row"],
        "lightgbm_memory_mb": results["lightgbm"]["peak_memory_fit_mb"],
        "seasonal_naive_infer_ms_per_row": results["seasonal_naive"]["infer_time_ms_per_row"],
        "ensemble_train_s": results["ensemble"]["train_time_s"],
        "ensemble_infer_ms_per_row": results["ensemble"]["infer_time_ms_per_row"],
        "n_features": n_features,
        "n_campaigns": n_campaigns,
    }
    reporter.save_summary(summary, "exp08_complexity")

    print(f"\n  Complexity Summary:")
    print(f"    LightGBM:     Train={results['lightgbm']['train_time_s']:.3f}s  "
          f"Infer={results['lightgbm']['infer_time_ms_per_row']:.4f}ms/row  "
          f"Mem={results['lightgbm']['peak_memory_fit_mb']:.1f}MB")
    print(f"    Ensemble:     Train={results['ensemble']['train_time_s']:.3f}s  "
          f"Infer={results['ensemble']['infer_time_ms_per_row']:.4f}ms/row")

    print("\nEXP08 complete.\n")
    return results
