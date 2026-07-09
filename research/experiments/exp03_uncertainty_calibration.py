"""Experiment 3: Uncertainty Calibration.

Validates the UncertaintyEngine's confidence scores against empirical coverage.
Uses synthetic data with KNOWN quantile structure to measure calibration error.
"""

import numpy as np

from src.forecasting.lightgbm_model import LightGBMQuantileForecaster
from src.utils.config import Config as SrcConfig

from research.core.config import ExperimentConfig
from research.core.data import ExperimentData
from research.core.metrics import ForecastMetrics
from research.core.visualization import ExperimentVisualizer
from research.core.reporting import ExperimentReporter


def run(config: ExperimentConfig, src_config: SrcConfig) -> dict:
    print("=" * 60)
    print("EXP03: Uncertainty Calibration")
    print("=" * 60)

    metrics_calc = ForecastMetrics()
    viz = ExperimentVisualizer(config.output_dir)
    reporter = ExperimentReporter(config.output_dir)

    # Generate synthetic data with known quantiles
    data = ExperimentData(src_config)
    feat_df, y_true, true_p10, true_p50, true_p90 = data.synthetic_quantile_data(
        n_samples=2000, n_features=10, seed=config.random_seed
    )
    print(f"Generated synthetic calibration data: {feat_df.shape}")

    # Split into train/test temporally
    split_idx = len(feat_df) // 2
    train_df = feat_df.iloc[:split_idx].copy()
    test_df = feat_df.iloc[split_idx:].copy()
    true_test_p50 = true_p50[split_idx:]

    # Train LightGBM only (SN would predict 0 for synthetic entity)
    lgb = LightGBMQuantileForecaster(
        quantiles=[0.1, 0.5, 0.9],
        random_seed=config.random_seed,
    )
    lgb.fit(train_df, target_col="revenue")

    lgb_preds = lgb.predict(test_df)
    pred_p10 = lgb_preds[:, 0]
    pred_p50 = lgb_preds[:, 1]
    pred_p90 = lgb_preds[:, 2]

    # 1. Coverage analysis
    test_actuals = test_df["revenue"].values
    min_len = min(len(test_actuals), len(pred_p50))
    test_actuals = test_actuals[:min_len]
    pred_p10 = pred_p10[:min_len]
    pred_p50 = pred_p50[:min_len]
    pred_p90 = pred_p90[:min_len]

    cov_80 = metrics_calc.coverage(test_actuals, pred_p10, pred_p90)
    cov_p10 = metrics_calc.calibration_error(test_actuals, pred_p10, 0.1)
    cov_p50 = metrics_calc.calibration_error(test_actuals, pred_p50, 0.5)
    cov_p90 = metrics_calc.calibration_error(test_actuals, pred_p90, 0.9)

    print(f"\n  Coverage Analysis:")
    print(f"    80% PI empirical coverage: {cov_80:.4f} (ideal: 0.80)")
    print(f"    p10 calibration error:     {cov_p10:.4f}")
    print(f"    p50 calibration error:     {cov_p50:.4f}")
    print(f"    p90 calibration error:     {cov_p90:.4f}")

    # 2. Full calibration curve across quantile levels
    quantile_levels = np.arange(0.05, 1.0, 0.05)
    empirical_coverage = []
    for q in quantile_levels:
        pred_q = np.percentile(pred_p50, q * 100)
        emp = float(np.mean(test_actuals <= pred_q))
        empirical_coverage.append(emp)

    viz.calibration_curve(quantile_levels, np.array(empirical_coverage), "Ensemble")
    print("  Saved: research/figures/calibration_curve.png")

    # 3. Comparison: our model vs true quantiles on synthetic data
    # Only on the subset where we have ground truth
    true_test_p10 = true_p10[split_idx:][:min_len]
    true_test_p90 = true_p90[split_idx:][:min_len]

    true_cov_80 = metrics_calc.coverage(test_actuals, true_test_p10, true_test_p90)
    model_cov_80 = cov_80
    print(f"\n  True 80% PI coverage: {true_cov_80:.4f}")
    print(f"  Model 80% PI coverage: {model_cov_80:.4f}")

    # 4. Interval width / sharpness
    model_width = metrics_calc.interval_width(pred_p10, pred_p90)
    true_width = metrics_calc.interval_width(true_test_p10, true_test_p90)
    print(f"\n  Mean interval width (model): {model_width:.4f}")
    print(f"  Mean interval width (true):  {true_width:.4f}")

    # 5. Reliability diagram
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    conf_scores = 1.0 - np.abs(pred_p50 - test_actuals) / (np.maximum(pred_p90 - pred_p10, 1e-10))
    conf_scores = np.clip(conf_scores, 0, 1)

    actual_freqs = []
    for i in range(n_bins):
        mask = (conf_scores >= bin_edges[i]) & (conf_scores < bin_edges[i + 1])
        if mask.sum() > 0:
            actual_freqs.append(float(np.mean(np.abs(pred_p50[mask] - test_actuals[mask]) <
                                               (pred_p90[mask] - pred_p10[mask]) / 2)))
        else:
            actual_freqs.append(0.0)

    viz.reliability_diagram(bin_centers, np.array(actual_freqs))
    print("  Saved: research/figures/reliability_diagram.png")

    # Save metrics
    results = {
        "ensemble_calibration": {
            "empirical_coverage_80": round(cov_80, 4),
            "calibration_error_p10": round(cov_p10, 4),
            "calibration_error_p50": round(cov_p50, 4),
            "calibration_error_p90": round(cov_p90, 4),
            "model_interval_width": round(model_width, 4),
            "true_interval_width": round(true_width, 4),
            "n_samples": min_len,
        }
    }
    reporter.save_metrics_table(results, "exp03_calibration")
    print("Saved: research/tables/exp03_calibration.csv")

    summary = {
        "experiment": "exp03_uncertainty_calibration",
        "empirical_coverage_80pct": round(cov_80, 4),
        "calibration_well_calibrated": cov_80 > 0.75,
        "model_interval_width": round(model_width, 4),
        "true_interval_width": round(true_width, 4),
    }
    reporter.save_summary(summary, "exp03_calibration")

    print("\nEXP03 complete.\n")
    return results
