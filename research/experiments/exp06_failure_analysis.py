"""Experiment 6: Failure Analysis.

Identifies the worst forecasts, analyzes residuals, and measures
how forecast quality degrades with horizon.
"""

import numpy as np
import pandas as pd

from src.forecasting.forecaster import Forecaster
from src.utils.config import Config as SrcConfig

from research.core.config import ExperimentConfig
from research.core.data import ExperimentData
from research.core.metrics import ForecastMetrics
from research.core.visualization import ExperimentVisualizer
from research.core.reporting import ExperimentReporter
from research.core.walkforward import ExpandingWindowCV


def run(config: ExperimentConfig, src_config: SrcConfig) -> dict:
    print("=" * 60)
    print("EXP06: Failure Analysis")
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

    all_residuals = []
    all_actuals = []
    all_preds = []
    all_p10 = []
    all_p90 = []
    campaign_errors = {}
    horizon_metrics = {h: {"rmse": [], "mae": [], "coverage_90": []}
                       for h in [30, 60, 90]}

    for fold_idx, (train_df, test_df) in enumerate(cv.split(feature_df)):
        forecaster = Forecaster(src_config)
        forecaster.fit(train_df)
        forecast_result =         forecaster.predict(train_df)

        for series in forecast_result.series:
            cid = series.entity_id
            h = series.horizon.value

            for point in series.points:
                actual_row = test_df[(test_df["campaign_id"] == cid) &
                                     (test_df["date"] == pd.Timestamp(point.date))]
                if actual_row.empty:
                    continue
                actual = actual_row["revenue"].values[0]
                pred_p50 = point.values.p50
                pred_p10 = point.values.p10
                pred_p90 = point.values.p90

                resid = actual - pred_p50
                all_residuals.append(resid)
                all_actuals.append(actual)
                all_preds.append(pred_p50)
                all_p10.append(pred_p10)
                all_p90.append(pred_p90)

                if cid not in campaign_errors:
                    campaign_errors[cid] = {"residuals": [], "actuals": [], "preds": [],
                                            "p10": [], "p90": [], "dates": []}
                campaign_errors[cid]["residuals"].append(resid)
                campaign_errors[cid]["actuals"].append(actual)
                campaign_errors[cid]["preds"].append(pred_p50)
                campaign_errors[cid]["p10"].append(pred_p10)
                campaign_errors[cid]["p90"].append(pred_p90)
                campaign_errors[cid]["dates"].append(point.date)

                if h in horizon_metrics:
                    err = abs(resid)
                    horizon_metrics[h]["rmse"].append(resid ** 2)
                    horizon_metrics[h]["mae"].append(err)
                    horizon_metrics[h]["coverage_90"].append(
                        1 if pred_p10 <= actual <= pred_p90 else 0
                    )

    # Compute campaign-level RMSE
    campaign_rmse = {}
    for cid, data_dict in campaign_errors.items():
        residuals = np.array(data_dict["residuals"])
        campaign_rmse[cid] = float(np.sqrt(np.mean(residuals ** 2)))

    # Worst campaigns
    sorted_campaigns = sorted(campaign_rmse.items(), key=lambda x: x[1], reverse=True)
    worst_5 = sorted_campaigns[:5]

    worst_rows = []
    for cid, rmse_val in worst_5:
        cd = campaign_errors[cid]
        for i in range(len(cd["dates"])):
            worst_rows.append({
                "campaign_id": cid,
                "date": str(cd["dates"][i]),
                "actual": round(cd["actuals"][i], 2),
                "predicted": round(cd["preds"][i], 2),
                "p10": round(cd["p10"][i], 2),
                "p90": round(cd["p90"][i], 2),
                "residual": round(cd["residuals"][i], 2),
                "rmse": round(rmse_val, 2),
            })
    worst_df = pd.DataFrame(worst_rows)
    reporter.save_dataframe(worst_df, "exp06_worst_forecasts")
    print("Saved: research/tables/exp06_worst_forecasts.csv")

    # Horizon degradation
    h_results = {}
    for h in sorted(horizon_metrics.keys()):
        hm = horizon_metrics[h]
        h_results[h] = {
            "rmse": float(np.sqrt(np.mean(hm["rmse"]))),
            "mae": float(np.mean(hm["mae"])),
            "coverage_90": float(np.mean(hm["coverage_90"])),
        }
        print(f"  Horizon {h:2d}d: RMSE={h_results[h]['rmse']:.2f}  "
              f"MAE={h_results[h]['mae']:.2f}  Coverage={h_results[h]['coverage_90']:.2f}")

    reporter.save_metrics_table(h_results, "exp06_horizon_degradation")
    print("Saved: research/tables/exp06_horizon_degradation.csv")

    # Residual plot
    viz.residual_plot(np.array(all_actuals), np.array(all_preds))
    print("  Saved: research/figures/residual_plot.png")

    # Horizon degradation plot
    viz.horizon_degradation(h_results)
    print("  Saved: research/figures/horizon_degradation.png")

    # Worst forecasts plot
    if not worst_df.empty:
        viz.worst_forecasts_plot(worst_df)
        print("  Saved: research/figures/worst_forecasts.png")

    summary = {
        "experiment": "exp06_failure_analysis",
        "total_predictions": len(all_residuals),
        "overall_rmse": round(float(np.sqrt(np.mean(np.array(all_residuals) ** 2))), 2),
        "overall_mae": round(float(np.mean(np.abs(all_residuals))), 2),
        "worst_campaign": worst_5[0][0] if worst_5 else "",
        "worst_campaign_rmse": round(worst_5[0][1], 2) if worst_5 else 0,
        "best_campaign": sorted_campaigns[-1][0] if sorted_campaigns else "",
        "best_campaign_rmse": round(sorted_campaigns[-1][1], 2) if sorted_campaigns else 0,
        "rmse_degradation_30_to_90": round(
            h_results.get(90, {}).get("rmse", 0) - h_results.get(30, {}).get("rmse", 0), 2
        ),
    }
    reporter.save_summary(summary, "exp06_failure")

    print("\nEXP06 complete.\n")
    return summary
