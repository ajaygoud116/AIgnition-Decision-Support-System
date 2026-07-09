from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class ExperimentVisualizer:
    """Generates all publication-quality figures for experiments."""

    def __init__(self, output_dir: Path):
        self._output_dir = Path(output_dir)
        self._figures_dir = self._output_dir / "figures"
        self._figures_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, name: str, dpi: int = 150):
        path = self._figures_dir / name
        plt.tight_layout()
        plt.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close()
        return path

    def benchmark_bar_chart(
        self, results: Dict[str, Dict[str, float]],
        metric: str = "rmse",
        title: str = "Model Comparison",
    ) -> Path:
        models = list(results.keys())
        values = [results[m].get(metric, 0) for m in models]
        colors = ["#e74c3c" if m == "ensemble" else "#3498db" for m in models]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.bar(range(len(models)), values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        return self._save(f"benchmark_{metric}.png")

    def benchmark_multi_metric_chart(self, results: Dict[str, Dict[str, float]],
                                     metrics: List[str]) -> Path:
        models = list(results.keys())
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 4))

        for ax, metric in zip(axes, metrics):
            values = [results[m].get(metric, 0) for m in models]
            colors = ["#e74c3c" if m == "ensemble" else "#3498db" for m in models]
            ax.bar(range(len(models)), values, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha="right", fontsize=7)
            ax.set_title(metric.upper(), fontsize=10)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Forecast Benchmark: Multi-Metric Comparison", fontsize=13, fontweight="bold")
        return self._save("benchmark_multi_metric.png")

    def calibration_curve(self, nominal_quantiles: np.ndarray,
                          empirical_coverage: np.ndarray,
                          model_name: str = "ensemble") -> Path:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(nominal_quantiles, empirical_coverage, "o-", color="#3498db",
                linewidth=2, markersize=6, label=model_name)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        ax.fill_between(nominal_quantiles, nominal_quantiles - 0.05,
                        nominal_quantiles + 0.05, alpha=0.15, color="gray",
                        label="±5% zone")
        ax.set_xlabel("Nominal Quantile", fontsize=11)
        ax.set_ylabel("Empirical Coverage", fontsize=11)
        ax.set_title("Calibration Curve", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return self._save("calibration_curve.png")

    def reliability_diagram(self, predicted_probs: np.ndarray,
                            actual_freqs: np.ndarray) -> Path:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(predicted_probs, actual_freqs, "o-", color="#2ecc71",
                linewidth=2, markersize=6)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("Predicted Confidence", fontsize=11)
        ax.set_ylabel("Observed Frequency", fontsize=11)
        ax.set_title("Reliability Diagram", fontsize=13, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return self._save("reliability_diagram.png")

    def residual_plot(self, y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "Residuals vs Predicted") -> Path:
        residuals = y_true - y_pred
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.scatter(y_pred, residuals, alpha=0.4, s=10, color="#3498db")
        ax1.axhline(y=0, color="red", linestyle="--", linewidth=1)
        ax1.set_xlabel("Predicted", fontsize=11)
        ax1.set_ylabel("Residual (Actual - Predicted)", fontsize=11)
        ax1.set_title(title, fontsize=12)
        ax1.grid(alpha=0.3)

        ax2.hist(residuals, bins=50, color="#3498db", edgecolor="black", alpha=0.7)
        ax2.axvline(x=0, color="red", linestyle="--", linewidth=1)
        ax2.set_xlabel("Residual", fontsize=11)
        ax2.set_ylabel("Frequency", fontsize=11)
        ax2.set_title("Residual Distribution", fontsize=12)
        ax2.grid(alpha=0.3)

        return self._save("residual_plot.png")

    def horizon_degradation(self, horizon_metrics: Dict[int, Dict[str, float]]) -> Path:
        horizons = sorted(horizon_metrics.keys())
        fig, ax = plt.subplots(figsize=(8, 5))
        for metric in ["rmse", "mae", "coverage_90"]:
            if metric not in horizon_metrics[horizons[0]]:
                continue
            values = [horizon_metrics[h][metric] for h in horizons]
            ax.plot(horizons, values, "o-", linewidth=2, label=metric.upper())

        ax.set_xlabel("Forecast Horizon (days)", fontsize=11)
        ax.set_ylabel("Metric Value", fontsize=11)
        ax.set_title("Forecast Degradation by Horizon", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        return self._save("horizon_degradation.png")

    def strategy_comparison(self, results: Dict[str, Dict[str, float]],
                            metrics: List[str]) -> Path:
        strategies = list(results.keys())
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

        for ax, metric in zip(axes, metrics):
            values = [results[s].get(metric, 0) for s in strategies]
            colors = ["#e74c3c" if s == "optimizer" else "#2ecc71" if s == "equal" else "#3498db"
                      for s in strategies]
            ax.bar(range(len(strategies)), values, color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels(strategies, rotation=30, ha="right", fontsize=8)
            ax.set_title(metric.replace("_", " ").title(), fontsize=10)
            ax.grid(axis="y", alpha=0.3)

        plt.suptitle("Business Strategy Comparison", fontsize=13, fontweight="bold")
        return self._save("strategy_comparison.png")

    def sensitivity_heatmap(self, results: Dict[str, Dict[str, float]],
                            base_metric: str = "rmse") -> Path:
        scenarios = list(results.keys())
        metrics = [k for k in results[scenarios[0]].keys() if k != "delta_pct"]

        data = np.zeros((len(scenarios), len(metrics)))
        for i, s in enumerate(scenarios):
            for j, m in enumerate(metrics):
                data[i, j] = results[s].get(m, 0)

        fig, ax = plt.subplots(figsize=(8, max(4, len(scenarios) * 0.5)))
        sns.heatmap(data, annot=True, fmt=".2f", xticklabels=metrics,
                    yticklabels=scenarios, cmap="RdYlGn_r", ax=ax,
                    cbar_kws={"label": "RMSE"}, linewidths=0.5)
        ax.set_title("Sensitivity Analysis: RMSE by Stress Scenario", fontsize=13, fontweight="bold")
        return self._save("sensitivity_heatmap.png")

    def feature_importance_chart(self, importance_dict: Dict[str, float],
                                 title: str = "Feature Importance",
                                 top_n: int = 20) -> Path:
        items = sorted(importance_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        names, scores = zip(*items)

        fig, ax = plt.subplots(figsize=(8, max(4, len(names) * 0.35)))
        colors = ["#e74c3c" if v < 0 else "#3498db" for v in scores]
        ax.barh(range(len(names)), scores, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()
        return self._save("feature_importance.png")

    def worst_forecasts_plot(self, worst_df: pd.DataFrame, n: int = 5) -> Path:
        campaigns = worst_df["campaign_id"].unique()[:n]
        fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
        if n == 1:
            axes = [axes]

        for ax, cid in zip(axes, campaigns):
            cdf = worst_df[worst_df["campaign_id"] == cid]
            ax.plot(cdf["date"], cdf["actual"], "o-", color="#3498db", markersize=3, label="Actual")
            ax.plot(cdf["date"], cdf["predicted"], "x-", color="#e74c3c", markersize=3, label="Predicted")
            ax.fill_between(cdf["date"], cdf["p10"], cdf["p90"], alpha=0.2, color="gray", label="80% PI")
            ax.set_title(f"Campaign: {cid}  (RMSE={cdf['rmse'].iloc[0]:.1f})", fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

        plt.suptitle("Worst Forecasts", fontsize=13, fontweight="bold")
        return self._save("worst_forecasts.png")
