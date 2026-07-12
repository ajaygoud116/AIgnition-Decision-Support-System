import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.decision.models import OptimizationReport
from src.models.common import ForecastResult, MetricType, SimulationResult
from src.uncertainty.models import UncertaintyReport
from src.utils.config import Config
from src.utils.logger import StructuredLogger


class ReportGenerator:
    """Generates CSV reports and a JSON summary from all engine outputs."""

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("report.generator")
        self._decimal_places = config.get("report.decimal_places", 2)
        self._output_format = config.get("report.output_format", "csv")
        self._include_raw = config.get("report.include_raw_forecasts", False)

    def generate(
        self,
        forecast_result: ForecastResult,
        uncertainty_report: UncertaintyReport,
        simulation_results: List[SimulationResult],
        optimization_report: OptimizationReport,
        output_dir: Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._write_forecasts(forecast_result, output_dir)
        self._write_uncertainty(uncertainty_report, output_dir)
        self._write_simulations(simulation_results, output_dir)
        self._write_recommendations(optimization_report, output_dir)
        self._write_summary(
            forecast_result, uncertainty_report,
            simulation_results, optimization_report,
            output_dir,
        )

        self._logger.info(
            "report_generated",
            output_dir=str(output_dir),
            forecasts=len(forecast_result.series),
            uncertainties=len(uncertainty_report.entities),
            simulations=len(simulation_results),
            recommendations=len(optimization_report.recommendations),
        )

    def _write_forecasts(
        self, forecast_result: ForecastResult, output_dir: Path
    ) -> None:
        rows = []
        for series in forecast_result.series:
            if not self._include_raw and series.metric != MetricType.REVENUE:
                continue
            for point in series.points:
                rows.append({
                    "entity_id": series.entity_id,
                    "channel": series.channel,
                    "horizon": series.horizon.value,
                    "metric": series.metric.value,
                    "date": str(point.date),
                    "p10": round(point.values.p10, self._decimal_places),
                    "p50": round(point.values.p50, self._decimal_places),
                    "p90": round(point.values.p90, self._decimal_places),
                })
        pdf = pd.DataFrame(rows, columns=["entity_id", "channel", "horizon", "metric", "date", "p10", "p50", "p90"])
        pdf.to_csv(output_dir / "forecasts.csv", index=False)

    def _write_uncertainty(
        self, uncertainty_report: UncertaintyReport, output_dir: Path
    ) -> None:
        rows = []
        for e in uncertainty_report.entities:
            rows.append({
                "entity_id": e.entity_id,
                "channel": e.channel,
                "confidence_score": round(e.confidence_score, self._decimal_places),
                "volatility": round(e.volatility, self._decimal_places),
                "stability_trend": e.stability_trend,
                "avg_relative_width": round(e.avg_relative_width, self._decimal_places),
            })
        pdf = pd.DataFrame(rows, columns=["entity_id", "channel", "confidence_score", "volatility", "stability_trend", "avg_relative_width"])
        pdf.to_csv(output_dir / "uncertainty.csv", index=False)

    def _write_simulations(
        self, simulation_results: List[SimulationResult], output_dir: Path
    ) -> None:
        rows = []
        for r in simulation_results:
            rows.append({
                "scenario": r.scenario.label,
                "projected_revenue": round(r.projected_revenue, self._decimal_places),
                "projected_spend": round(r.projected_spend, self._decimal_places),
                "projected_roas": round(r.projected_roas, self._decimal_places),
            })
        pdf = pd.DataFrame(rows, columns=["scenario", "projected_revenue", "projected_spend", "projected_roas"])
        pdf.to_csv(output_dir / "simulations.csv", index=False)

    def _write_recommendations(
        self, optimization_report: OptimizationReport, output_dir: Path
    ) -> None:
        rows = []
        for r in optimization_report.recommendations:
            rows.append({
                "entity_id": r.entity_id,
                "channel": r.channel,
                "current_budget": round(r.current_budget, self._decimal_places),
                "recommended_budget": round(r.recommended_budget, self._decimal_places),
                "budget_change": round(
                    r.recommended_budget - r.current_budget, self._decimal_places
                ),
                "expected_roas": round(r.expected_roas, self._decimal_places) if r.expected_roas else "",
                "rationale": r.rationale,
            })
        pdf = pd.DataFrame(rows, columns=["entity_id", "channel", "current_budget", "recommended_budget", "budget_change", "expected_roas", "rationale"])
        pdf.to_csv(output_dir / "recommendations.csv", index=False)

    def _write_summary(
        self,
        forecast_result: ForecastResult,
        uncertainty_report: UncertaintyReport,
        simulation_results: List[SimulationResult],
        optimization_report: OptimizationReport,
        output_dir: Path,
    ) -> None:
        total_rev = sum(
            p.values.p50 for s in forecast_result.series
            if s.metric == MetricType.REVENUE
            for p in s.points
        )
        campaigns = len(set(s.entity_id for s in forecast_result.series))

        summary = {
            "campaigns_forecasted": campaigns,
            "total_forecast_revenue_p50": round(total_rev, self._decimal_places),
            "overall_uncertainty_confidence": round(
                uncertainty_report.overall_confidence, self._decimal_places
            ),
            "overall_volatility": round(
                uncertainty_report.overall_volatility, self._decimal_places
            ),
            "high_uncertainty_campaigns": uncertainty_report.high_uncertainty_count,
            "scenarios_simulated": len(simulation_results),
            "recommendations": len(optimization_report.recommendations),
            "campaigns_flagged": optimization_report.summary.get("campaigns_flagged", 0),
            "total_flags": optimization_report.summary.get("total_flags", 0),
        }

        with open(output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
