import pickle
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.decision.engine import DecisionEngine
from src.features.builder import FeatureBuilder
from src.forecasting.forecaster import Forecaster
from src.ingestion.pipeline import IngestionPipeline
from src.models.common import (
    BudgetAdjustment,
    ForecastResult,
    SimulationScenario,
)
from src.report.generator import ReportGenerator
from src.simulation.baselines import extract_baselines
from src.simulation.simulator import ScenarioSimulator
from src.uncertainty.engine import UncertaintyEngine
from src.utils.config import Config
from src.utils.logger import StructuredLogger
from src.validation.validator import ValidationEngine


class PipelineRunner:
    """End-to-end forecasting pipeline.

    Supports two modes:
      - train:  ingest → validate → build features → fit → pickle model
      - predict: ingest → validate → build features → load model → forecast
                 → uncertainty → simulate → decide → report
    """

    def __init__(self, config: Optional[Config] = None):
        self._config = config or Config()
        self._logger = StructuredLogger("pipeline.runner")
        self._forecaster: Optional[Forecaster] = None  # kept for scenario sim

    def run(
        self,
        data_dir: Path,
        model_path: Path,
        output_dir: Path,
        force_retrain: bool = False,
    ) -> None:
        data_dir = Path(data_dir)
        model_path = Path(model_path)
        output_dir = Path(output_dir)

        self._logger.info("pipeline_start", data_dir=str(data_dir))

        df = self._ingest(data_dir)
        self._validate(df)
        feature_df = self._build_features(df)

        # ── Conformal calibration hold-out ──────────────────────────────
        cal_days = self._config.get("uncertainty.calibration_holdout_days", 14)
        cal_actuals = self._extract_calibration_set(feature_df, holdout_days=cal_days)

        forecast_result = self._forecast(feature_df, model_path, force_retrain)

        # ── Calibrate uncertainty intervals ─────────────────────────────
        uncertainty_engine = UncertaintyEngine(self._config)
        if cal_actuals is not None and len(cal_actuals) >= 100:
            cal_forecast = self._forecast_calibration(feature_df, holdout_days=cal_days)
            if cal_forecast is not None:
                uncertainty_engine.calibrate(cal_forecast, cal_actuals)
                self._logger.info(
                    "calibration_applied",
                    alpha=round(uncertainty_engine._calibration_alpha or 1.0, 4),
                )
            else:
                self._logger.info("calibration_skipped", reason="calibration_forecast_failed")
        else:
            self._logger.info("calibration_skipped", reason="insufficient_calibration_data")

        uncertainty_report = uncertainty_engine.compute(forecast_result)
        baselines = extract_baselines(feature_df)
        simulation_results = self._simulate(forecast_result, baselines, feature_df)
        optimization_report = self._decide(
            forecast_result, uncertainty_report, baselines, feature_df,
        )

        report = ReportGenerator(self._config)
        report.generate(
            forecast_result, uncertainty_report,
            simulation_results, optimization_report,
            output_dir,
        )

        self._logger.info("pipeline_complete", output_dir=str(output_dir))

    def _extract_calibration_set(
        self, feature_df: pd.DataFrame, holdout_days: int = 14,
    ) -> Optional[pd.DataFrame]:
        """Hold out the last *holdout_days* of each campaign for calibration.

        Returns a DataFrame containing ONLY the held-out rows (raw metrics
        plus campaign_id + date) to serve as calibration actuals.
        Returns None if no campaign has enough data.
        """
        if "campaign_id" not in feature_df.columns:
            return None

        cal_rows = []
        for _, grp in feature_df.sort_values(["campaign_id", "date"]).groupby("campaign_id", sort=False):
            if len(grp) <= holdout_days + 10:
                continue
            cal_rows.append(grp.tail(holdout_days))

        if not cal_rows:
            return None
        result = pd.concat(cal_rows, ignore_index=True)
        self._logger.info(
            "calibration_set_extracted",
            rows=len(result),
            campaigns=result["campaign_id"].nunique(),
            holdout_days=holdout_days,
        )
        return result

    def _forecast_calibration(
        self, feature_df: pd.DataFrame, holdout_days: int = 14,
    ) -> Optional[ForecastResult]:
        train_rows = []
        for _, grp in feature_df.sort_values(["campaign_id", "date"]).groupby("campaign_id", sort=False):
            if len(grp) <= holdout_days + 10:
                continue
            train_rows.append(grp.head(len(grp) - holdout_days))
        if not train_rows:
            return None

        train_df = pd.concat(train_rows, ignore_index=True)
        cal_forecaster = Forecaster(self._config)
        cal_forecaster.fit(train_df)
        cal_forecast = cal_forecaster.predict(train_df)
        self._logger.info("calibration_forecast_complete", series=len(cal_forecast.series))
        return cal_forecast

    def _ingest(self, data_dir: Path) -> pd.DataFrame:
        self._logger.info("step_ingest")
        pipeline = IngestionPipeline()
        df = pipeline.ingest_all(data_dir)
        if df.empty:
            self._logger.error("ingestion_empty")
            sys.exit(1)
        self._logger.info("step_ingest_done", rows=len(df))
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        self._logger.info("step_validate")
        validator = ValidationEngine(self._config)
        report = validator.validate(df)
        self._logger.info("step_validate_done", issues=len(report.errors) + len(report.warnings))

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        self._logger.info("step_features")
        builder = FeatureBuilder(self._config)
        feature_df = builder.build(df)
        self._logger.info("step_features_done", rows=len(feature_df), cols=len(feature_df.columns))
        return feature_df

    def _forecast(
        self,
        feature_df: pd.DataFrame,
        model_path: Path,
        force_retrain: bool,
    ) -> ForecastResult:
        self._logger.info("step_forecast")

        if model_path.exists() and not force_retrain:
            self._logger.info("load_model", path=str(model_path))
            with open(model_path, "rb") as f:
                self._forecaster = pickle.load(f)
        else:
            self._logger.info("train_model")
            self._forecaster = Forecaster(self._config)
            self._forecaster.fit(feature_df)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(self._forecaster, f)
            self._logger.info("model_saved", path=str(model_path))

        forecast_result = self._forecaster.predict(feature_df)
        self._logger.info(
            "step_forecast_done",
            series=len(forecast_result.series),
            campaigns=forecast_result.metadata.get("campaigns_forecasted", 0),
        )
        return forecast_result

    def _compute_uncertainty(
        self, forecast_result: ForecastResult,
    ) -> "UncertaintyReport":
        """Produces an UncertaintyReport (no calibration).

        DEPRECATED: use UncertaintyEngine directly with calibrate().
        Retained for backward compatibility.
        """
        self._logger.info("step_uncertainty")
        engine = UncertaintyEngine(self._config)
        report = engine.compute(forecast_result)
        self._logger.info(
            "step_uncertainty_done",
            entities=len(report.entities),
            high_uncertainty=report.high_uncertainty_count,
        )
        return report

    def _simulate(
        self,
        forecast_result: ForecastResult,
        baselines: dict,
        feature_df: Optional[pd.DataFrame] = None,
    ) -> List:
        self._logger.info("step_simulation")
        scenarios = self._default_scenarios(baselines)
        simulator = ScenarioSimulator(self._config)
        results = simulator.simulate(
            scenarios, forecast_result, baselines,
            forecaster=self._forecaster,
            feature_df=feature_df,
        )
        self._logger.info("step_simulation_done", results=len(results))
        return results

    def _decide(
        self,
        forecast_result: ForecastResult,
        uncertainty_report: "UncertaintyReport",
        baselines: dict,
        feature_df: pd.DataFrame,
    ) -> "OptimizationReport":
        self._logger.info("step_decision")
        engine = DecisionEngine(self._config)
        report = engine.analyze(
            forecast_result, uncertainty_report, baselines, feature_df,
        )
        self._logger.info(
            "step_decision_done",
            assessments=len(report.assessments),
            recommendations=len(report.recommendations),
        )
        return report

    @staticmethod
    def _default_scenarios(baselines: dict) -> List[SimulationScenario]:
        return [
            SimulationScenario(label="baseline", adjustments=[]),
            SimulationScenario(
                label="budget_increase_10pct",
                adjustments=[
                    BudgetAdjustment(entity_id=eid, channel=bl.channel, relative=0.1)
                    for eid, bl in baselines.items()
                ],
            ),
            SimulationScenario(
                label="budget_decrease_10pct",
                adjustments=[
                    BudgetAdjustment(entity_id=eid, channel=bl.channel, relative=-0.1)
                    for eid, bl in baselines.items()
                ],
            ),
        ]
