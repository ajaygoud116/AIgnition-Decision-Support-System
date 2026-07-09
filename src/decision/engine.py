from typing import Dict, Optional

import pandas as pd

from src.decision.assessor import CampaignAssessor
from src.decision.models import OptimizationReport
from src.decision.optimizer import BudgetOptimizer
from src.models.common import ForecastResult
from src.simulation.baselines import CampaignBaseline
from src.uncertainty.models import UncertaintyReport
from src.utils.config import Config
from src.utils.logger import StructuredLogger


class DecisionEngine:
    """End-to-end decision engine.

    Assesses campaigns against thresholds, then optimizes budget
    allocation and generates recommendations.
    """

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("decision.engine")
        self._assessor = CampaignAssessor(config)
        self._optimizer = BudgetOptimizer(config)

    def analyze(
        self,
        forecast_result: ForecastResult,
        uncertainty_report: UncertaintyReport,
        baselines: Dict[str, CampaignBaseline],
        feature_df: Optional[pd.DataFrame] = None,
    ) -> OptimizationReport:
        self._logger.info("analyze_start")

        assessments = self._assessor.assess_all(
            baselines, uncertainty_report, forecast_result, feature_df
        )
        recommendations = self._optimizer.optimize(assessments, baselines)

        total_current = sum(bl.total_spend for bl in baselines.values())
        total_recommended = sum(r.recommended_budget for r in recommendations)

        summary = {
            "total_current_budget": total_current,
            "total_recommended_budget": total_recommended,
            "budget_change": total_recommended - total_current,
            "campaigns_assessed": len(assessments),
            "campaigns_flagged": sum(1 for a in assessments if a.flags),
            "total_flags": sum(len(a.flags) for a in assessments),
            "recommendations_generated": len(recommendations),
        }

        self._logger.info("analyze_complete", summary=summary)
        return OptimizationReport(
            assessments=assessments,
            recommendations=recommendations,
            summary=summary,
        )
