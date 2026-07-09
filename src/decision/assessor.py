from typing import Dict, List, Optional

import pandas as pd

from src.decision.models import CampaignAssessment
from src.models.common import ForecastResult
from src.simulation.baselines import CampaignBaseline
from src.uncertainty.models import UncertaintyReport
from src.utils.config import Config
from src.utils.logger import StructuredLogger


class CampaignAssessor:
    """Evaluates each campaign against decision thresholds."""

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("decision.assessor")
        self._min_roas = config.get("decision.min_roas_target", 3.0)
        self._volatility_threshold = config.get("decision.volatility_threshold", 0.5)
        self._zero_revenue_days = config.get("decision.zero_revenue_threshold_days", 45)
        self._cost_inflation_threshold = config.get("decision.cost_inflation_threshold", 0.2)
        self._concentration_threshold = config.get("decision.concentration_threshold", 0.6)

    def assess_all(
        self,
        baselines: Dict[str, CampaignBaseline],
        uncertainty_report: UncertaintyReport,
        forecast_result: ForecastResult,
        feature_df: Optional[pd.DataFrame] = None,
    ) -> List[CampaignAssessment]:
        self._logger.info("assess_start", campaigns=len(baselines))
        uncertainty_map = {e.entity_id: e for e in uncertainty_report.entities}
        total_spend = sum(bl.total_spend for bl in baselines.values()) or 1.0

        assessments: List[CampaignAssessment] = []
        for eid, bl in baselines.items():
            unc = uncertainty_map.get(eid)
            flags: List[str] = []

            if bl.historical_roas < self._min_roas:
                flags.append("below_roas_target")

            if unc is not None and unc.confidence_score < (1.0 - self._volatility_threshold):
                flags.append("high_uncertainty")

            if feature_df is not None and self._has_zero_revenue(feature_df, eid):
                flags.append("zero_revenue")

            if feature_df is not None and self._has_cost_inflation(feature_df, eid):
                flags.append("cost_inflation")

            if bl.total_spend / total_spend > self._concentration_threshold:
                flags.append("concentration_risk")

            score = self._compute_score(bl, unc)

            assessments.append(
                CampaignAssessment(
                    entity_id=eid,
                    channel=bl.channel,
                    campaign_type=bl.campaign_type,
                    current_spend=bl.total_spend,
                    current_roas=bl.historical_roas,
                    confidence_score=unc.confidence_score if unc else 0.5,
                    volatility=unc.volatility if unc else 0.5,
                    stability_trend=unc.stability_trend if unc else "stable",
                    flags=flags,
                    score=score,
                )
            )

        self._logger.info(
            "assess_complete",
            assessed=len(assessments),
            flagged=sum(1 for a in assessments if a.flags),
        )
        return assessments

    def _has_zero_revenue(self, feature_df: pd.DataFrame, entity_id: str) -> bool:
        group = feature_df[feature_df["campaign_id"] == entity_id].sort_values("date")
        if group.empty:
            return False
        zero_streak = 0
        for rev in group["revenue"]:
            if rev == 0.0:
                zero_streak += 1
                if zero_streak >= self._zero_revenue_days:
                    return True
            else:
                zero_streak = 0
        return False

    def _has_cost_inflation(self, feature_df: pd.DataFrame, entity_id: str) -> bool:
        group = feature_df[feature_df["campaign_id"] == entity_id].sort_values("date")
        if len(group) < 30:
            return False
        mid = len(group) // 2
        early = group.iloc[:mid]
        late = group.iloc[mid:]

        early_spend = early["spend"].sum()
        early_rev = early["revenue"].sum()
        late_spend = late["spend"].sum()
        late_rev = late["revenue"].sum()

        early_ratio = early_spend / early_rev if early_rev > 0 else float("inf")
        late_ratio = late_spend / late_rev if late_rev > 0 else float("inf")

        if early_ratio == 0.0:
            return False
        if early_ratio == float("inf") and late_ratio == float("inf"):
            return False

        if early_ratio == float("inf"):
            return late_ratio > 0

        inflation = (late_ratio - early_ratio) / early_ratio
        return inflation > self._cost_inflation_threshold

    def _compute_score(
        self,
        bl: CampaignBaseline,
        unc: Optional["EntityUncertainty"],
    ) -> float:
        roas_factor = min(bl.historical_roas / self._min_roas, 3.0) if self._min_roas > 0 else 1.0
        confidence = unc.confidence_score if unc else 0.5
        volatility_penalty = 1.0 / (1.0 + (unc.volatility if unc else 0.5))
        return roas_factor * confidence * volatility_penalty

    def get_zero_revenue_days(self) -> int:
        return self._zero_revenue_days
