from typing import Dict, List

from src.decision.assessor import CampaignAssessment
from src.models.common import AllocationRecommendation
from src.simulation.baselines import CampaignBaseline
from src.utils.config import Config
from src.utils.logger import StructuredLogger


class BudgetOptimizer:
    """Computes optimal budget allocation across campaigns.

    Uses a score-based proportional allocation with smoothing constraints
    to prevent drastic budget swings.
    """

    def __init__(self, config: Config):
        self._config = config
        self._logger = StructuredLogger("decision.optimizer")
        self._min_roas = config.get("decision.min_roas_target", 3.0)
        self._diminishing_periods = config.get("decision.diminishing_returns_periods", 3)
        self._concentration_threshold = config.get("decision.concentration_threshold", 0.6)
        self._max_change_ratio = 0.5

    def optimize(
        self,
        assessments: List[CampaignAssessment],
        baselines: Dict[str, CampaignBaseline],
    ) -> List[AllocationRecommendation]:
        self._logger.info("optimize_start", assessments=len(assessments))
        if not assessments:
            return []

        total_budget = sum(bl.total_spend for bl in baselines.values())
        total_score = sum(max(a.score, 0.0) for a in assessments)

        # Step 1: Raw proportional allocations
        raw_alloc: Dict[str, float] = {}
        for a in assessments:
            if a.entity_id not in baselines:
                continue
            if total_score > 0:
                share = max(a.score, 0.0) / total_score
            else:
                share = 1.0 / len(assessments)
            raw_alloc[a.entity_id] = total_budget * share

        # Step 2: Iterative clamping + re-distribution to conserve total budget
        final_alloc = dict(raw_alloc)
        flexible = set(raw_alloc.keys())

        for _ in range(10):
            newly_fixed: set = set()
            for eid in flexible:
                bl = baselines[eid]
                mc = bl.total_spend * self._max_change_ratio
                lower = max(0.0, bl.total_spend - mc)
                upper = bl.total_spend + mc
                if final_alloc[eid] < lower:
                    final_alloc[eid] = lower
                    newly_fixed.add(eid)
                elif final_alloc[eid] > upper:
                    final_alloc[eid] = upper
                    newly_fixed.add(eid)

            if not newly_fixed:
                break

            flexible -= newly_fixed
            if not flexible:
                break

            fixed_spend = sum(final_alloc[eid] for eid in set(final_alloc) - flexible)
            remaining = total_budget - fixed_spend
            flex_score = sum(
                max(a.score, 0.0) for a in assessments
                if a.entity_id in flexible
            )

            if flex_score > 0 and remaining > 0:
                for eid in flexible:
                    a_next = next(a for a in assessments if a.entity_id == eid)
                    share = max(a_next.score, 0.0) / flex_score
                    final_alloc[eid] = remaining * share

        # Step 3: Build recommendations
        recommendations: List[AllocationRecommendation] = []
        for a in assessments:
            bl = baselines.get(a.entity_id)
            if bl is None:
                continue
            recommended = max(0.0, final_alloc.get(a.entity_id, 0.0))
            rationale = self._build_rationale(a, recommended, bl.total_spend)
            recommendations.append(
                AllocationRecommendation(
                    entity_id=a.entity_id,
                    channel=a.channel,
                    current_budget=bl.total_spend,
                    recommended_budget=recommended,
                    rationale=rationale,
                    expected_roas=a.current_roas,
                )
            )

        total_recommended = sum(r.recommended_budget for r in recommendations)
        if total_recommended > 0 and abs(total_recommended - total_budget) / total_budget > 0.001:
            self._logger.warning(
                "budget_allocation_conflict",
                total_budget=round(total_budget, 2),
                total_allocated=round(total_recommended, 2),
                gap_pct=round(abs(total_recommended - total_budget) / total_budget * 100, 2),
                detail="Budget constraints prevent full allocation; deficit accepted.",
            )

        self._logger.info(
            "optimize_complete",
            recommendations=len(recommendations),
            total_recommended=sum(r.recommended_budget for r in recommendations),
        )
        return recommendations

    def _build_rationale(
        self,
        assessment: CampaignAssessment,
        recommended: float,
        current: float,
    ) -> str:
        parts: List[str] = []
        diff = recommended - current
        if abs(diff) < 1.0:
            parts.append("Maintain current budget.")
        elif diff > 0:
            pct = diff / current * 100 if current > 0 else 0.0
            parts.append(f"Increase budget {pct:.0f}% (${diff:.0f}).")
        else:
            pct = -diff / current * 100 if current > 0 else 0.0
            parts.append(f"Decrease budget {pct:.0f}% (${-diff:.0f}).")

        if "below_roas_target" in assessment.flags:
            parts.append("Campaign ROAS is below target.")
        if "high_uncertainty" in assessment.flags:
            parts.append("Forecast confidence is low.")
        if "zero_revenue" in assessment.flags:
            parts.append("Campaign has prolonged zero revenue.")
        if "cost_inflation" in assessment.flags:
            parts.append("Costs are outpacing revenue growth.")
        if "concentration_risk" in assessment.flags:
            parts.append("Budget concentration is high.")

        if not assessment.flags:
            parts.append("Campaign performing well.")

        return " ".join(parts)
