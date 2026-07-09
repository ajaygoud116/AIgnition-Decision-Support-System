from dataclasses import dataclass, field
from typing import Dict, List

from src.models.common import AllocationRecommendation


@dataclass
class CampaignAssessment:
    entity_id: str
    channel: str
    campaign_type: str
    current_spend: float
    current_roas: float
    confidence_score: float
    volatility: float
    stability_trend: str
    flags: List[str] = field(default_factory=list)
    score: float = 0.0


@dataclass
class OptimizationReport:
    assessments: List[CampaignAssessment] = field(default_factory=list)
    recommendations: List[AllocationRecommendation] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
