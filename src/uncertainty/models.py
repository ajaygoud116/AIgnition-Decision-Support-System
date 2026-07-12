from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EntityUncertainty:
    entity_id: str
    channel: str
    avg_interval_width: float
    avg_relative_width: float
    confidence_score: float
    volatility: float
    stability_trend: str  # "narrowing", "stable", "widening"
    calibrated_coverage: float = 0.0
    calibration_alpha: float = 1.0
    horizon_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class ChannelUncertainty:
    channel: str
    avg_confidence: float
    avg_volatility: float
    campaign_count: int
    high_uncertainty_campaigns: List[str] = field(default_factory=list)


@dataclass
class UncertaintyReport:
    entities: List[EntityUncertainty] = field(default_factory=list)
    channels: List[ChannelUncertainty] = field(default_factory=list)
    overall_confidence: float = 0.0
    overall_volatility: float = 0.0
    high_uncertainty_count: int = 0
    metadata: Optional[dict] = None
