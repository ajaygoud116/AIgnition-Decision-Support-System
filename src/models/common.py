from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Dict, List, Optional


class Horizon(int, Enum):
    D30 = 30
    D60 = 60
    D90 = 90


class Granularity(str, Enum):
    CAMPAIGN = "campaign"
    CHANNEL = "channel"
    OVERALL = "overall"


class MetricType(str, Enum):
    SPEND = "spend"
    REVENUE = "revenue"
    ROAS = "roas"
    CONVERSIONS = "conversions"
    CLICKS = "clicks"
    IMPRESSIONS = "impressions"
    DAILY_BUDGET = "daily_budget"


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def contains(self, d: date) -> bool:
        return self.start <= d <= self.end


@dataclass(frozen=True)
class QuantileValue:
    p10: float
    p50: float
    p90: float


@dataclass
class ForecastPoint:
    date: date
    values: QuantileValue


@dataclass
class ForecastSeries:
    entity_id: str
    channel: str
    granularity: Granularity
    metric: MetricType
    horizon: Horizon
    points: List[ForecastPoint] = field(default_factory=list)


@dataclass
class ForecastResult:
    series: List[ForecastSeries] = field(default_factory=list)
    metadata: Optional[dict] = None


@dataclass
class BudgetAdjustment:
    entity_id: str
    channel: str
    absolute: Optional[float] = None
    relative: Optional[float] = None

    def __post_init__(self) -> None:
        if self.absolute is None and self.relative is None:
            raise ValueError("Provide absolute or relative adjustment")
        if self.absolute is not None and self.relative is not None:
            raise ValueError("Provide absolute or relative adjustment, not both")


@dataclass
class SimulationScenario:
    label: str
    adjustments: List[BudgetAdjustment] = field(default_factory=list)
    base_forecast: Optional[ForecastResult] = None


@dataclass
class SimulationResult:
    scenario: SimulationScenario
    projected_revenue: float
    projected_spend: float
    projected_roas: float


@dataclass
class ROASMetric:
    value: float
    channel: Optional[str] = None
    campaign_id: Optional[str] = None


@dataclass
class AllocationRecommendation:
    entity_id: str
    channel: str
    current_budget: float
    recommended_budget: float
    rationale: str
    expected_roas: Optional[float] = None
