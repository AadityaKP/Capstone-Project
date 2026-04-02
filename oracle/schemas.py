from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from enum import Enum

class UrgencyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class GrowthOutlook(str, Enum):
    ACCELERATING = "ACCELERATING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"
    COLLAPSING = "COLLAPSING"

class MacroCondition(str, Enum):
    EXPANSION = "EXPANSION"
    NEUTRAL = "NEUTRAL"
    RECESSION = "RECESSION"

class ExpectedOutcome(str, Enum):
    GROWTH = "GROWTH"
    STAGNATION = "STAGNATION"
    DECLINE = "DECLINE"

class TrendDirection(str, Enum):
    INCREASING = "INCREASING"
    FLAT = "FLAT"
    DECREASING = "DECREASING"

# Global mapping exactly as prescribed
URGENCY_MAPPING = {
    "LOW": 0.0,
    "MEDIUM": 0.5,
    "HIGH": 1.0,
    "CRITICAL": 1.5,
    "ACCELERATING": 1.0,
    "STABLE": 0.5,
    "DECLINING": 0.25,
    "COLLAPSING": 0.0,
    "EXPANSION": 1.0,
    "NEUTRAL": 0.5,
    "RECESSION": 0.0,
}

class OracleBrief(BaseModel):
    risk_level: UrgencyLevel
    growth_outlook: GrowthOutlook
    efficiency_pressure: UrgencyLevel
    innovation_urgency: UrgencyLevel
    macro_condition: MacroCondition
    expected_outcome: Optional[ExpectedOutcome] = None

    key_risks: List[str] = Field(default_factory=list)
    key_opportunities: List[str] = Field(default_factory=list)
    recommended_focus: List[str] = Field(default_factory=list)
    
    # Confidence in the analysis (0.0 to 1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

class StateSnapshot(BaseModel):
    global_month: int = Field(..., ge=0)
    source_month: int = Field(..., ge=0)
    episode_seed: Optional[int] = None
    mrr: float
    avg_churn: float = Field(..., ge=0.0)
    innovation: float = Field(..., ge=0.0, le=1.0)

class TrendContext(BaseModel):
    mrr_trend: TrendDirection = TrendDirection.FLAT
    innovation_trend: TrendDirection = TrendDirection.FLAT
    churn_trend: TrendDirection = TrendDirection.FLAT
    history_points: int = Field(default=1, ge=1)
    previous_mrr: Optional[float] = None
    current_mrr: Optional[float] = None
    mrr_delta_pct: Optional[float] = None
    previous_avg_churn: Optional[float] = None
    current_avg_churn: Optional[float] = None
    churn_delta: Optional[float] = None

class PendingMemoryEntry(BaseModel):
    snapshot: StateSnapshot
    trend_context: TrendContext

class RetrievedMemoryCandidate(BaseModel):
    document: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    distance: float = 1.0
    similarity_score: float = 0.5
    recency_factor: float = 1.0
    memory_weight: float = 0.5


class OracleRefreshSnapshot(BaseModel):
    months_elapsed: int = Field(..., ge=0)
    mrr: float
    avg_churn: float = Field(..., ge=0.0)
    consumer_confidence: float
    unemployment: float = Field(..., ge=0.0)
    runway_months: float = Field(..., ge=0.0)


class OracleEpisodeStats(BaseModel):
    oracle_refresh_requests: int = 0
    cadence_refreshes: int = 0
    event_refreshes: int = 0
    cache_hits: int = 0
    llm_calls: int = 0

def default_neutral_brief() -> OracleBrief:
    """Mandatory neutral fallback for JSON execution failures."""
    return OracleBrief(
        risk_level=UrgencyLevel.MEDIUM,
        growth_outlook=GrowthOutlook.STABLE,
        efficiency_pressure=UrgencyLevel.MEDIUM,
        innovation_urgency=UrgencyLevel.MEDIUM,
        macro_condition=MacroCondition.NEUTRAL,
        key_risks=["LLM parsed failed, assuming neutral risk."],
        key_opportunities=[],
        recommended_focus=["Maintain baseline operations."],
        confidence=0.1, # Very low confidence on fallback
        expected_outcome=ExpectedOutcome.STAGNATION,
    )
