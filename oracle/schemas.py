from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Enums


class TrendDirection(str, Enum):
    INCREASING = "INCREASING"
    FLAT = "FLAT"
    DECREASING = "DECREASING"


class RiskLevel(str, Enum):
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


URGENCY_MAPPING: Dict[str, float] = {
    "LOW": 0.1,
    "MEDIUM": 0.5,
    "HIGH": 0.8,
    "CRITICAL": 1.0,
    "ACCELERATING": 0.9,
    "STABLE": 0.5,
    "DECLINING": 0.2,
    "COLLAPSING": 0.0,
    "EXPANSION": 0.9,
    "NEUTRAL": 0.5,
    "RECESSION": 0.1,
}


# Oracle brief


class OracleBrief(BaseModel):
    risk_level: RiskLevel = RiskLevel.MEDIUM
    growth_outlook: GrowthOutlook = GrowthOutlook.STABLE
    efficiency_pressure: RiskLevel = RiskLevel.MEDIUM
    innovation_urgency: RiskLevel = RiskLevel.MEDIUM
    macro_condition: MacroCondition = MacroCondition.NEUTRAL
    expected_outcome: Optional[ExpectedOutcome] = None
    key_risks: List[str] = Field(default_factory=list)
    key_opportunities: List[str] = Field(default_factory=list)
    recommended_focus: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def default_neutral_brief() -> OracleBrief:
    return OracleBrief(
        risk_level=RiskLevel.MEDIUM,
        growth_outlook=GrowthOutlook.STABLE,
        efficiency_pressure=RiskLevel.MEDIUM,
        innovation_urgency=RiskLevel.MEDIUM,
        macro_condition=MacroCondition.NEUTRAL,
        confidence=0.5,
    )


# State + memory schemas


class StateSnapshot(BaseModel):
    global_month: int
    source_month: int
    episode_seed: Optional[int] = None
    mrr: float = 0.0
    avg_churn: float = 0.0
    innovation: float = 1.0


class TrendContext(BaseModel):
    mrr_trend: TrendDirection = TrendDirection.FLAT
    innovation_trend: TrendDirection = TrendDirection.FLAT
    churn_trend: TrendDirection = TrendDirection.FLAT
    history_points: int = 0
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
    similarity_score: float = 0.0
    recency_factor: float = 1.0
    memory_weight: float = 0.0


# Oracle episode tracking


class OracleEpisodeStats(BaseModel):
    oracle_refresh_requests: int = 0
    cadence_refreshes: int = 0
    event_refreshes: int = 0
    cache_hits: int = 0
    llm_calls: int = 0


class OracleRefreshSnapshot(BaseModel):
    months_elapsed: int = 0
    mrr: float = 0.0
    avg_churn: float = 0.0
    consumer_confidence: float = 100.0
    unemployment: float = 4.0
    runway_months: float = 0.0


# Graph context schemas


class GraphShockRecord(BaseModel):
    """One historical shock event with its decision context and outcome."""

    episode_id: int
    shock_type: str
    shock_month: int
    mrr_tier: str
    brief_risk_level: str
    marketing_spend: float
    rd_spend: float
    hires: int
    recovery_months: Optional[int] = None
    recovered: bool = False
    post_shock_rule40: float = 0.0
    mrr_change_pct: float = 0.0


class CausalChainSummary(BaseModel):
    """Aggregated statistics for a shock type used in prompt context."""

    shock_type: str
    total_occurrences: int
    mean_recovery_months: float
    recovery_rate: float
    mean_post_shock_rule40: float
    best_risk_level: Optional[str] = None
    worst_risk_level: Optional[str] = None


class GraphContext(BaseModel):
    """Container passed from Oracle.get_context() into build_prompt()."""

    similar_shocks: List[GraphShockRecord] = Field(default_factory=list)
    causal_summary: Optional[CausalChainSummary] = None
    active_shock_type: Optional[str] = None
