from pydantic import BaseModel, Field
from typing import List
from enum import Enum

class UrgencyLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class GrowthOutlook(str, Enum):
    WEAK = "WEAK"
    STABLE = "STABLE"
    STRONG = "STRONG"

class MacroCondition(str, Enum):
    EXPANSION = "EXPANSION"
    NEUTRAL = "NEUTRAL"
    RECESSION = "RECESSION"

# Global mapping exactly as prescribed
URGENCY_MAPPING = {
    "LOW": 0.0,
    "MEDIUM": 0.5,
    "HIGH": 1.0,
    "CRITICAL": 1.5,
    "WEAK": 0.0,
    "STABLE": 0.5,
    "STRONG": 1.0,
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

    key_risks: List[str] = Field(default_factory=list)
    key_opportunities: List[str] = Field(default_factory=list)
    recommended_focus: List[str] = Field(default_factory=list)
    
    # Confidence in the analysis (0.0 to 1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

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
        confidence=0.1 # Very low confidence on fallback
    )
