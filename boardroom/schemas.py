from pydantic import BaseModel, Field
from typing import List, Optional

class ScoreVector(BaseModel):
    efficiency: float = Field(default=0.0, ge=0.0, le=1.0)
    growth: float = Field(default=0.0, ge=0.0, le=1.0)
    innovation: float = Field(default=0.0, ge=0.0, le=1.0)
    macro: float = Field(default=0.0, ge=0.0, le=1.0)

class Proposal(BaseModel):
    agent: str

    # Strategy
    objective: str
    actions: dict  # partial action dict

    # Evaluation
    expected_impact: str
    risks: List[str]
    confidence: float

    # Vector Score (assigned by boardroom)
    score_vector: Optional[ScoreVector] = None

class NegotiationState(BaseModel):
    proposals: List[Proposal] = []
    round_number: int = 0
    consensus_reached: bool = False
    final_action: Optional[dict] = None
