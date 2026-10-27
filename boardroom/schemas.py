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
    # The falsifiable form of expected_impact (plan section 3.2a): signed
    # per-KPI deltas over a stated horizon, e.g.
    #   {"mrr_pct": 4.0, "churn_pp": -0.3, "cash_pct": -6.1,
    #    "runway_months": -0.4, "horizon_months": 2}
    # Produced by boardroom.expectation on the product path; None on every
    # research arm, which must keep reproducing byte-identically.
    expected_delta: Optional[dict] = None
    # One sentence saying how the agent's own track record changed this
    # proposal (plan section 3.2b), or None when it did not.
    adaptation: Optional[str] = None
    rationale: Optional[str] = None
    causal_confidence: Optional[float] = None
    risks: List[str]
    confidence: float

    # Vector Score (assigned by boardroom)
    score_vector: Optional[ScoreVector] = None

class NegotiationState(BaseModel):
    proposals: List[Proposal] = []
    round_number: int = 0
    consensus_reached: bool = False
    final_action: Optional[dict] = None
