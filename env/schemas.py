from pydantic import BaseModel, Field
from typing import Dict, Any, Literal

# ==========================================
# Schema Definitions
# ==========================================
# We use Pydantic to strictly define the structure of data moving through the system.
# This guarantees type safety and provides auto-generated documentation for Agents.

class StartupState(BaseModel):
    """
    Represents the full snapshot of the Startup at a specific point in time (t).
    This is what the Agent 'sees' (Observation Space).
    """
    
    # Financials
    cash: float = Field(..., description="Current liquid cash balance. if <= 0, Game Over.")
    revenue: float = Field(..., ge=0.0, description="Revenue generated in the LAST step (Weekly).")
    burn_rate: float = Field(..., ge=0.0, description="Total expenses incurred in the LAST step (Salaries + Spend).")
    
    # User Metrics
    users: int = Field(..., ge=0, description="Total active users currently on the platform.")
    growth_rate: float = Field(..., description="Percentage growth in users vs previous step.")
    churn: float = Field(..., ge=0.0, le=1.0, description="Percentage of users lost this step (0.0 to 1.0).")
    
    # Market/Product Metrics
    cac: float = Field(..., ge=0.0, description="Effective Cost Per Acquisition for the last marketing push.")
    product_quality: float = Field(..., ge=0.0, le=1.0, description="Product score (0=Broken, 1=Perfect). Reduces Churn.")
    brand_strength: float = Field(..., ge=0.0, description="Accumulated brand equity. Multiplier for marketing efficiency.")
    price: float = Field(..., ge=0.0, description="Current subscription price per user.")
    
    # Meta
    time_step: int = Field(..., ge=0, description="Current simulation week (0 to MAX_STEPS).")

class Action(BaseModel):
    """
    Defines the structure of a valid decision an Agent can make.
    """
    # The Type governs which parameters are respected.
    type: Literal["marketing", "hiring", "product", "pricing", "skip"]
    
    # Flexible dictionary for parameters (e.g., {"amount": 5000})
    # We use a Dict here instead of strict fields to allow the Adapter to handle
    # partial/messy inputs from LLMs gracefully.
    params: Dict[str, Any] = Field(default_factory=dict)
