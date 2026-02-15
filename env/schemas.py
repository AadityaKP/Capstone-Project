from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Literal, Optional

# ==========================================
# Schema Definitions
# ==========================================
# We use Pydantic to strictly define the structure of data moving through the system.
# This guarantees type safety and provides auto-generated documentation for Agents.

class EnvState(BaseModel):
    """
    Represents the full snapshot of the Startup at a specific point in time (t).
    This is what the Agent 'sees' (Observation Space).
    Includes Core Financials, Unit Economics, Segmented Churn, Macro Variables, and Lifecycle.
    """
    
    # Core Financials
    mrr: float = Field(..., description="Monthly Recurring Revenue ($).")
    cash: float = Field(..., description="Current liquid cash balance ($). If <= 0, Game Over.")
    
    # Unit Economics
    cac: float = Field(..., description="Customer Acquisition Cost ($).")
    ltv: float = Field(..., description="Lifetime Value ($).")
    
    # Churn by Segment (Monthly Rates 0.0-1.0)
    churn_enterprise: float = Field(..., ge=0.0, le=1.0)
    churn_smb: float = Field(..., ge=0.0, le=1.0)
    churn_b2c: float = Field(..., ge=0.0, le=1.0)
    
    # Macro Variables
    interest_rate: float = Field(..., description="Current Interest Rate (%). Affects capital costs.")
    consumer_confidence: float = Field(..., description="Consumer Confidence Index (0-200). Affects demand.")
    competitors: int = Field(..., ge=0, description="Number of direct competitors.")
    
    # Product / Quality Proxy
    product_quality: float = Field(..., ge=0.0, le=1.0, description="Product Quality Score (0=Broken, 1=Perfect).")
    
    # Pricing
    price: float = Field(..., ge=0.0, description="Average revenue per user (ARPU) / Price.")

    # Lifecycle
    months_elapsed: int = Field(default=0, ge=0, description="Time step / simulation month.")
    
    # Internal Ops
    headcount: int = Field(default=1, ge=1, description="Number of full-time employees.")
    
    # Macro Feedback State (Shock Engine)
    valuation_multiple: float = Field(default=10.0, description="Current revenue valuation multiple (e.g. 10x ARR).")
    unemployment: float = Field(default=4.0, ge=0.0, le=100.0, description="National unemployment rate (%).")
    innovation_factor: float = Field(default=1.0, ge=0.0, le=1.0, description="R&D efficiency multiplier (1.0 = normal).")
    months_in_depression: int = Field(default=0, ge=0, description="Consecutive months with low consumer confidence (<50).")


# =============================
# ACTION DEFINITIONS
# =============================

class MarketingAction(BaseModel):
    spend: float = Field(..., ge=0.0, description="Marketing spend amount ($).")
    channel: Literal["ppc", "brand"] = Field(..., description="Marketing channel strategy.")

class HiringAction(BaseModel):
    hires: int = Field(..., ge=0, description="Number of new employees to hire.")
    cost_per_employee: float = Field(..., ge=0.0, description="Cost per new hire (recruiting + salary setup).")

class ProductAction(BaseModel):
    r_and_d_spend: float = Field(..., ge=0.0, description=" investment in Product R&D ($).")

class PricingAction(BaseModel):
    price_change_pct: float = Field(..., description="Percentage change in price (e.g., 0.1 = +10%).")

class ActionBundle(BaseModel):
    """
    A bundle of actions to be executed simultaneously in a single time step.
    """
    marketing: MarketingAction
    hiring: HiringAction
    product: ProductAction
    pricing: PricingAction
