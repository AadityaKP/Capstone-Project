from __future__ import annotations

from pydantic import BaseModel, Field


class ScenarioConfig(BaseModel):
    company_name: str = "Demo SaaS"
    startup_stage: str = "Series A"
    business_model: str = "B2B SaaS"
    market_segment: str = "SMB"
    initial_headcount: int = Field(default=1, ge=1, le=10000)
    initial_mrr: float = Field(default=50000, gt=0)
    initial_cash: float = Field(default=1000000, gt=0)
    average_price: float = Field(default=50, gt=0)
    valuation_multiple: float = Field(default=10, gt=0)
    cac: float = Field(default=50, gt=0)
    ltv: float = Field(default=7000, gt=0)
    churn_smb: float = Field(default=0.03, ge=0, le=1)
    churn_enterprise: float = Field(default=0.01, ge=0, le=1)
    churn_b2c: float = Field(default=0.05, ge=0, le=1)
    interest_rate: float = Field(default=3, ge=0, le=100)
    consumer_confidence: float = Field(default=100, ge=0, le=200)
    competitors: int = Field(default=5, ge=0)
    unemployment: float = Field(default=4, ge=0, le=100)
    product_quality: float = Field(default=0.1, ge=0, le=1)
    innovation_factor: float = Field(default=1, ge=0, le=1)
    max_months: int = Field(default=120, ge=1, le=120)


class ScenarioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    config: ScenarioConfig = Field(default_factory=ScenarioConfig)


class SimulationCreate(BaseModel):
    scenario_id: int | None = None
    policy: str = "boardroom"
    episodes: int = Field(default=1, ge=1, le=25)
    seed_start: int = Field(default=0, ge=0)
    oracle_frequency: int = Field(default=5, ge=1, le=120)


# Founder product (spec G1/G2). Fields mirror the client's buildAdvisePayload;
# every one is optional-with-default because onboarding deliberately asks for a
# minimum set and enriches later.
class FounderConfig(BaseModel):
    company_name: str = "My company"
    initial_mrr: float = Field(default=0, ge=0)
    initial_cash: float = Field(default=0, ge=0)
    average_price: float = Field(default=50, gt=0)
    cac: float | None = Field(default=None, ge=0)
    ltv: float | None = Field(default=None, ge=0)
    churn_enterprise: float = Field(default=0.01, ge=0, le=1)
    churn_smb: float = Field(default=0.03, ge=0, le=1)
    churn_b2c: float = Field(default=0.05, ge=0, le=1)
    competitors: int = Field(default=5, ge=0)
    product_quality: float = Field(default=0.5, ge=0, le=1)
    initial_headcount: int = Field(default=1, ge=1, le=10000)
    monthly_burn_override: float | None = Field(default=None, ge=0)
    interest_rate: float | None = Field(default=None, ge=0, le=100)
    consumer_confidence: float | None = Field(default=None, ge=0, le=200)


class FounderHistoryEntry(BaseModel):
    mrr: float | None = None
    churn: float | None = None
    entered_at: str | None = None


class AdviseRequest(BaseModel):
    company_id: str
    company_age_months: int = Field(default=0, ge=0)
    month_index: int = Field(default=0, ge=0)
    config: FounderConfig
    history: list[FounderHistoryEntry] = Field(default_factory=list)
