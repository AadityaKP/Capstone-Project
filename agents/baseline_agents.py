from typing import Dict, Any
from env import business_logic
from env.schemas import EnvState

class BaseAgent:
    """
    Abstract interface for all C-suite agents.

    `scale` scales the absolute dollar amounts these agents propose (spec G11).
    They are tuned for a ~$50k-MRR company; a founder running at $12k MRR would
    otherwise be told to spend more than the company earns. Product surfaces
    pass mrr/50k; research runs leave it at 1.0 and are unaffected.

    physics_v2 (F3): `corridor="scale_aware"` re-expresses every dollar tier as
    a fraction of the company's CURRENT MRR, chosen so that at exactly the
    $50k-MRR calibration point the scale-aware tiers equal the legacy ones
    (20k/10k/2k marketing -> 40%/20%/4% of MRR, etc.). `scale` multiplied by
    INITIAL mrr/50k freezes ratios at the starting size for the whole episode;
    the corridor tracks the company as it grows. "legacy" (default) keeps
    recorded behaviour byte-identical.
    """
    def __init__(self, scale: float = 1.0, corridor: str = "legacy"):
        self.scale = max(0.01, float(scale))
        if corridor not in {"legacy", "scale_aware"}:
            raise ValueError(f"unknown corridor: {corridor!r}")
        self.corridor = corridor

    def act(self, state: EnvState) -> Dict[str, Any]:
        """
        Given the current environment state, return a partial action dictionary.
        """
        raise NotImplementedError

class CFOAgent(BaseAgent):
    """
    CFO Agent: Focuses on survival (runway), efficiency (Rule of 40), and pricing.
    """
    def act(self, state: EnvState) -> Dict[str, Any]:
        monthly_burn_est = business_logic.monthly_burn(state)
        runway = state.cash / max(monthly_burn_est, 1)

        hires = 0
        if runway > 24:
            hires = 1
        
        if state.ltv / max(state.cac, 1) < 3:
            hires = 0 

        price_change = 0.0

        if state.ltv / max(state.cac, 1) < 3:
            price_change = 0.05

        # Recruiting cost is per-head, not per-revenue: multiplying it by
        # mrr/50k charged a $30M-MRR company $6M per hire. The scale-aware
        # corridor uses the real dollar figure; legacy keeps the multiplier
        # because recorded runs depend on it.
        cost_per_employee = (10000.0 if self.corridor == "scale_aware"
                             else 10000 * self.scale)
        return {
            "hiring": {"hires": hires, "cost_per_employee": cost_per_employee},
            "pricing": {"price_change_pct": price_change}
        }

class CMOAgent(BaseAgent):
    """
    CMO Agent: Focuses on growth (New MRR) and efficiency (CAC).
    """
    def act(self, state: EnvState) -> Dict[str, Any]:
        ratio = state.ltv / max(state.cac, 1)

        if self.corridor == "scale_aware":
            # Legacy tiers at the $50k calibration point, as MRR fractions.
            # 40% of monthly revenue ~ the EDGAR panel's p50 S&M intensity
            # (43.8%); 20% sits between p10 and p25; 4% is the punitive tier
            # for LTV:CAC < 2.
            if ratio > 4:
                spend = state.mrr * 0.40
            elif ratio > 2:
                spend = state.mrr * 0.20
            else:
                spend = state.mrr * 0.04
        elif ratio > 4:
            spend = 20000 * self.scale
        elif ratio > 2:
            spend = 10000 * self.scale
        else:
            spend = 2000 * self.scale

        channel = "ppc" if state.consumer_confidence < 90 else "brand"

        return {
            "marketing": {"spend": spend, "channel": channel}
        }

class CPOAgent(BaseAgent):
    """
    CPO Agent: Focuses on product quality, retention (churn), and NRR.
    """
    def act(self, state: EnvState) -> Dict[str, Any]:
        avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0

        if self.corridor == "scale_aware":
            # Legacy tiers at the $50k calibration point, as MRR fractions.
            # 30% of monthly revenue ~ EDGAR R&D p75 (29.4%); 16% ~ p25 - p50;
            # 6% below p10 for low-churn cruising. Cash guard: legacy $200k at
            # $50k MRR = 4 months of revenue in the bank.
            if avg_churn > 0.04:
                r_and_d = state.mrr * 0.30
            elif avg_churn > 0.02:
                r_and_d = state.mrr * 0.16
            else:
                r_and_d = state.mrr * 0.06
            if state.cash < 4.0 * state.mrr:
                r_and_d *= 0.5
            return {"product": {"r_and_d_spend": r_and_d}}

        if avg_churn > 0.04:
            r_and_d = 15000 * self.scale
        elif avg_churn > 0.02:
            r_and_d = 8000 * self.scale
        else:
            r_and_d = 3000 * self.scale

        if state.cash < 200000 * self.scale:
            r_and_d *= 0.5

        return {
            "product": {"r_and_d_spend": r_and_d}
        }

def merge_actions(state: EnvState) -> Dict[str, Any]:
    """
    Runs all three agents and merges their actions into a single ActionBundle dict.
    """
    cfo = CFOAgent().act(state)
    cmo = CMOAgent().act(state)
    cpo = CPOAgent().act(state)

    action_bundle = {}
    action_bundle.update(cfo)
    action_bundle.update(cmo)
    action_bundle.update(cpo)

    return action_bundle
