from typing import Dict, Any
from env.schemas import EnvState

class BaseAgent:
    """
    Abstract interface for all C-suite agents.
    """
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
        monthly_burn_est = state.headcount * 8000
        runway = state.cash / max(monthly_burn_est, 1)

        hires = 0
        if runway > 24:
            hires = 1
        
        if state.ltv / max(state.cac, 1) < 3:
            hires = 0 

        price_change = 0.0
        
        if state.ltv / max(state.cac, 1) < 3:
            price_change = 0.05

        return {
            "hiring": {"hires": hires, "cost_per_employee": 10000},
            "pricing": {"price_change_pct": price_change}
        }

class CMOAgent(BaseAgent):
    """
    CMO Agent: Focuses on growth (New MRR) and efficiency (CAC).
    """
    def act(self, state: EnvState) -> Dict[str, Any]:
        ratio = state.ltv / max(state.cac, 1)

        if ratio > 4:
            spend = 20000 
        elif ratio > 2:
            spend = 10000 
        else:
            spend = 2000  

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

        if avg_churn > 0.04:
            r_and_d = 15000 
        elif avg_churn > 0.02:
            r_and_d = 8000  
        else:
            r_and_d = 3000  

        if state.cash < 200000:
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
