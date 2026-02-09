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
        # --- Runway Calculation ---
        # Estimate monthly burn. 
        # Heuristic: headcount * 8000 (salary) + approximate other spend
        # We use a safe estimate for 'other spend' if we don't have access to last month's spend directly in state.
        # But for now, let's just use salary burn as the main component for the runway heuristic, 
        # or use a fixed buffer. The prompt suggests: monthly_burn = state.headcount * 8000
        monthly_burn_est = state.headcount * 8000
        runway = state.cash / max(monthly_burn_est, 1)

        # --- Hiring Logic ---
        # IF Rule-of-40 < 15 -> freeze hiring
        # ELSE IF runway > 24 months -> allow small hiring
        # ELSE -> no hiring
        
        # We need to approximate Rule of 40. 
        # Rule of 40 = Growth Rate + Profit Margin.
        # We don't have historical growth easily accessible here without memory, 
        # so this heuristic might need to be simplified or we assume the agent has external context.
        # However, the prompt gives a specific logic:
        # "IF Rule-of-40 < 15 -> freeze hiring"
        # Since we can't easily calc Rule of 40 effectively without history in this stateless agent,
        # we might need to rely on the prompt's *simplified* implementation example which 
        # implicitly looks at runway and efficiency (LTV:CAC) as proxies, OR ignores Rule of 40 for the MVP.
        # The prompt's "Minimal CFO implementation" uses:
        # if runway > 24: hires = 1
        # if state.ltv / max(state.cac, 1) < 3: hires = 0
        
        hires = 0
        if runway > 24:
            hires = 1
        
        # Efficiency Check: Freeze if LTV:CAC is poor
        if state.ltv / max(state.cac, 1) < 3:
            hires = 0 

        # --- Pricing Logic ---
        # IF CAC:LTV < 3 -> raise price slightly
        # IF churn rising -> avoid price increase (Not implemented in minimal version, but good to have)
        
        price_change = 0.0
        # The prompt says: "IF CAC:LTV < 3". 
        # Wait, usually it's LTV:CAC.
        # If LTV/CAC < 3, that means we are inefficient? Or efficient? 
        # Standard: LTV > 3*CAC is good.
        # If LTV/CAC < 3, we are inefficient. Raising price *might* help LTV.
        # Prompt code: "if state.ltv / max(state.cac, 1) < 3: price_change = 0.05"
        
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

        # --- Marketing Spend ---
        if ratio > 4:
            spend = 20000 # Aggressive growth
        elif ratio > 2:
            spend = 10000 # Moderate growth
        else:
            spend = 2000  # Pull back

        # --- Channel Selection ---
        # Prefer PPC (performance) when confidence is low (recession fears)
        # Prefer Brand when confidence is high
        channel = "ppc" if state.consumer_confidence < 90 else "brand"

        return {
            "marketing": {"spend": spend, "channel": channel}
        }

class CPOAgent(BaseAgent):
    """
    CPO Agent: Focuses on product quality, retention (churn), and NRR.
    """
    def act(self, state: EnvState) -> Dict[str, Any]:
        # Approximate average churn
        avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0

        # --- R&D Investment ---
        if avg_churn > 0.04:
            r_and_d = 15000 # Emergency fix
        elif avg_churn > 0.02:
            r_and_d = 8000  # Maintain/Improve
        else:
            r_and_d = 3000  # Maintenance mode

        # Constraint: Reduce R&D if cash is tight
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

    # Merge dictionaries. 
    # Since they return distinct keys (hiring/pricing, marketing, product), 
    # a simple update works fine.
    action_bundle = {}
    action_bundle.update(cfo)
    action_bundle.update(cmo)
    action_bundle.update(cpo)

    return action_bundle
