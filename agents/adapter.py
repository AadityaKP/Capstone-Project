from typing import Dict, Any, Union
import logging

# Configure logging to capture invalid actions for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentAdapter")

class ActionAdapter:
    """
    The Safety Layer between Agents and the Sim.
    
    Responsibilities:
    1. Validate Input Structure (ActionBundle).
    2. Sanitize Sub-Action Params.
    3. Provide Default/No-Op values for missing components.
    4. Fail Gracefully.
    """
    
    @staticmethod
    def translate_action(agent_output: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Converts raw Agent output into a clean, Gym-compatible ActionBundle Dict.
        Expected Input Structure:
        {
            "marketing": {"spend": 1000, "channel": "ppc"},
            "hiring": {"hires": 1, ...},
            ...
        }
        """
        
        # 1. Fallback for total gibberish
        if not isinstance(agent_output, dict):
            logger.warning(f"Received non-dict action: {agent_output}. Returning Defaults.")
            return ActionAdapter._get_noop()
            
        clean_action = {}
        
        # 2. Marketing
        try:
            mkt = agent_output.get("marketing", {})
            clean_action["marketing"] = {
                "spend": max(0.0, float(mkt.get("spend", 0.0))),
                "channel": mkt.get("channel", "ppc") if mkt.get("channel") in ["ppc", "brand"] else "ppc"
            }
        except Exception:
             clean_action["marketing"] = {"spend": 0.0, "channel": "ppc"}

        # 3. Hiring
        try:
            hire = agent_output.get("hiring", {})
            clean_action["hiring"] = {
                "hires": max(0, int(hire.get("hires", 0))),
                "cost_per_employee": max(1.0, float(hire.get("cost_per_employee", 10000.0)))
            }
        except Exception:
            clean_action["hiring"] = {"hires": 0, "cost_per_employee": 10000.0}
            
        # 4. Product
        try:
            prod = agent_output.get("product", {})
            clean_action["product"] = {
                "r_and_d_spend": max(0.0, float(prod.get("r_and_d_spend", 0.0)))
            }
        except Exception:
            clean_action["product"] = {"r_and_d_spend": 0.0}
            
        # 5. Pricing
        try:
            price = agent_output.get("pricing", {})
            # Clamp percentage change to avoiding crashing the math (e.g. -200% price)
            pct = float(price.get("price_change_pct", 0.0))
            clean_action["pricing"] = {
                "price_change_pct": max(-0.5, min(1.0, pct)) # Safety clamps -50% to +100%
            }
        except Exception:
             clean_action["pricing"] = {"price_change_pct": 0.0}
            
        return clean_action

    @staticmethod
    def _get_noop() -> Dict[str, Any]:
        return {
            "marketing": {"spend": 0.0, "channel": "ppc"},
            "hiring": {"hires": 0, "cost_per_employee": 10000},
            "product": {"r_and_d_spend": 0.0},
            "pricing": {"price_change_pct": 0.0}
        }
