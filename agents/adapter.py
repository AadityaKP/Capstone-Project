from typing import Dict, Any, Union
import logging

# Configure logging to capture invalid actions for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AgentAdapter")

class ActionAdapter:
    """
    The Safety Layer between Agents and the Sim.
    
    Why this exists:
    - LLM Agents (CrewAI, LangGraph) are probabilistic. They make mistakes.
    - They might output `{"price": -10}` or `{"marketing": "lots"}`.
    - If we plug that directly into `env.step()`, the math breaks.
    
    Responsibilities:
    1. Validate TYPES: Is 'fire_everyone' a valid action? (No).
    2. Sanitize PARAMS: Is price negative? (Clamp it). Is amount a string? (Cast it).
    3. Fail Gracefully: If action is garbage, return 'skip' rather than crashing.
    """
    
    VALID_ACTIONS = ["marketing", "hiring", "product", "pricing", "skip"]
    
    @staticmethod
    def translate_action(agent_output: Union[Dict[str, Any], str]) -> Dict[str, Any]:
        """
        Converts raw Agent output into a clean, Gym-compatible Action Dict.
        """
        
        # 1. Fallback for total gibberish (e.g., Agent returns a String instead of JSON)
        if not isinstance(agent_output, dict):
            logger.warning(f"Received non-dict action: {agent_output}. Skipping.")
            return {"type": "skip", "params": {}}
            
        # 2. Extract and Normalize Type
        raw_type = agent_output.get("type", "").lower().strip()
        params = agent_output.get("params", {})
        
        # 3. Type Validation
        if raw_type not in ActionAdapter.VALID_ACTIONS:
            if raw_type:
               logger.warning(f"Unknown action type: '{raw_type}'. Skipping.")
            return {"type": "skip", "params": {}}
            
        # 4. Parameter Sanitization (Per Action Type)
        clean_params = {}
        
        try:
            if raw_type == "marketing":
                # Ensure 'amount' is a positive float
                amount = float(params.get("amount", 0.0))
                clean_params["amount"] = max(0.0, amount) 
                
            elif raw_type == "hiring":
                # Ensure 'count' is a positive integer
                count = int(params.get("count", 0))
                clean_params["count"] = max(0, count) 
                
            elif raw_type == "product":
                # Ensure 'amount' is a positive float
                amount = float(params.get("amount", 0.0))
                clean_params["amount"] = max(0.0, amount)
                
            elif raw_type == "pricing":
                # Ensure 'price' is a positive float (Min $0.01)
                price = float(params.get("price", 10.0))
                clean_params["price"] = max(0.01, price) 
                
        except (ValueError, TypeError) as e:
            # If casting fails (e.g. float("free")), log it and skip.
            logger.error(f"Parameter validation failed for {raw_type}: {e}")
            return {"type": "skip", "params": {}}
            
        return {"type": raw_type, "params": clean_params}
