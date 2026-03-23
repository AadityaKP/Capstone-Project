from env.schemas import EnvState
from oracle.prompt_builder import build_prompt
from oracle.parser import parse_llm_response
from oracle.schemas import OracleBrief

class DummyLLMClient:
    """Fallback structural placeholder until proper LLM is wired."""
    def generate(self, prompt: str) -> str:
        print("[WARNING] DummyLLMClient used! `ollama` package might not be installed, yielding identical metrics.")
        return '{"risk_level":"MEDIUM","growth_outlook":"STABLE","efficiency_pressure":"MEDIUM","innovation_urgency":"MEDIUM","macro_condition":"NEUTRAL","key_risks":[],"key_opportunities":[],"recommended_focus":[],"confidence":0.5}'

try:
    from agents.llm_client import LLMClient
except ImportError:
    LLMClient = DummyLLMClient

class Oracle:
    def __init__(self):
        self.llm = LLMClient()
        
    def generate_brief(self, state: EnvState) -> OracleBrief:
        """
        Pure function: interprets state, calls LLM, and parses into OracleBrief.
        Does NOT store to memory here.
        """
        prompt = build_prompt(state)
        
        # Safely invoke LLM regardless of actual class method structure implementations
        if hasattr(self.llm, 'complete'):
            raw_output = self.llm.complete("You are a strategic SaaS oracle. Only output perfect JSON.", prompt)
        elif hasattr(self.llm, 'generate'):
            raw_output = self.llm.generate(prompt)
        elif hasattr(self.llm, 'call'):
            raw_output = self.llm.call(prompt)
        else:
            raw_output = DummyLLMClient().generate(prompt)
            
        if not raw_output:
            print("[WARNING] LLMClient returned completely empty output. Ollama might be offline. Using fallback.")
            
        brief = parse_llm_response(str(raw_output))
        return brief
