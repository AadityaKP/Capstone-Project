import json
from oracle.schemas import OracleBrief, default_neutral_brief

def parse_llm_response(raw_text: str) -> OracleBrief:
    """Parses LLM JSON out. Fallbacks to neutral brief on failure."""
    try:
        # Strip markdown if present
        text = raw_text.strip()
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        elif text.startswith("```"):
            text = text[3:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
            
        data = json.loads(text.strip())
        return OracleBrief(**data)
    except Exception as e:
        print(f"Oracle parsing failed: {e}. Falling back to default.")
        return default_neutral_brief()
