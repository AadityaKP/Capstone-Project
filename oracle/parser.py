import json

from oracle.schemas import OracleBrief, default_neutral_brief


_ENUM_DEFAULTS = {
    "risk_level": "MEDIUM",
    "growth_outlook": "STABLE",
    "efficiency_pressure": "MEDIUM",
    "innovation_urgency": "MEDIUM",
    "macro_condition": "NEUTRAL",
    "expected_outcome": None,
}

_ENUM_ALLOWED = {
    "risk_level": {"LOW", "MEDIUM", "HIGH", "CRITICAL"},
    "growth_outlook": {"ACCELERATING", "STABLE", "DECLINING", "COLLAPSING"},
    "efficiency_pressure": {"LOW", "MEDIUM", "HIGH", "CRITICAL"},
    "innovation_urgency": {"LOW", "MEDIUM", "HIGH", "CRITICAL"},
    "macro_condition": {"EXPANSION", "NEUTRAL", "RECESSION"},
    "expected_outcome": {"GROWTH", "STAGNATION", "DECLINE"},
}


def _strip_markdown(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```json"):
        text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    elif text.startswith("```"):
        text = text[3:].strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    return text.strip()


def _extract_json_object(text: str):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text[index:])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("No JSON object found", text, 0)


def _coerce_list_of_strings(value) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]

    coerced: list[str] = []
    for item in value:
        if item is None:
            continue
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            parts = []
            for key, raw in item.items():
                if raw is None:
                    continue
                if isinstance(raw, (str, int, float, bool)):
                    parts.append(f"{key}: {raw}")
            text = "; ".join(parts)
        else:
            text = str(item).strip()

        if text:
            coerced.append(text)
    return coerced


def _normalize_enum(field_name: str, value):
    default = _ENUM_DEFAULTS[field_name]
    if value is None:
        return default
    text = str(value).strip().upper()
    return text if text in _ENUM_ALLOWED[field_name] else default


def _normalize_confidence(value) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _normalize_payload(data: dict) -> dict:
    normalized = dict(data)

    for field_name in _ENUM_DEFAULTS:
        normalized[field_name] = _normalize_enum(field_name, normalized.get(field_name))

    for field_name in ("key_risks", "key_opportunities", "recommended_focus"):
        normalized[field_name] = _coerce_list_of_strings(normalized.get(field_name))

    normalized["confidence"] = _normalize_confidence(normalized.get("confidence", 0.5))
    return normalized


def parse_llm_response(raw_text: str) -> OracleBrief:
    """Parses LLM JSON out. Fallbacks to neutral brief on failure."""
    try:
        text = _strip_markdown(raw_text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = _extract_json_object(text)

        if not isinstance(data, dict):
            raise ValueError("Parsed LLM output was not a JSON object.")

        return OracleBrief(**_normalize_payload(data))
    except Exception as e:
        print(f"Oracle parsing failed: {e}. Falling back to default.")
        return default_neutral_brief()
