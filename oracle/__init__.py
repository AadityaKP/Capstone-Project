__all__ = ["Oracle", "OracleBrief", "URGENCY_MAPPING", "default_neutral_brief", "WeightAdapter"]


def __getattr__(name):
    if name == "Oracle":
        from .oracle import Oracle

        return Oracle
    if name == "WeightAdapter":
        from .weight_adapter import WeightAdapter

        return WeightAdapter
    if name in {"OracleBrief", "URGENCY_MAPPING", "default_neutral_brief"}:
        from .schemas import OracleBrief, URGENCY_MAPPING, default_neutral_brief

        mapping = {
            "OracleBrief": OracleBrief,
            "URGENCY_MAPPING": URGENCY_MAPPING,
            "default_neutral_brief": default_neutral_brief,
        }
        return mapping[name]
    raise AttributeError(f"module 'oracle' has no attribute {name!r}")
