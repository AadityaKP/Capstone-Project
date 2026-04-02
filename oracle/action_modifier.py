from __future__ import annotations

from copy import deepcopy


class ActionModifier:
    """
    Translates Oracle brief fields into multiplicative action modifiers.
    Called BEFORE _apply_sanity_bounds in boardroom.decide().
    """

    RISK_MODIFIERS = {
        "CRITICAL": {"marketing_scale": 0.3, "rd_scale": 1.5, "hiring_cap": 0},
        "HIGH": {"marketing_scale": 0.5, "rd_scale": 1.3, "hiring_cap": 0},
        "MEDIUM": {"marketing_scale": 0.85, "rd_scale": 1.1, "hiring_cap": 1},
        "LOW": {"marketing_scale": 1.2, "rd_scale": 0.9, "hiring_cap": 2},
    }
    GROWTH_MODIFIERS = {
        "ACCELERATING": {"marketing_scale": 1.3},
        "STABLE": {"marketing_scale": 1.0},
        "DECLINING": {"marketing_scale": 0.6},
        "COLLAPSING": {"marketing_scale": 0.3},
    }
    EFFICIENCY_MODIFIERS = {
        "CRITICAL": {"spend_scale": 0.65},
        "HIGH": {"spend_scale": 0.8},
        "MEDIUM": {"spend_scale": 1.0},
        "LOW": {"spend_scale": 1.05},
    }
    INNOVATION_MODIFIERS = {
        "CRITICAL": {"rd_scale": 1.6},
        "HIGH": {"rd_scale": 1.4},
        "MEDIUM": {"rd_scale": 1.0},
        "LOW": {"rd_scale": 0.9},
    }

    def modify(self, action: dict, brief) -> dict:
        if brief is None:
            return action

        modified = deepcopy(action)

        risk = self._brief_value(brief, "risk_level", "MEDIUM")
        growth = self._brief_value(brief, "growth_outlook", "STABLE")
        efficiency = self._brief_value(brief, "efficiency_pressure", "MEDIUM")
        innovation = self._brief_value(brief, "innovation_urgency", "MEDIUM")

        rm = self.RISK_MODIFIERS.get(risk, self.RISK_MODIFIERS["MEDIUM"])
        gm = self.GROWTH_MODIFIERS.get(growth, self.GROWTH_MODIFIERS["STABLE"])
        em = self.EFFICIENCY_MODIFIERS.get(efficiency, self.EFFICIENCY_MODIFIERS["MEDIUM"])
        im = self.INNOVATION_MODIFIERS.get(innovation, self.INNOVATION_MODIFIERS["MEDIUM"])

        marketing_scale = rm["marketing_scale"] * gm["marketing_scale"] * em["spend_scale"]
        rd_scale = rm["rd_scale"] * em["spend_scale"] * im["rd_scale"]

        modified.setdefault("marketing", {})
        modified.setdefault("product", {})
        modified.setdefault("hiring", {})

        modified["marketing"]["spend"] = modified["marketing"].get("spend", 0.0) * marketing_scale
        modified["product"]["r_and_d_spend"] = modified["product"].get("r_and_d_spend", 0.0) * rd_scale
        modified["hiring"]["hires"] = min(modified["hiring"].get("hires", 0), rm["hiring_cap"])

        return modified

    @staticmethod
    def _brief_value(brief, field_name: str, default: str) -> str:
        value = getattr(brief, field_name, default)
        if hasattr(value, "value"):
            value = value.value
        return str(value).upper()


class NoOpActionModifier(ActionModifier):
    """Explicit no-op modifier for ablation policies."""

    def modify(self, action: dict, brief) -> dict:
        return deepcopy(action)
