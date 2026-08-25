import json
from typing import Any

from boardroom.schemas import Proposal
from env.schemas import EnvState
from oracle.schemas import CausalGraphContext
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent


class BatchedCausalProposalGenerator:
    """Generate CFO, CMO, and CPO proposals in one causal-context LLM call."""

    ROLES = ("CFO", "CMO", "CPO")

    def __init__(self, llm_client, scale: float = 1.0):
        self.llm_client = llm_client
        # scale carries the G11 calibration factor into the fallback path, so a
        # dropped LLM call degrades to correctly-sized advice rather than to
        # constants tuned for a ~$50k-MRR company.
        self.scale = scale
        self.fallback_agents = {
            "CFO": CFOProposalAgent(scale=scale),
            "CMO": CMOProposalAgent(scale=scale),
            "CPO": CPOProposalAgent(scale=scale),
        }
        self.llm_calls = 0
        self.last_source = "none"
        self.last_error: str | None = None

    def propose_all(
        self,
        state: EnvState,
        causal_contexts: dict[str, CausalGraphContext],
        stress_persistence_months: int | None = None,
        recent_action_pattern: dict[str, Any] | None = None,
    ) -> list[Proposal]:
        if not self._has_required_context(causal_contexts):
            return self._fallback(state, "fallback_no_context")

        system_prompt = (
            "You are a startup boardroom simulator. Output only valid JSON. "
            "Do not include markdown, comments, or prose outside the JSON object."
        )
        user_prompt = self._build_prompt(
            state,
            causal_contexts,
            stress_persistence_months=stress_persistence_months,
            recent_action_pattern=recent_action_pattern,
        )

        try:
            self.llm_calls += 1
            raw = self.llm_client.complete(system_prompt, user_prompt)
        except Exception as exc:
            return self._fallback(state, "fallback_llm_error", str(exc))

        if not raw:
            return self._fallback(state, "fallback_llm_empty")

        try:
            parsed = self._parse_json_object(str(raw))
            proposals = self._proposals_from_payload(parsed, state, causal_contexts, int(stress_persistence_months or 0))
        except Exception as exc:
            return self._fallback(state, "fallback_parse_error", str(exc))

        self.last_source = "llm"
        self.last_error = None
        return proposals

    def _fallback(
        self,
        state: EnvState,
        source: str,
        error: str | None = None,
    ) -> list[Proposal]:
        self.last_source = source
        self.last_error = error
        proposals = []
        for role in self.ROLES:
            proposal = self.fallback_agents[role].propose(state)
            proposals.append(proposal.model_copy(update={"causal_confidence": None}))
        return proposals

    def _has_required_context(
        self,
        causal_contexts: dict[str, CausalGraphContext],
    ) -> bool:
        for role in self.ROLES:
            context = causal_contexts.get(role)
            if context is None or not context.raw_triples:
                return False
        return True

    def _build_prompt(
        self,
        state: EnvState,
        causal_contexts: dict[str, CausalGraphContext],
        stress_persistence_months: int | None = None,
        recent_action_pattern: dict[str, Any] | None = None,
    ) -> str:
        avg_churn = (
            state.churn_enterprise + state.churn_smb + state.churn_b2c
        ) / 3.0
        runway = state.cash / max(state.headcount * 8000.0, 1.0)
        ltv_cac = state.ltv / max(state.cac, 1.0)
        stress_node = next(
            (
                context.stress_node
                for context in causal_contexts.values()
                if getattr(context, "stress_node", None)
            ),
            "Unknown",
        )
        persistence_months = max(0, int(stress_persistence_months or 0))
        recent_action_summary = self._summarize_action_pattern(recent_action_pattern)
        context_lines = []
        for role in self.ROLES:
            context = causal_contexts[role]
            context_lines.append(
                f"{role} context for {context.stress_node}: "
                f"{context.chain_summary} "
                f"(graph_confidence={context.confidence:.2f})"
            )
        bounds = self._action_bounds(state, stress_node, persistence_months)

        return f"""
Current KPIs:
- MRR: {state.mrr:.0f}
- Cash: {state.cash:.0f}
- Runway months: {runway:.1f}
- Avg churn: {avg_churn:.4f}
- LTV/CAC: {ltv_cac:.2f}
- Innovation factor: {state.innovation_factor:.2f}

Causal graph context:
{chr(10).join(context_lines)}

Stress persistence signal:
- Current stress node: {stress_node}
- Consecutive months in this stress node: {persistence_months}
- Previous final action pattern: {recent_action_summary}
- If the same stress persists for more than 10 months despite similar actions, switch or escalate levers rather than repeating the same spend, hiring, product, or pricing response.

Generate one proposal for each role. Keep action keys exactly as specified.
Use these numeric action bounds exactly:
- CFO hiring.hires: 0 to {bounds["hires_max"]}
- CFO hiring.cost_per_employee: {bounds["cost_min"]:.0f} to {bounds["cost_max"]:.0f}
- CFO pricing.price_change_pct: {bounds["price_min"]:.2f} to {bounds["price_max"]:.2f}
- CMO marketing.spend: 0 to {bounds["marketing_max"]:.0f}; if positive, use at least {bounds["marketing_positive_floor"]:.0f}
- CPO product.r_and_d_spend: 0 to {bounds["rd_max"]:.0f}

CFO pricing decision rule:
- Price increases are valid and often correct when MRR growth, LTV/CAC, or demand signals are healthy; do not default to price cuts.
- If LTV/CAC > 3 and average churn is low or stable, choose a positive price_change_pct such as 0.01 to 0.03 unless the causal context specifically points to price sensitivity, churn from pricing, or demand weakness.
- Cut price only when the causal chain indicates price sensitivity or weak demand.
- Example when LTV/CAC is healthy and churn is low: {{"pricing": {{"price_change_pct": 0.02}}}} because pricing power can extend runway without extra burn.

Set each proposal's causal_confidence from that role's graph_confidence. Do not use the same causal_confidence for all roles unless the graph confidences are identical.

Respond with this exact JSON shape:
{{
  "CFO": {{
    "objective": "...",
    "actions": {{
      "hiring": {{"hires": 0, "cost_per_employee": 10000}},
      "pricing": {{"price_change_pct": 0.0}}
    }},
    "expected_impact": "...",
    "risks": ["..."],
    "rationale": "1-2 sentences citing the causal chain",
    "confidence": 0.0,
    "causal_confidence": 0.0
  }},
  "CMO": {{
    "objective": "...",
    "actions": {{
      "marketing": {{"spend": 10000, "channel": "ppc"}}
    }},
    "expected_impact": "...",
    "risks": ["..."],
    "rationale": "1-2 sentences citing the causal chain",
    "confidence": 0.0,
    "causal_confidence": 0.0
  }},
  "CPO": {{
    "objective": "...",
    "actions": {{
      "product": {{"r_and_d_spend": 10000}}
    }},
    "expected_impact": "...",
    "risks": ["..."],
    "rationale": "1-2 sentences citing the causal chain",
    "confidence": 0.0,
    "causal_confidence": 0.0
  }}
}}
""".strip()

    def _parse_json_object(self, raw: str) -> dict[str, Any]:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in LLM response")
        parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("LLM response was not a JSON object")
        return parsed

    def _proposals_from_payload(
        self,
        parsed: dict[str, Any],
        state: EnvState,
        causal_contexts: dict[str, CausalGraphContext],
        stress_persistence_months: int = 0,
    ) -> list[Proposal]:
        role_payloads = parsed.get("proposals", parsed)
        if not isinstance(role_payloads, dict):
            raise ValueError("Missing proposal object")

        proposals = []
        for role in self.ROLES:
            payload = role_payloads.get(role)
            if not isinstance(payload, dict):
                raise ValueError(f"Missing {role} proposal")

            actions = payload.get("actions")
            if not isinstance(actions, dict):
                raise ValueError(f"Missing {role} actions")

            context = causal_contexts[role]
            proposals.append(
                Proposal(
                    agent=role,
                    objective=str(payload.get("objective") or self._default_objective(role)),
                    actions=self._normalize_actions(role, actions, state, context, stress_persistence_months),
                    expected_impact=str(
                        payload.get("expected_impact") or self._default_impact(role)
                    ),
                    risks=self._normalize_risks(payload.get("risks")),
                    rationale=self._optional_str(payload.get("rationale")),
                    confidence=self._clamp_float(payload.get("confidence"), 0.75),
                    causal_confidence=self._ground_causal_confidence(
                        payload.get("causal_confidence"),
                        context.confidence,
                    ),
                )
            )
        return proposals

    def _normalize_actions(
        self,
        role: str,
        actions: dict[str, Any],
        state: EnvState,
        context: CausalGraphContext | None = None,
        stress_persistence_months: int = 0,
    ) -> dict[str, Any]:
        if role == "CFO":
            hiring = actions.get("hiring")
            pricing = actions.get("pricing")
            if not isinstance(hiring, dict) or not isinstance(pricing, dict):
                raise ValueError("CFO actions must include hiring and pricing")
            stress_node = getattr(context, "stress_node", None)
            bounds = self._action_bounds(state, stress_node, stress_persistence_months)
            price_change_pct = self._clamp_number(
                pricing.get("price_change_pct", 0.0),
                bounds["price_min"],
                bounds["price_max"],
                0.0,
            )
            price_change_pct = self._nudge_healthy_pricing_power(
                price_change_pct,
                state,
                context,
                bounds,
            )
            if (
                stress_node == "Churn_Spike"
                and stress_persistence_months >= 10
                and price_change_pct <= 0.0
            ):
                tier = 0.04 if stress_persistence_months >= 30 else 0.02
                price_change_pct = min(bounds["price_max"], tier)
            return {
                "hiring": {
                    "hires": int(
                        self._clamp_number(
                            hiring.get("hires", 0),
                            0,
                            bounds["hires_max"],
                            0,
                        )
                    ),
                    "cost_per_employee": self._clamp_number(
                        hiring.get("cost_per_employee", 10_000),
                        bounds["cost_min"],
                        bounds["cost_max"],
                        10_000,
                    ),
                },
                "pricing": {
                    "price_change_pct": price_change_pct
                },
            }

        if role == "CMO":
            marketing = actions.get("marketing")
            if not isinstance(marketing, dict):
                raise ValueError("CMO actions must include marketing")
            bounds = self._action_bounds(state, getattr(context, "stress_node", None), stress_persistence_months)
            channel = marketing.get("channel", "ppc")
            if channel not in {"ppc", "brand"}:
                channel = "ppc"
            spend = self._clamp_number(
                marketing.get("spend", state.mrr * 0.05),
                0.0,
                bounds["marketing_max"],
                min(state.mrr * 0.05, bounds["marketing_max"]),
            )
            if 0.0 < spend < bounds["marketing_positive_floor"]:
                spend = min(bounds["marketing_positive_floor"], bounds["marketing_max"])
            return {
                "marketing": {
                    "spend": spend,
                    "channel": channel,
                }
            }

        product = actions.get("product")
        if not isinstance(product, dict):
            raise ValueError("CPO actions must include product")
        bounds = self._action_bounds(state, getattr(context, "stress_node", None), stress_persistence_months)
        return {
            "product": {
                "r_and_d_spend": self._clamp_number(
                    product.get("r_and_d_spend", state.mrr * 0.05),
                    0.0,
                    bounds["rd_max"],
                    min(state.mrr * 0.05, bounds["rd_max"]),
                )
            }
        }

    @staticmethod
    def _summarize_action_pattern(action: dict[str, Any] | None) -> str:
        if not action:
            return "none"

        marketing = action.get("marketing", {}) if isinstance(action, dict) else {}
        hiring = action.get("hiring", {}) if isinstance(action, dict) else {}
        product = action.get("product", {}) if isinstance(action, dict) else {}
        pricing = action.get("pricing", {}) if isinstance(action, dict) else {}

        return (
            f"marketing_spend={float(marketing.get('spend', 0.0) or 0.0):.0f}, "
            f"hires={int(hiring.get('hires', 0) or 0)}, "
            f"rd_spend={float(product.get('r_and_d_spend', 0.0) or 0.0):.0f}, "
            f"price_change_pct={float(pricing.get('price_change_pct', 0.0) or 0.0):.3f}"
        )

    def _nudge_healthy_pricing_power(
        self,
        price_change_pct: float,
        state: EnvState,
        context: CausalGraphContext | None,
        bounds: dict[str, float | int],
    ) -> float:
        if abs(price_change_pct) > 1e-9:
            return price_change_pct

        avg_churn = (
            state.churn_enterprise + state.churn_smb + state.churn_b2c
        ) / 3.0
        ltv_cac = state.ltv / max(state.cac, 1.0)
        if ltv_cac <= 3.0 or avg_churn > 0.05:
            return price_change_pct

        context_text = ""
        if context is not None:
            context_text = " ".join(
                [
                    context.chain_summary or "",
                    " ".join(" ".join(triple) for triple in context.raw_triples),
                ]
            ).lower()
        price_sensitivity_terms = (
            "price sensitivity",
            "price_sensitive",
            "pricing sensitivity",
            "churn from pricing",
            "price-driven churn",
            "demand weakness",
        )
        if any(term in context_text for term in price_sensitivity_terms):
            return price_change_pct

        return min(float(bounds["price_max"]), max(0.01, float(bounds["price_min"])))

    @staticmethod
    def _action_bounds(
        state: EnvState,
        stress_node: str | None = None,
        stress_persistence_months: int = 0,
    ) -> dict[str, float | int]:
        positive_cash = max(0.0, float(state.cash))
        marketing_max = min(float(state.mrr) * 0.20, positive_cash * 0.10, 50_000.0)
        marketing_positive_floor = 2_000.0 if positive_cash > 0 and marketing_max >= 2_000.0 else 0.0
        rd_max = min(float(state.mrr) * 0.15, max(positive_cash * 0.25, 30_000.0))
        sp = int(stress_persistence_months or 0)
        price_max = 0.05
        if stress_node == "Churn_Spike" and sp >= 30:
            price_max = 0.12
        elif stress_node == "Churn_Spike" and sp >= 10:
            price_max = 0.08
        return {
            "hires_max": 2,
            "cost_min": 8_000.0,
            "cost_max": 12_000.0,
            "price_min": -0.05,
            "price_max": price_max,
            "marketing_max": max(0.0, marketing_max),
            "marketing_positive_floor": marketing_positive_floor,
            "rd_max": max(0.0, rd_max),
        }

    @staticmethod
    def _clamp_number(value: Any, minimum: float, maximum: float, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(minimum, min(maximum, numeric))

    @staticmethod
    def _normalize_risks(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str) and value:
            return [value]
        return []

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _clamp_float(value: Any, default: float) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(0.0, min(1.0, numeric))

    def _ground_causal_confidence(self, value: Any, graph_confidence: float) -> float:
        llm_confidence = self._clamp_float(value, graph_confidence)
        graph_confidence = self._clamp_float(graph_confidence, 0.5)
        grounded = (graph_confidence * 0.75) + (llm_confidence * 0.25)
        return max(0.0, min(1.0, grounded))

    @staticmethod
    def _default_objective(role: str) -> str:
        return {
            "CFO": "Preserve runway and capital efficiency",
            "CMO": "Improve efficient growth",
            "CPO": "Reduce churn and improve retention",
        }[role]

    @staticmethod
    def _default_impact(role: str) -> str:
        return {
            "CFO": "Improved survival probability",
            "CMO": "Higher qualified growth",
            "CPO": "Improved product retention",
        }[role]
