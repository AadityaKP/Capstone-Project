"""Run the founder product with no Ollama and no Neo4j.

This exists so the product can be demonstrated on a laptop that is not running
the model server or the graph database. It is a record-and-replay layer, not a
mock: `record_demo_fixtures.py` captures what the real stack actually returned
for a given set of founder inputs, and this module hands that recording back
when the same inputs arrive again.

Two tiers, in order:

  1. **Replay.** The inputs match a recording, so the founder sees exactly the
     analysis the full stack produced - the strategist's own brief, the causal
     evidence, the board's plan. `llm_ok` is whatever it was at record time.

  2. **Offline board.** No recording matches. Rather than failing in front of an
     audience, the board runs for real without the Oracle: CFOProposalAgent and
     friends subclass the heuristic agents and only reach for a model to
     decorate their rationale, so `use_oracle=False` with no proposal generator
     is a complete, deterministic analysis that touches no external service. It
     reports `llm_ok: false`, which the UI already renders as an honest banner
     saying the strategist could not be reached.

Neither tier invents a number. Tier 1 is a real past answer; tier 2 is a real
present one computed without the model.

`/api/whatif` needs nothing from this module - the projection is pure CPU and
already runs offline.
"""

from __future__ import annotations

import json
import os
from typing import Any

from backend.database import utc_now

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_fixtures")

# Environment switch. On this branch it defaults ON, because the branch exists
# for offline demonstration; set FOUNDER_DEMO_FIXTURES=0 to force the live path.
def enabled() -> bool:
    return os.getenv("FOUNDER_DEMO_FIXTURES", "1") not in ("0", "false", "False")


# The founder-visible inputs that identify a recording. Deliberately not the
# whole payload: company_id and month_index are bookkeeping, and matching on
# them would mean a recording only replayed for the exact company row that
# produced it.
SIGNATURE_FIELDS = (
    "initial_mrr", "initial_cash", "monthly_costs", "average_price",
    "cac", "churn_smb", "competitors", "product_quality", "initial_headcount",
)


def signature(payload: dict[str, Any]) -> str:
    """A stable key for one set of founder inputs.

    Every number is normalised to the same fixed-point form. Two reasons, and
    the first one bit during development: the recorder hands save() the raw
    dict it built, where initial_cash is the int 95000, while at runtime the
    same field arrives through pydantic as the float 95000.0 - formatted
    differently, a recording could never match the request that produced it.
    The second is that the client derives `cac` by dividing marketing spend by
    new customers, and a lookup should not miss on a float ending in ...9999.
    """
    config = payload.get("config") or {}
    parts = [f"age={int(payload.get('company_age_months') or 0)}"]
    for field in SIGNATURE_FIELDS:
        value = config.get(field)
        if value is None:
            parts.append(f"{field}=none")
        else:
            parts.append(f"{field}={float(value):.4f}")
    return "|".join(parts)


def _fixture_path(name: str) -> str:
    return os.path.join(FIXTURE_DIR, f"{name}.json")


def _load_all() -> list[dict[str, Any]]:
    if not os.path.isdir(FIXTURE_DIR):
        return []
    out = []
    for filename in sorted(os.listdir(FIXTURE_DIR)):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(FIXTURE_DIR, filename), encoding="utf-8") as handle:
            out.append(json.load(handle))
    return out


def lookup(payload: dict[str, Any]) -> dict[str, Any] | None:
    """The recorded analysis for these inputs, or None.

    `created_at` is refreshed to now. The recording is a real past answer, but
    presenting it with its original timestamp would make every screen that
    reasons about freshness - "numbers from Aug 1", the stale-analysis banner -
    say something false about when the founder asked.
    """
    if not enabled():
        return None
    from backend import sim_profile

    want = signature(payload)
    for fixture in _load_all():
        # A recording is only a truthful replay under the profile that produced
        # it: a founder-profile server must not hand back review2 answers or
        # vice versa. Legacy recordings predate the stamp and came from the
        # founder stack.
        if fixture.get("sim_profile", "founder") != sim_profile.get_profile():
            continue
        if fixture.get("signature") == want:
            analysis = dict(fixture["analysis"])
            analysis["created_at"] = utc_now()
            analysis["source"] = "recorded"
            analysis["recorded_at"] = fixture.get("recorded_at")
            return analysis
    return None


def save(name: str, payload: dict[str, Any], analysis: dict[str, Any]) -> str:
    """Write one recording. Used by record_demo_fixtures.py, never at runtime."""
    from backend import sim_profile

    os.makedirs(FIXTURE_DIR, exist_ok=True)
    path = _fixture_path(name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "name": name,
                "signature": signature(payload),
                "sim_profile": sim_profile.get_profile(),
                "recorded_at": utc_now(),
                "inputs": payload.get("config"),
                "company_age_months": payload.get("company_age_months"),
                "analysis": analysis,
            },
            handle,
            indent=1,
        )
    return path


def _rule_based_brief(state, payload: dict[str, Any]) -> dict[str, Any]:
    """A brief computed from the founder's numbers, with no model involved.

    Without this the offline board returns no brief at all, and RiskChip falls
    back to MEDIUM - so the screen would have claimed "Moderate risk" about a
    company nothing had assessed. Every field here is derived arithmetic, and
    `parse_ok` stays False so the UI keeps saying the strategist was not
    reached.
    """
    import calibration as cal
    from backend import founder_view
    from env import business_logic

    runway = founder_view.runway_months(
        state.cash, business_logic.monthly_burn(state), state.mrr
    )
    if state.cash <= 0:
        risk = "CRITICAL"
    elif runway is None:
        risk = "LOW"                     # revenue covers costs
    elif runway < 3:
        risk = "CRITICAL"
    elif runway < 6:
        risk = "HIGH"
    elif runway < 12:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # Growth needs two points. With one, say steady rather than guess.
    history = payload.get("history") or []
    outlook = "STABLE"
    previous = next((h.get("mrr") for h in reversed(history) if h.get("mrr")), None)
    if previous:
        change = (state.mrr - previous) / previous
        if change >= 0.08:
            outlook = "ACCELERATING"
        elif change <= -0.15:
            outlook = "COLLAPSING"
        elif change < 0:
            outlook = "DECLINING"

    avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0
    risks: list[str] = []
    benchmark = cal.monthly_churn(state.price, kind="gross")
    if benchmark.is_observed and avg_churn > benchmark.value:
        risks.append(
            f"Churn is {avg_churn * 100:.1f}% a month against a published median of "
            f"{benchmark.value * 100:.1f}% at your price point"
        )
    if runway is not None and runway < 12:
        risks.append(f"About {runway:.0f} months of cash left at current costs")
    if state.cac > 0 and state.ltv / state.cac < 3:
        risks.append("A customer costs more to win than they pay back over three times")

    opportunities: list[str] = []
    if runway is None:
        opportunities.append("Revenue currently covers your costs")
    elif runway >= 18:
        opportunities.append(f"About {runway:.0f} months of cash gives room to invest")
    if state.cac > 0 and state.ltv / state.cac >= 3:
        opportunities.append("Customers pay back comfortably more than they cost to win")

    return {
        "risk_level": risk,
        "growth_outlook": outlook,
        "key_risks": risks,
        "key_opportunities": opportunities,
        # No model spoke, so there is no model confidence to report. None makes
        # founder_view.confidence start at Moderate and the assumption count
        # cap it from there.
        "confidence": None,
        "parse_ok": False,
    }


def offline_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """A real analysis with no Oracle, no model and no graph database.

    The board is genuinely run: the proposal agents are the heuristic C-suite,
    the risk modifier and the cash-safety resolver both apply, and the founder's
    own numbers drive all of it. What is missing is the strategist's read, so
    `llm_ok` is False and the UI says so rather than passing built-in rules off
    as a model's judgement.
    """
    # Imported here rather than at module scope: advise_service pulls in the
    # memory store, and the point of this path is to touch as little as possible.
    from backend.advise_service import assumed_fields, build_env_state
    from backend import founder_view, sim_profile
    from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
    from boardroom.boardroom import Boardroom
    from env import business_logic

    state = build_env_state(payload)
    # Same profile resolution as the live path: review2 runs the unscaled
    # research board with no founder guards, founder keeps the scaled floors.
    scale = sim_profile.get_agent_scale(state.mrr)

    board = Boardroom(
        [CFOProposalAgent(scale=scale), CMOProposalAgent(scale=scale),
         CPOProposalAgent(scale=scale)],
        use_oracle=False,
        **sim_profile.get_boardroom_kwargs(state.mrr),
    )
    board.start_episode(episode_seed=None)
    action = board.decide(state)

    trace = dict(board.get_last_decision_trace() or {})
    trace.setdefault("final_action", action)
    trace["assumed_fields"] = assumed_fields(payload)
    trace["absolute_scale"] = scale

    brief = trace.get("brief") or _rule_based_brief(state, payload)
    trace["brief"] = brief
    final_action = trace["final_action"]
    outflow = (
        business_logic.monthly_burn(state)
        + float((final_action.get("marketing") or {}).get("spend", 0.0) or 0.0)
        + float((final_action.get("product") or {}).get("r_and_d_spend", 0.0) or 0.0)
    )
    display = {
        "confidence": founder_view.confidence(
            brief.get("confidence"), len(trace["assumed_fields"])
        ),
        "runway": founder_view.runway_phrase(
            state.cash, business_logic.monthly_burn(state), state.mrr
        ),
        "spend_ratio": founder_view.spend_ratio_phrase(outflow, state.mrr),
        "show_rule_of_40": founder_view.rule_of_40_is_meaningful(state.mrr),
        "monthly_burn": business_logic.monthly_burn(state),
        "monthly_burn_supplied": state.monthly_burn is not None,
    }

    return {
        "brief": brief,
        "trace": trace,
        "display": display,
        "llm_ok": False,
        "oracle_mode": "offline_rules",
        "source": "offline",
        "created_at": utc_now(),
    }
