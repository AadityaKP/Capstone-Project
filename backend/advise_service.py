"""Advise-now: one Boardroom analysis of a founder-supplied state (spec G1).

Unlike the simulation runner, this never steps the environment. It builds a
single EnvState from the founder's own numbers, replays their history through
Oracle.observe_state so trend context is real, then calls Boardroom.decide()
once and stores the resulting brief + decision trace.

Memory isolation: founder analyses read and write a *copy* of the research
corpus (chroma_db_founder) so live usage cannot contaminate the thesis
memories. CHROMA_PATH is set before the Oracle is constructed, because
OracleMemoryStore resolves its path at construction time.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOUNDER_CHROMA_PATH = os.path.join(ROOT_DIR, "chroma_db_founder")

# Must precede any Oracle/memory import-time construction.
os.environ.setdefault("CHROMA_PATH", FOUNDER_CHROMA_PATH)

from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
from boardroom.boardroom import Boardroom
from env import business_logic
from env.schemas import EnvState

from backend.database import connect, utc_now

# The founder product runs oracle_v4: v4 reasoning without the causal graph
# store, which only oracle_v4_causal instantiates. Keeping it off means founder
# analyses write nothing to the shared Neo4j graph.
ORACLE_MODE = os.getenv("FOUNDER_ORACLE_MODE", "oracle_v4")

# One analysis per request, so the Oracle must refresh on every call rather
# than on its usual multi-month cadence.
ORACLE_FREQUENCY = 1

# Engine defaults for macro conditions the founder is never asked for. Surfaced
# in the UI as "estimated by the system", never as measurement.
DEFAULT_INTEREST_RATE = 3.0
DEFAULT_CONSUMER_CONFIDENCE = 100.0

# The boardroom's absolute spend floors are calibrated at this MRR (spec G11).
CALIBRATION_MRR = 50_000.0


def absolute_scale(mrr: float) -> float:
    """Scale factor for the boardroom's absolute floors (G11).

    Clamped to <= 1.0 so companies at or above the calibration point keep the
    validated behaviour untouched; only smaller companies scale down. Floored
    at 0.05 so a pre-revenue company still gets a non-zero plan.
    """
    if mrr <= 0:
        return 0.05
    return min(1.0, max(0.05, mrr / CALIBRATION_MRR))


def _f(value: Any, fallback: float) -> float:
    try:
        if value is None:
            return float(fallback)
        return float(value)
    except (TypeError, ValueError):
        return float(fallback)


def build_env_state(payload: dict[str, Any]) -> EnvState:
    """Founder inputs -> EnvState (spec section 5 mappings).

    The founder's real burn reaches the board through initial_headcount, which
    the client derives as costs / $8k salary slots: Boardroom estimates runway
    as headcount * 8000, so virtual headcount makes that estimate match the
    founder's actual costs without any engine change.
    """
    config = payload.get("config") or {}

    price = _f(config.get("average_price"), 50.0)
    churn_smb = _f(config.get("churn_smb"), 0.03)
    ltv = _f(config.get("ltv"), business_logic.compute_ltv(price, churn_smb))

    return EnvState(
        mrr=_f(config.get("initial_mrr"), 0.0),
        cash=_f(config.get("initial_cash"), 0.0),
        cac=_f(config.get("cac"), 50.0),
        ltv=ltv,
        churn_enterprise=min(max(_f(config.get("churn_enterprise"), 0.01), 0.0), 1.0),
        churn_smb=min(max(churn_smb, 0.0), 1.0),
        churn_b2c=min(max(_f(config.get("churn_b2c"), 0.05), 0.0), 1.0),
        interest_rate=_f(config.get("interest_rate"), DEFAULT_INTEREST_RATE),
        consumer_confidence=_f(config.get("consumer_confidence"), DEFAULT_CONSUMER_CONFIDENCE),
        competitors=int(_f(config.get("competitors"), 5)),
        product_quality=min(max(_f(config.get("product_quality"), 0.5), 0.0), 1.0),
        price=price,
        months_elapsed=int(_f(payload.get("company_age_months"), 0)),
        headcount=max(1, int(_f(config.get("initial_headcount"), 1))),
    )


def _replay_history(boardroom: Boardroom, state: EnvState, history: list[dict]) -> int:
    """Feed prior months through Oracle.observe_state so trend context is the
    founder's own trajectory rather than a single point. Returns months seen."""
    if not boardroom.use_oracle or not history:
        return 0

    seen = 0
    for entry in history:
        past = state.model_copy(deep=True)
        past.mrr = _f(entry.get("mrr"), state.mrr)
        churn = entry.get("churn")
        if churn is not None:
            past.churn_smb = min(max(_f(churn, state.churn_smb), 0.0), 1.0)
        past.months_elapsed = max(0, state.months_elapsed - (len(history) - seen))
        boardroom.oracle.observe_state(past)
        seen += 1
    return seen


def run_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """One analysis. Returns brief, decision trace and an honest llm_ok flag."""
    state = build_env_state(payload)

    scale = absolute_scale(state.mrr)
    boardroom = Boardroom(
        [
            CFOProposalAgent(scale=scale),
            CMOProposalAgent(scale=scale),
            CPOProposalAgent(scale=scale),
        ],
        use_oracle=True,
        oracle_mode=ORACLE_MODE,
        oracle_frequency=ORACLE_FREQUENCY,
        scale_absolutes=scale,
    )
    boardroom.start_episode(episode_seed=None)

    months_replayed = _replay_history(boardroom, state, payload.get("history") or [])

    action = boardroom.decide(state)
    trace = boardroom.get_last_decision_trace() or {}
    brief = trace.get("brief") or {}

    # G3: a fallback brief carries no signal. Say so rather than letting the UI
    # present safe defaults as though the board had actually read the numbers.
    llm_ok = bool(brief.get("parse_ok", False))

    trace = dict(trace)
    trace["history_months_replayed"] = months_replayed
    trace["absolute_scale"] = scale
    trace["final_action"] = trace.get("final_action") or action

    return {
        "brief": brief,
        "trace": trace,
        "llm_ok": llm_ok,
        "oracle_mode": ORACLE_MODE,
        "created_at": utc_now(),
    }


def store_analysis(company_id: str, month_index: int, result: dict[str, Any]) -> str:
    """Persist one analysis (G2). Returns the analysis id."""
    import json

    analysis_id = str(uuid.uuid4())
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO analyses (
                id, company_id, month_index, status, brief_json, trace_json,
                llm_ok, oracle_mode, created_at, completed_at
            ) VALUES (?, ?, ?, 'complete', ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis_id,
                company_id,
                month_index,
                json.dumps(result.get("brief")),
                json.dumps(result.get("trace")),
                1 if result.get("llm_ok") else 0,
                result.get("oracle_mode"),
                result.get("created_at"),
                utc_now(),
            ),
        )
    return analysis_id
