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
FOUNDER_CHROMA_PATH = os.environ.get(
    "FOUNDER_CHROMA_PATH", os.path.join(ROOT_DIR, "chroma_db_founder")
)

from agents.causal_proposal_agents import BatchedCausalProposalGenerator
from agents.llm_client import create_llm_client
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
from boardroom.boardroom import Boardroom
from env import business_logic
from env.schemas import EnvState
from oracle.memory import OracleMemoryStore
from oracle.oracle import Oracle

from backend import founder_view
from backend.database import connect, utc_now
import calibration as cal

# The founder product runs oracle_v4_causal with batched causal proposals.
#
# Mode choice matters here. Plain oracle_v4_causal reads the graph only through
# build_graph_context(), which returns empty without an active shock - and a
# founder's monthly review has no shock - so the graph would contribute nothing.
# The batched causal generator instead queries role-specific evidence keyed on a
# stress node derived from the founder's own numbers (Cash_Shortage,
# Churn_Spike, CAC_Pressure...), which needs no shock and grounds every
# analysis.
#
# This path is read-only against Neo4j: graph writes require an active shock
# label or Oracle.end_episode(), and one Boardroom.decide() triggers neither.
# Set FOUNDER_ORACLE_MODE=oracle_v4 to run without Neo4j entirely.
ORACLE_MODE = os.getenv("FOUNDER_ORACLE_MODE", "oracle_v4_causal")
USE_CAUSAL_PROPOSALS = ORACLE_MODE == "oracle_v4_causal"

# One analysis per request, so the Oracle must refresh on every call rather
# than on its usual multi-month cadence.
ORACLE_FREQUENCY = 1

# Engine defaults for conditions the founder is never asked for. Surfaced in the
# UI as "estimated by the system", never as measurement.
#
# The last four used to be omitted from build_env_state entirely and picked up
# EnvState's own pydantic defaults instead. That is worse than a default: it is a
# default invisible even in the code that builds the state, so nothing could
# report it and the UI's "estimated inputs" count silently missed all four. They
# are set explicitly here so assumed_fields() can enumerate them.
DEFAULT_INTEREST_RATE = 3.0
DEFAULT_CONSUMER_CONFIDENCE = 100.0
DEFAULT_UNEMPLOYMENT = 4.0
DEFAULT_VALUATION_MULTIPLE = 10.0
DEFAULT_INNOVATION_FACTOR = 1.0
DEFAULT_MONTHS_IN_DEPRESSION = 0

# The boardroom's absolute spend floors are calibrated at this MRR (spec G11).
CALIBRATION_MRR = 50_000.0

# CFOAgent's own rule: no hiring under 24 months of runway. Enforced on the
# final action because an LLM proposal generator does not inherit it.
HIRING_RUNWAY_GUARD_MONTHS = 24.0

# How far above the published median a plan's discretionary spend may sit before
# it stops being advice and starts being a way to run out of money. The median
# itself comes from SaaS Capital's 2026 survey (n>1000) via calibration/bands.json;
# this multiple is a product judgement, not a measurement, and is stated as such.
DISCRETIONARY_SPEND_MEDIAN_MULTIPLE = 2.0


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

    The founder's costs travel as `monthly_costs` and land on EnvState as
    `monthly_burn`, which every burn consumer now reads through
    business_logic.monthly_burn: the physics, the board's runway estimate, the
    Oracle, the prompt the model actually sees, and both agent modules.

    They used to travel as *virtual headcount* - costs divided into $8k salary
    slots on the client - because Boardroom estimated runway as headcount * 8000.
    That encoding floored at one slot, so every company with costs between $0 and
    $12,000/month was charged exactly $8,000. A founder spending $500 was charged
    sixteen times over, died in month 0 of every projection, and was told by a
    prompt reading "Monthly burn: 8,000" that they had a cost problem.

    `initial_headcount` still arrives, but it is now the founder's real team size
    and carries no money.
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
        monthly_burn=(
            None
            if config.get("monthly_costs") is None
            else max(0.0, float(config["monthly_costs"]))
        ),
        unemployment=DEFAULT_UNEMPLOYMENT,
        valuation_multiple=DEFAULT_VALUATION_MULTIPLE,
        innovation_factor=DEFAULT_INNOVATION_FACTOR,
        months_in_depression=DEFAULT_MONTHS_IN_DEPRESSION,
    )


def assumed_fields(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Every EnvState field the founder did not supply, with its value and why.

    The UI previously showed a hardcoded count of "estimated inputs" computed on
    the client, which could not see the four macro fields the server filled in
    and therefore undercounted them. This is the server saying what it actually
    assumed, so the disclosure cannot drift from the behaviour.

    `correctable` splits them. Interest rate, consumer confidence, unemployment,
    valuation multiple and innovation factor are EnvState internals: no founder
    has an opinion on any of them, and inviting one to "enter anything here you
    actually know" invites an invented number into the analysis. They collapse
    into a single line about market conditions. The rest are things a founder
    genuinely could supply, and those are worth asking for.
    """
    config = payload.get("config") or {}
    assumed: list[dict[str, Any]] = []

    def add(field: str, value: Any, why: str, correctable: bool = False) -> None:
        assumed.append({"field": field, "value": value, "why": why,
                        "correctable": correctable})

    if config.get("interest_rate") is None:
        add("Interest rate", f"{DEFAULT_INTEREST_RATE}%", "not asked at onboarding; typical conditions")
    if config.get("consumer_confidence") is None:
        add("Consumer confidence", DEFAULT_CONSUMER_CONFIDENCE, "not asked at onboarding; index where 100 is neutral")
    add("Unemployment", f"{DEFAULT_UNEMPLOYMENT}%", "not asked at onboarding; typical conditions")
    add("Valuation multiple", f"{DEFAULT_VALUATION_MULTIPLE}x ARR", "not asked at onboarding; engine default")
    add("Innovation factor", DEFAULT_INNOVATION_FACTOR, "no scarring assumed at the start of an analysis")

    if config.get("monthly_costs") is None:
        add("Monthly costs", "$8,000 per person on the team",
            "not supplied, so the engine falls back to its own salary-slot convention; "
            "your real monthly costs change the plan more than any other number",
            correctable=True)
    if config.get("cac") is None:
        add("Acquisition cost", "$50",
            "not supplied and not derivable from marketing spend and new customers",
            correctable=True)
    if config.get("ltv") is None:
        add("Lifetime value", "price / monthly churn",
            "derived from your own numbers, never asked")
    if not any(config.get(k) is not None for k in ("churn_enterprise", "churn_smb", "churn_b2c")):
        add("Churn split", "one blended rate applied to all three segments",
            "the engine models enterprise, SMB and consumer churn separately; your blended figure fills all three, so the average it uses is exactly your number",
            correctable=True)
    return assumed


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


def _apply_spend_ceiling(action: dict[str, Any], state: EnvState) -> dict[str, Any] | None:
    """Cap marketing + product spend against the published median.

    The audit's remaining pre-revenue violation was a plan spending 125% of MRR.
    Median private B2B SaaS spends 8% of ARR on marketing and 22% on R&D, which
    for a monthly figure is 8% and 22% of MRR - so a sane ceiling exists in
    published data rather than having to be invented. Returns the applied
    ceiling for the trace, or None when the benchmark is absent (in which case
    nothing is capped, because an uncalibrated guard is worse than no guard).
    """
    benchmark = cal.discretionary_spend_pct_of_mrr()
    if not benchmark.is_observed or state.mrr <= 0:
        return None

    ceiling = state.mrr * (float(benchmark.value) / 100.0) * DISCRETIONARY_SPEND_MEDIAN_MULTIPLE
    # The only spend breakdown that survived verification is printed for the
    # $3-5M ARR band. Applying it to a founder two orders of magnitude smaller
    # is extrapolation, and the trace has to say so rather than let the citation
    # imply the figure was published for this company's size.
    extrapolated = not cal.spend_band_applies_to(state.mrr * 12.0)
    marketing = float((action.get("marketing") or {}).get("spend", 0.0) or 0.0)
    rnd = float((action.get("product") or {}).get("r_and_d_spend", 0.0) or 0.0)
    total = marketing + rnd
    if total <= ceiling:
        return {"ceiling_usd": round(ceiling), "applied": False,
                "median_pct_of_mrr": benchmark.value, "source": benchmark.citation(),
                "source_band": "$3-5M ARR", "extrapolated": extrapolated}

    # Scale both down proportionally rather than picking a winner: the board's
    # judgement about the product/marketing balance is preserved, only the
    # magnitude is corrected.
    factor = ceiling / total
    action.setdefault("marketing", {})["spend"] = marketing * factor
    action.setdefault("product", {})["r_and_d_spend"] = rnd * factor
    return {"ceiling_usd": round(ceiling), "applied": True, "scaled_by": round(factor, 3),
            "median_pct_of_mrr": benchmark.value, "source": benchmark.citation(),
            "source_band": "$3-5M ARR", "extrapolated": extrapolated}


def run_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """One analysis. Returns brief, decision trace and an honest llm_ok flag."""
    state = build_env_state(payload)

    scale = absolute_scale(state.mrr)

    # Memory isolation is injected, not inherited from CHROMA_PATH: .env.example
    # recommends pointing that at the research corpus, and a founder analysis
    # must never write there regardless of how the environment is configured.
    # Published median churn for this company's price point, if a source covers
    # it. None when it does not, in which case the prompt simply omits the line
    # rather than showing an invented comparison.
    churn_benchmark = cal.monthly_churn(state.price, kind="gross")

    oracle = Oracle(
        mode=ORACLE_MODE,
        memory_store=OracleMemoryStore(
            run_id=str(uuid.uuid4()),
            chroma_path=FOUNDER_CHROMA_PATH,
        ),
        include_burn_context=True,
        churn_benchmark_pct=(
            churn_benchmark.value * 100.0 if churn_benchmark.is_observed else None
        ),
    )

    proposal_generator = None
    if USE_CAUSAL_PROPOSALS:
        proposal_generator = BatchedCausalProposalGenerator(
            create_llm_client("ollama", "llama3.1:8b"),
            scale=scale,
        )

    boardroom = Boardroom(
        [
            CFOProposalAgent(scale=scale),
            CMOProposalAgent(scale=scale),
            CPOProposalAgent(scale=scale),
        ],
        use_oracle=True,
        oracle_mode=ORACLE_MODE,
        oracle_frequency=ORACLE_FREQUENCY,
        oracle_instance=oracle,
        proposal_generator=proposal_generator,
        scale_absolutes=scale,
        hiring_runway_guard_months=HIRING_RUNWAY_GUARD_MONTHS,
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
    final_action = trace.get("final_action") or action
    trace["spend_ceiling"] = _apply_spend_ceiling(final_action, state)
    trace["churn_benchmark"] = (
        {
            "median_monthly_pct": round(churn_benchmark.value * 100.0, 2),
            "company_monthly_pct": round(
                (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0 * 100.0, 2
            ),
            "arpa_band": cal.band_for_arpa(state.price),
            "source": churn_benchmark.citation(),
            "derivation": churn_benchmark.page_or_figure,
        }
        if churn_benchmark.is_observed
        else None
    )
    trace["final_action"] = final_action
    trace["assumed_fields"] = assumed_fields(payload)
    trace["history_months_replayed"] = months_replayed
    trace["absolute_scale"] = scale
    trace["graph_summary"] = _graph_summary(trace)

    # Engine vocabulary is translated once, here, and the client renders the
    # result. Raw brief and trace keys are untouched underneath for debugging.
    monthly_outflow = (
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
        "spend_ratio": founder_view.spend_ratio_phrase(monthly_outflow, state.mrr),
        "show_rule_of_40": founder_view.rule_of_40_is_meaningful(state.mrr),
        "monthly_burn": business_logic.monthly_burn(state),
        "monthly_burn_supplied": state.monthly_burn is not None,
    }

    return {
        "brief": brief,
        "trace": trace,
        "display": display,
        "llm_ok": llm_ok,
        "oracle_mode": ORACLE_MODE,
        "created_at": utc_now(),
    }


# Only causal edges belong in founder-facing evidence. OBSERVED_WITH links a
# stress node to raw action-pattern names ("marketing_high|rd_high|hires_0"),
# which is co-occurrence, not causation, and unreadable besides.
#
# The two causal predicates are NOT interchangeable and must not be presented
# as one thing. CONFIRMED_CAUSE is written from observed run outcomes.
# MAY_CAUSE is largely seeded by CausalGraphStore._ensure_seed_edges() -- a
# hand-authored prior with hand-assigned confidences, created "before live
# evidence exists". Rendering a seeded prior as something that happened in past
# runs would be a fabricated claim, so they travel separately.
CONFIRMED_PREDICATE = "CONFIRMED_CAUSE"
HYPOTHESIS_PREDICATE = "MAY_CAUSE"


def _graph_summary(trace: dict[str, Any]) -> dict[str, Any] | None:
    """Causal evidence behind this analysis, as engine vocabulary.

    Deliberately structured rather than prose: the enum-to-copy tables in
    frontend/src/copy.js are the only sanctioned path from engine terms to the
    screen (spec section 26), so this ships node names and lets the client
    translate them.
    """
    contexts = trace.get("causal_contexts") or {}
    if not contexts:
        return None

    observed: list[str] = []
    expected: list[str] = []
    confidences: list[float] = []
    for context in contexts.values():
        context = context or {}
        confidence = context.get("confidence")
        if confidence is not None:
            confidences.append(float(confidence))
        for triple in context.get("raw_triples") or []:
            if len(triple) < 3:
                continue
            subject, predicate, obj = triple[0], triple[1], triple[2]
            if predicate == CONFIRMED_PREDICATE and obj not in observed:
                observed.append(obj)
            elif predicate == HYPOTHESIS_PREDICATE and obj not in expected:
                expected.append(obj)

    if not observed and not expected:
        return None

    return {
        "stress_node": trace.get("causal_stress_node"),
        "observed": observed,
        "expected": expected,
        "confidence": (sum(confidences) / len(confidences)) if confidences else None,
        "roles": len(contexts),
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
                # display rides inside the trace so a stored analysis replays
                # with the same words it was first shown with, rather than
                # being re-translated by whatever the rules say later.
                json.dumps({**(result.get("trace") or {}),
                            "display": result.get("display")}),
                1 if result.get("llm_ok") else 0,
                result.get("oracle_mode"),
                result.get("created_at"),
                utc_now(),
            ),
        )
    return analysis_id
