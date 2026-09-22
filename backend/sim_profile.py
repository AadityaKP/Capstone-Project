"""Which physics/oracle configuration the product endpoints run (SIM_PROFILE).

Two profiles, resolved at call time from the SIM_PROFILE environment variable:

  review2  (default) The Review 2 research configuration: exactly the defaults
           the batch runner uses in run_simulation / _build_agent_for_policy.
           Every getter below returns empty kwargs or neutral values, so the
           engine objects are constructed the same way the thesis experiments
           construct them - repo chroma_db, oracle_v3, no founder scaling, no
           real-burn override, scheduled shocks on.

  founder  The founder-calibrated product configuration, unchanged from the
           founder-scale-fix line: scale-aware curves, real monthly_burn,
           scheduled shocks off, isolated chroma_db_founder, oracle_v4_causal,
           absolute floors scaled by mrr/50k, hiring runway guard.

Every place the API layer builds a StartupEnv, Boardroom, Oracle or agent must
take its profile-dependent kwargs from this module and nowhere else, so the
switch is the single point of truth. The founder constants live here (moved
from advise_service, which re-exports them for compatibility).
"""

from __future__ import annotations

import os
import uuid
from typing import Any

PROFILE_REVIEW2 = "review2"
PROFILE_FOUNDER = "founder"
_VALID_PROFILES = {PROFILE_REVIEW2, PROFILE_FOUNDER}

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Founder-profile constants (moved verbatim from advise_service)
# ---------------------------------------------------------------------------

FOUNDER_CHROMA_PATH = os.environ.get(
    "FOUNDER_CHROMA_PATH", os.path.join(ROOT_DIR, "chroma_db_founder")
)

# See advise_service's original comment block: plain oracle_v4_causal reads the
# graph only under an active shock, so the founder product pairs it with the
# batched causal proposal generator. FOUNDER_ORACLE_MODE=oracle_v4 runs without
# Neo4j entirely.
FOUNDER_ORACLE_MODE = os.getenv("FOUNDER_ORACLE_MODE", "oracle_v4_causal")

# One analysis per request, so the founder Oracle refreshes on every call.
# (The multi-month cycle overrides this to 0 so events earn the refresh -
# see backend/cycle_service.CYCLE_ORACLE_FREQUENCY.)
FOUNDER_ORACLE_FREQUENCY = 1

# Which marketing-response curve the founder physics run (docs/oefa_loop_decisions.md,
# decision 9 - an OPEN decision). "scale_aware" is the customer/CAC-anchored
# curve with the ASSUMED saturation rate 0.20, what the founder product has
# shipped with; "v2" is the same curve with the rate physics_v2 fitted on the
# CAL panel (0.0727), which falsified 0.20. The product's expected_delta is a
# prediction the founder is later scored against, and on the seeded demo
# company ($31.9k MRR, $10.2k of marketing) the assumed curve predicts MRR
# +41.9% over two months where the fitted one predicts +17.4%. Under the fitted
# curve the what-if fixtures in tests/test_whatif.py no longer survive, so the
# default stays as shipped until that trade-off is decided; set
# FOUNDER_MARKETING_CURVE=v2 to run the fitted curve.
FOUNDER_MARKETING_CURVE = os.getenv("FOUNDER_MARKETING_CURVE", "scale_aware")

# The boardroom's absolute spend floors are calibrated at this MRR (spec G11).
CALIBRATION_MRR = 50_000.0

# CFOAgent's own rule: no hiring under 24 months of runway. Enforced on the
# final action because an LLM proposal generator does not inherit it.
HIRING_RUNWAY_GUARD_MONTHS = 24.0

# ---------------------------------------------------------------------------
# Review-2 profile constants: the batch runner's own defaults.
# ---------------------------------------------------------------------------

# The memory arm of the Review 2 ablation: _build_agent_for_policy("oracle_v3")
# with run_simulation's default oracle_frequency.
REVIEW2_ORACLE_MODE = "oracle_v3"
REVIEW2_ORACLE_FREQUENCY = 3


def get_profile() -> str:
    """The active profile. Read per call so tests and harnesses can switch."""
    value = (os.getenv("SIM_PROFILE") or PROFILE_REVIEW2).strip().lower()
    if value not in _VALID_PROFILES:
        raise ValueError(
            f"SIM_PROFILE must be one of {sorted(_VALID_PROFILES)}, got {value!r}"
        )
    return value


def is_founder() -> bool:
    return get_profile() == PROFILE_FOUNDER


def absolute_scale(mrr: float) -> float:
    """Scale factor for the boardroom's absolute floors (G11).

    Clamped to <= 1.0 so companies at or above the calibration point keep the
    validated behaviour untouched; only smaller companies scale down. Floored
    at 0.05 so a pre-revenue company still gets a non-zero plan.
    """
    if mrr <= 0:
        return 0.05
    return min(1.0, max(0.05, mrr / CALIBRATION_MRR))


# ---------------------------------------------------------------------------
# Profile-resolved kwargs. review2 always returns the engine's own defaults
# (empty dicts / neutral values) - the whole point is that Review 2 behaviour
# is what the objects do when nothing is passed.
# ---------------------------------------------------------------------------


def get_agent_scale(mrr: float) -> float:
    """`scale` for the heuristic C-suite agents and proposal agents."""
    return absolute_scale(mrr) if is_founder() else 1.0


def get_oracle_mode() -> str:
    return FOUNDER_ORACLE_MODE if is_founder() else REVIEW2_ORACLE_MODE


def get_oracle_frequency() -> int:
    return FOUNDER_ORACLE_FREQUENCY if is_founder() else REVIEW2_ORACLE_FREQUENCY


def use_causal_proposals() -> bool:
    return is_founder() and get_oracle_mode() == "oracle_v4_causal"


def get_memory_scope(company_id: str | None) -> str:
    """The Oracle run_id for a product analysis: stable per company.

    OracleMemoryStore.retrieve_similar filters on run_id, so the scope is what
    decides whether an analysis can read what a previous one wrote. A fresh
    UUID per request (the previous behaviour) made founder memory write-only:
    nothing any earlier analysis stored was ever reachable. One key per company
    is what lets Adapt and the HITL close see the company's own past.

    Without a company id there is nothing to scope to, and a UUID keeps the
    old isolation rather than pooling anonymous requests together.
    """
    if not company_id:
        return str(uuid.uuid4())
    return f"company:{company_id}"


def get_oracle_kwargs(
    churn_benchmark_pct: float | None = None,
    company_id: str | None = None,
) -> dict[str, Any]:
    """Extra kwargs for Oracle(...) beyond mode.

    founder: the isolated chroma_db_founder store (never the research corpus),
    burn context in the prompt, and the published churn benchmark when one
    covers the company's price band.

    review2: only the memory scope and month de-duplication. Oracle(mode=
    "oracle_v3") still builds its own OracleMemoryStore against CHROMA_PATH
    (default ./chroma_db) and the research prompt template stays byte-identical;
    what changes is that a second analysis of the same company can now retrieve
    memories the first one matured, which under a per-request UUID it never
    could.
    """
    scope = get_memory_scope(company_id)
    if not is_founder():
        return {"run_id": scope, "dedupe_months": True}
    from oracle.memory import OracleMemoryStore

    return {
        "run_id": scope,
        "dedupe_months": True,
        "memory_store": OracleMemoryStore(
            run_id=scope,
            chroma_path=FOUNDER_CHROMA_PATH,
        ),
        "include_burn_context": True,
        "churn_benchmark_pct": churn_benchmark_pct,
    }


def get_boardroom_kwargs(mrr: float) -> dict[str, Any]:
    """Extra kwargs for Boardroom(...). review2: the batch runner passes
    neither, so Boardroom's own defaults (scale_absolutes=1.0, no hiring
    runway guard) apply."""
    if not is_founder():
        return {}
    return {
        "scale_absolutes": absolute_scale(mrr),
        "hiring_runway_guard_months": HIRING_RUNWAY_GUARD_MONTHS,
    }


def get_env_kwargs(gross_margin: float | None = None) -> dict[str, Any]:
    """initial_config for StartupEnv. review2: empty, i.e. StartupEnv()
    exactly as run_simulation constructs it - 120-month cap, scheduled shocks
    on, absolute marketing/R&D constants, revenue booked at 100% margin."""
    if not is_founder():
        return {}
    if FOUNDER_MARKETING_CURVE not in {"scale_aware", "v2"}:
        raise ValueError(
            f"FOUNDER_MARKETING_CURVE must be 'v2' or 'scale_aware', got {FOUNDER_MARKETING_CURVE!r}"
        )
    return {
        "max_months": 10_000,          # horizon is controlled by the caller
        "scheduled_shocks": False,     # research fixture, not founder physics
        # The customer/CAC-anchored curve either way (ships with the burn fix;
        # see whatif_service); "v2" additionally uses the fitted saturation rate.
        "scale_aware_marketing": True,
        "marketing_curve": FOUNDER_MARKETING_CURVE,
        "scale_aware_rnd": True,       # R&D that can move the product at all
        "gross_margin": gross_margin,
    }


def apply_monthly_burn() -> bool:
    """Whether founder-supplied monthly_costs land on EnvState.monthly_burn.

    review2 leaves monthly_burn=None so every burn consumer falls back to the
    engine's headcount-slot convention (headcount x $8,000), exactly as in
    every recorded research run.
    """
    return is_founder()


def apply_spend_ceiling() -> bool:
    """The calibrated discretionary-spend cap is a founder-product guard on
    the final action; Review 2 never modified the boardroom's decision."""
    return is_founder()


def get_applied_gross_margin() -> float | None:
    """The gross margin the environment actually applies, or None.

    review2: None - revenue books to cash at 100% margin, as in research runs.
    founder: the calibrated figure, when the source provides one.
    """
    if not is_founder():
        return None
    import calibration as cal

    margin = cal.gross_margin_pct()
    return (float(margin.value) / 100.0) if margin.value is not None else None
