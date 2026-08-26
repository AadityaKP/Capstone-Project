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
FOUNDER_ORACLE_FREQUENCY = 1

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


def get_oracle_kwargs(churn_benchmark_pct: float | None = None) -> dict[str, Any]:
    """Extra kwargs for Oracle(...) beyond mode.

    founder: the isolated chroma_db_founder store (never the research corpus),
    burn context in the prompt, and the published churn benchmark when one
    covers the company's price band.

    review2: nothing - Oracle(mode="oracle_v3") builds its own
    OracleMemoryStore against CHROMA_PATH (default ./chroma_db, the repo
    research corpus) and the research prompt stays byte-identical.
    """
    if not is_founder():
        return {}
    from oracle.memory import OracleMemoryStore

    return {
        "memory_store": OracleMemoryStore(
            run_id=str(uuid.uuid4()),
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
    return {
        "max_months": 10_000,          # horizon is controlled by the caller
        "scheduled_shocks": False,     # research fixture, not founder physics
        "scale_aware_marketing": True, # ships with the burn fix; see whatif_service
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
