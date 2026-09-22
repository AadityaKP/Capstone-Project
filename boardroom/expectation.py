"""expected_delta: the board's prediction, in a form that can be checked.

Plan section 3.2(a). A proposal's `expected_impact` is a slogan ("Lower burn,
improved survival probability") and a slogan cannot be scored against an
outcome, so neither the in-cycle Feedback step nor the founder's HITL close
has anything to compare. This module produces the number instead.

What the number is
------------------
The same physics that runs the what-if projection, rolled forward from the
current state under the proposed action held constant, over a small set of
seeds, reported as the median per-KPI change at the horizon. It is therefore
the board's *model-based* prediction: "if you do this, the simulator expects
MRR +4.0% and churn -0.3pp in two months." The language model chooses the
action on the causal path; it does not invent these numbers, because it has
no calibrated numeric sense and a checkable prediction is the whole point
(docs/oefa_loop_decisions.md, decision 5).

Two honesty notes carried in the payload rather than left implicit:

  basis="simulated"  Inside a cycle the realized month is also simulated, so
                     the in-cycle prediction error measures seed noise plus
                     the compounding of the model's own physics. Against a
                     founder's real month (the HITL close) it is the real
                     thing, and that is the number the thesis reports.
  band               p25/p75 of the end-of-horizon values across seeds. The
                     Plan page draws projection uncertainty with this band and
                     nothing else (plan section 5.2).

RNG hygiene
-----------
StartupEnv.reset(seed) reseeds the process-global `random` module. A cycle
that called this mid-trajectory would have its own world perturbed, so the
global state is saved and restored around the rollout, and the rollout's
environment runs on its private stream (deterministic_rng=True) so it
consumes no global draws at all.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Callable

import numpy as np

from backend import founder_view
from env import business_logic
from env.schemas import EnvState
from env.startup_env import StartupEnv

# The KPIs the board is held to. Percent changes for the money KPIs, points
# for churn, months for runway; one horizon for all of them.
KPI_KEYS = ("mrr_pct", "cash_pct", "churn_pp", "runway_months")
DEFAULT_HORIZON_MONTHS = 2
DEFAULT_N_SEEDS = 8
# Seeds for the expectation rollouts are offset from the cycle's own seed
# range so the prediction is never scored against the very seed it was made
# with.
EXPECTATION_SEED_START = 1_000

# Tolerance under which a miss is noise rather than a miss (used both for the
# prediction_error `within_tolerance` flag and the agents' track-record rule).
TOLERANCE = {"mrr_pct": 1.0, "cash_pct": 2.0, "churn_pp": 0.2, "runway_months": 0.5}

NOOP_ACTION: dict[str, Any] = {
    "marketing": {"spend": 0.0, "channel": "ppc"},
    "hiring": {"hires": 0, "cost_per_employee": 10000.0},
    "product": {"r_and_d_spend": 0.0},
    "pricing": {"price_change_pct": 0.0},
}

ExpectationFn = Callable[[EnvState, dict[str, Any]], dict[str, Any]]


def complete_action(partial: dict[str, Any] | None) -> dict[str, Any]:
    """A partial action bundle (one agent's domain) -> a complete one, with
    the missing domains at no-op. A proposal is therefore predicted in
    isolation against a no-spend base, which is the honest reading of "what
    this lever alone is expected to do"."""
    action = deepcopy(NOOP_ACTION)
    if not partial:
        return action
    for domain in ("marketing", "hiring", "product", "pricing"):
        block = partial.get(domain)
        if isinstance(block, dict):
            action[domain].update({k: v for k, v in block.items() if v is not None})
    action["marketing"]["spend"] = max(0.0, float(action["marketing"].get("spend", 0.0) or 0.0))
    if action["marketing"].get("channel") not in {"ppc", "brand"}:
        action["marketing"]["channel"] = "ppc"
    action["hiring"]["hires"] = max(0, int(action["hiring"].get("hires", 0) or 0))
    action["hiring"]["cost_per_employee"] = max(
        1.0, float(action["hiring"].get("cost_per_employee", 10000.0) or 10000.0)
    )
    action["product"]["r_and_d_spend"] = max(
        0.0, float(action["product"].get("r_and_d_spend", 0.0) or 0.0)
    )
    action["pricing"]["price_change_pct"] = float(
        action["pricing"].get("price_change_pct", 0.0) or 0.0
    )
    return action


def avg_churn_pct(state: EnvState) -> float:
    return (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0 * 100.0


def kpi_values(state: EnvState) -> dict[str, float | None]:
    """The four KPIs as levels, for the trace and the timeline cards."""
    return {
        "mrr": float(state.mrr),
        "cash": float(state.cash),
        "churn_pct": avg_churn_pct(state),
        "runway_months": founder_view.runway_months(
            state.cash, business_logic.monthly_burn(state), state.mrr
        ),
    }


def kpi_delta_between(before: EnvState, after: EnvState) -> dict[str, float | None]:
    """Signed change per KPI from one state to another, in the units of
    KPI_KEYS. runway_months is None when either side is not burning cash: a
    delta between a number and "not applicable" is not a number."""
    before_runway = founder_view.runway_months(
        before.cash, business_logic.monthly_burn(before), before.mrr
    )
    after_runway = founder_view.runway_months(
        after.cash, business_logic.monthly_burn(after), after.mrr
    )
    return {
        "mrr_pct": (after.mrr - before.mrr) / max(abs(before.mrr), 1.0) * 100.0,
        "cash_pct": (after.cash - before.cash) / max(abs(before.cash), 1.0) * 100.0,
        "churn_pp": avg_churn_pct(after) - avg_churn_pct(before),
        "runway_months": (
            None if before_runway is None or after_runway is None
            else after_runway - before_runway
        ),
    }


def prediction_error(expected: dict[str, Any] | None,
                     realized: dict[str, Any] | None) -> dict[str, Any] | None:
    """realized minus expected per KPI, with the tolerance verdict.

    A positive error on mrr_pct means the company did better than predicted;
    a positive error on churn_pp means churn was worse than predicted. The
    sign convention is "realized - expected" everywhere, and the founder-facing
    sentence ("we were 6% high") is built from it on the client.
    """
    if not expected or not realized:
        return None
    out: dict[str, Any] = {}
    for key in KPI_KEYS:
        exp = expected.get(key)
        got = realized.get(key)
        if exp is None or got is None:
            out[key] = None
            continue
        err = float(got) - float(exp)
        out[key] = {
            "expected": round(float(exp), 3),
            "realized": round(float(got), 3),
            "error": round(err, 3),
            "within_tolerance": abs(err) <= TOLERANCE[key],
            # Did the board even get the direction right? A miss on size is
            # calibration; a miss on sign is a wrong claim.
            "sign_agrees": (exp == 0 and abs(got) <= TOLERANCE[key])
                           or (exp > 0 and got > -TOLERANCE[key])
                           or (exp < 0 and got < TOLERANCE[key]),
        }
    scored = [v for v in out.values() if v is not None]
    out["summary"] = {
        "kpis_scored": len(scored),
        "within_tolerance": sum(1 for v in scored if v["within_tolerance"]),
        "sign_agrees": sum(1 for v in scored if v["sign_agrees"]),
        "horizon_months": expected.get("horizon_months"),
    }
    return out


def predict_expected_delta(
    state: EnvState,
    action: dict[str, Any] | None,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    n_seeds: int = DEFAULT_N_SEEDS,
    env_kwargs: dict[str, Any] | None = None,
    seed_start: int = EXPECTATION_SEED_START,
) -> dict[str, Any]:
    """Roll `action` forward `horizon_months` months over `n_seeds` seeds and
    return the median per-KPI delta plus the p25/p75 band of end values."""
    horizon = max(1, int(horizon_months))
    seeds = list(range(seed_start, seed_start + max(1, int(n_seeds))))
    full_action = complete_action(action)
    config = dict(env_kwargs or {})
    config["deterministic_rng"] = True

    saved_state = random.getstate()
    try:
        deltas: list[dict[str, float | None]] = []
        ends: dict[str, list[float]] = {"mrr": [], "cash": [], "churn_pct": []}
        survived = 0
        for seed in seeds:
            env = StartupEnv(initial_config=config)
            env.reset(seed=seed)
            env.state = state.model_copy(deep=True)
            alive = True
            for _ in range(horizon):
                _, _, terminated, _, _ = env.step(deepcopy(full_action))
                if terminated:
                    alive = False
                    break
            survived += int(alive)
            after = env.state
            deltas.append(kpi_delta_between(state, after))
            values = kpi_values(after)
            ends["mrr"].append(values["mrr"])
            ends["cash"].append(values["cash"])
            ends["churn_pct"].append(values["churn_pct"])
    finally:
        random.setstate(saved_state)

    def median_of(key: str) -> float | None:
        column = [d[key] for d in deltas if d.get(key) is not None]
        return round(float(np.median(column)), 3) if column else None

    def band_of(key: str) -> dict[str, float]:
        column = ends[key]
        return {
            "p25": round(float(np.percentile(column, 25)), 2),
            "median": round(float(np.median(column)), 2),
            "p75": round(float(np.percentile(column, 75)), 2),
        }

    return {
        **{key: median_of(key) for key in KPI_KEYS},
        "horizon_months": horizon,
        "n_seeds": len(seeds),
        "survival": round(survived / len(seeds), 3),
        "basis": "simulated",
        "band": {key: band_of(key) for key in ends},
    }


def make_expectation(
    env_kwargs: dict[str, Any] | None = None,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    n_seeds: int = DEFAULT_N_SEEDS,
) -> ExpectationFn:
    """The predictor the product path hands to agents and the Boardroom."""
    frozen = dict(env_kwargs or {})

    def expectation(state: EnvState, action: dict[str, Any] | None) -> dict[str, Any]:
        return predict_expected_delta(
            state, action,
            horizon_months=horizon_months, n_seeds=n_seeds, env_kwargs=frozen,
        )

    return expectation


# --------------------------------------------------------------------------
# Track record: what an agent proposed last time, what it predicted, and what
# actually happened (plan section 3.2b). One shape, used by the rule-based
# agents, the causal generator's prompt, and the cycle/HITL services.
# --------------------------------------------------------------------------

# Which KPI each role is held to. CFO owns cash; CMO owns revenue; CPO owns
# retention. The sign of a "good" move differs (churn down is good), which
# prediction_error already encodes via sign_agrees.
ROLE_KPI = {"CFO": "cash_pct", "CMO": "mrr_pct", "CPO": "churn_pp"}


def build_track_record(
    action: dict[str, Any] | None,
    expected: dict[str, Any] | None,
    realized: dict[str, Any] | None,
    month_label: str | None = None,
    source: str = "simulated",
) -> dict[str, Any] | None:
    """The record handed to the next month's agents. None when there is no
    prediction to be held to, so a cold first month carries nothing."""
    if not action or not expected:
        return None
    return {
        "month": month_label,
        "source": source,  # "simulated" inside a cycle; "observed" after a HITL close
        "action": deepcopy(action),
        "expected_delta": {k: expected.get(k) for k in (*KPI_KEYS, "horizon_months")},
        "realized_delta": (
            {k: realized.get(k) for k in KPI_KEYS} if realized else None
        ),
        "prediction_error": prediction_error(expected, realized),
    }


def role_miss(record: dict[str, Any] | None, role: str) -> dict[str, Any] | None:
    """The one KPI a role is held to, when last month's prediction for it
    missed on SIGN. A miss on size is calibration and does not change the
    heuristic; a miss on direction does."""
    if not record:
        return None
    error = (record.get("prediction_error") or {}).get(ROLE_KPI.get(role, ""))
    if not error or error.get("sign_agrees", True):
        return None
    return {"kpi": ROLE_KPI[role], **error}


def describe_track_record(record: dict[str, Any] | None) -> str:
    """The record as prompt text for an LLM agent."""
    if not record:
        return "none - this is the first month the board has been held to a prediction"
    lines = []
    action = record.get("action") or {}
    lines.append(
        "previous action: "
        f"marketing_spend={float((action.get('marketing') or {}).get('spend', 0) or 0):.0f}, "
        f"rd_spend={float((action.get('product') or {}).get('r_and_d_spend', 0) or 0):.0f}, "
        f"hires={int((action.get('hiring') or {}).get('hires', 0) or 0)}, "
        f"price_change_pct={float((action.get('pricing') or {}).get('price_change_pct', 0) or 0):.3f}"
    )
    errors = record.get("prediction_error") or {}
    for key in KPI_KEYS:
        item = errors.get(key)
        if not item:
            continue
        verdict = "as predicted" if item["sign_agrees"] else "WRONG DIRECTION"
        lines.append(
            f"{key}: predicted {item['expected']:+.2f}, actual {item['realized']:+.2f} "
            f"(error {item['error']:+.2f}, {verdict})"
        )
    if record.get("source") == "observed":
        lines.append("these actuals are the founder's real numbers, not a simulation")
    return "\n".join(lines)
