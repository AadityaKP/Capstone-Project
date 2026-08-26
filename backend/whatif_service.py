"""What-if: roll the founder's own state forward under competing policies (D5).

The advise path answers "what should I do this month". It never steps the
environment, so until now nothing in the product showed what *happens* if the
founder takes the board's advice. This module is that missing half.

Three policies over the same horizon, the same seeds and the same shock tape:

  A `recommended`  the board's plan for this month, held for the horizon
  B `hold`         the founder's current spend, held. The do-nothing arm.
  C `rule_based`   the heuristic C-suite (agents/baseline_agents), recomputed
                   each month from the state it finds itself in.

What makes this a fair comparison, and where that stops
-------------------------------------------------------
`env/business_logic.py` draws from the global `random` module, and its per-step
draw count is state-dependent: `apply_recession_cascade` only draws when
unemployment > 8 AND interest rate > 7. Two policies that reach different macro
states therefore desynchronise the shared stream and stop sharing a world.

Measured: over a 12-month horizon from founder-typical conditions that condition
is reached in 0 of 50 rollouts, because it needs roughly three interest-rate
shocks and four unemployment shocks to land inside a single year. So within this
horizon the three policies do share a shock tape, and the seed genuinely holds
the world fixed. `_verify_shared_shock_tape` re-checks that per request rather
than trusting the measurement, and the result travels in the response as
`shock_tape_shared`. At long horizons it would go False, and the comparison
would need the deeper fix in the environment.

One residual difference is deliberate rather than a defect: competitor entry
scales with MRR (`competitive_entry_shock`), so a policy that grows the company
faster genuinely attracts more competitors off the same random draw. That is the
model having an opinion, not the seeds failing.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any

import numpy as np

import calibration as cal
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent
from env import business_logic
from env.schemas import EnvState
from env.startup_env import StartupEnv

from backend.advise_service import absolute_scale, build_env_state

HORIZON_MONTHS = 12
N_SEEDS = 50
SEED_START = 0
SHOCK_MONTH = 6
SHOCK_TYPE = "competitor_surge"

POLICY_RECOMMENDED = "recommended"
POLICY_HOLD = "hold"
POLICY_RULE_BASED = "rule_based"
POLICIES = (POLICY_RECOMMENDED, POLICY_HOLD, POLICY_RULE_BASED)

POLICY_LABELS = {
    POLICY_RECOMMENDED: "Take the board's plan",
    POLICY_HOLD: "Keep doing what you're doing",
    POLICY_RULE_BASED: "Standard playbook",
}

CAVEAT = (
    "Simulated counterfactual. Demonstrates the model's internal dynamics, "
    "not a forecast of real-world outcomes."
)

NOOP_ACTION: dict[str, Any] = {
    "marketing": {"spend": 0.0, "channel": "ppc"},
    "hiring": {"hires": 0, "cost_per_employee": 10000.0},
    "product": {"r_and_d_spend": 0.0},
    "pricing": {"price_change_pct": 0.0},
}


def _clean_action(raw: dict[str, Any] | None) -> dict[str, Any]:
    """A partial action bundle -> a complete one, without inventing spend."""
    action = deepcopy(NOOP_ACTION)
    if not raw:
        return action
    marketing = raw.get("marketing") or {}
    hiring = raw.get("hiring") or {}
    product = raw.get("product") or {}
    pricing = raw.get("pricing") or {}
    action["marketing"]["spend"] = max(0.0, float(marketing.get("spend", 0.0) or 0.0))
    if marketing.get("channel") in {"ppc", "brand"}:
        action["marketing"]["channel"] = marketing["channel"]
    action["hiring"]["hires"] = max(0, int(hiring.get("hires", 0) or 0))
    action["hiring"]["cost_per_employee"] = max(
        1.0, float(hiring.get("cost_per_employee", 10000.0) or 10000.0)
    )
    action["product"]["r_and_d_spend"] = max(0.0, float(product.get("r_and_d_spend", 0.0) or 0.0))
    # Price changes are held flat across the horizon on purpose: the elasticity
    # behind them is recorded as unidentified in calibration/bands.json, so
    # projecting a price move would be projecting a number we have said we
    # cannot justify.
    action["pricing"]["price_change_pct"] = 0.0
    return action


def _rule_based_action(state: EnvState, scale: float) -> dict[str, Any]:
    """The heuristic C-suite, scaled to the company (spec G11).

    baseline_agents' dollar amounts are tuned for a ~$50k-MRR company. Run
    unscaled against a $12k-MRR founder they propose more marketing than the
    company earns, and the arm "wins" on MRR purely by burning cash it does not
    have. Scaling it is what makes it a fair comparator rather than a strawman -
    the same correction Boardroom already applies via scale_absolutes.
    """
    merged: dict[str, Any] = {}
    for agent in (CFOAgent(scale=scale), CMOAgent(scale=scale), CPOAgent(scale=scale)):
        merged.update(agent.act(state))
    return _clean_action(merged)


def _make_env(state: EnvState, gross_margin: float | None) -> StartupEnv:
    """The environment a founder is projected in.

    `scale_aware_marketing` is on here and off in research runs, and it had to
    ship in the same change as the burn fix rather than after it. Measured on a
    $2,500-MRR founder, same seeds, recommended arm:

        as shipped               $3,404 flat,  0% survival
        real burn only          $10,802,     100% survival
        real burn + this flag    $4,431,     100% survival
        this flag only           $2,591 flat,  0% survival

    The burn constant is the sole cause of death - this flag alone changes
    nothing. But the absolute Hill constants are a money printer at founder
    scale, because beta is drawn as $10k-100k of new MRR regardless of company
    size: $250/month of ads takes a $2,500 company to $10,802 in a year. Fixing
    burn alone would replace a visibly broken projection with a plausible-looking
    wrong one, which is worse. See marketing_curve_params for the reparameterised
    curve and docs/founder_scale_fix_plan.md for the full measurement.
    """
    return StartupEnv(
        initial_config={
            "max_months": 10_000,          # horizon is controlled by the caller
            "scheduled_shocks": False,     # research fixture, not founder physics
            "scale_aware_marketing": True, # see above - ships with the burn fix
            "gross_margin": gross_margin,
        }
    )


def _rollout(
    base_state: EnvState,
    policy: str,
    seed: int,
    recommended: dict[str, Any],
    hold: dict[str, Any],
    horizon: int,
    gross_margin: float | None,
    shock: bool,
    scale: float,
) -> dict[str, Any]:
    """One seeded 12-month path. Returns per-month series plus outcome flags."""
    env = _make_env(base_state, gross_margin)
    env.reset(seed=seed)                       # also seeds the global RNG
    env.state = base_state.model_copy(deep=True)

    mrr, cash, churn, rule40 = [], [], [], []
    survived = True
    pre_shock_mrr: float | None = None
    recovery_month: int | None = None
    drawdown = False
    cascade_triggered = False

    for month in range(horizon):
        if shock and month == SHOCK_MONTH:
            pre_shock_mrr = env.state.mrr
            business_logic.inject_hard_shock(env.state, SHOCK_TYPE)

        if policy == POLICY_RECOMMENDED:
            action = recommended
        elif policy == POLICY_HOLD:
            action = hold
        else:
            action = _rule_based_action(env.state, scale)

        # Watch for the state that would desynchronise the shared RNG stream.
        if env.state.unemployment > 8.0 and env.state.interest_rate > 7.0:
            cascade_triggered = True

        _, _, terminated, _, info = env.step(deepcopy(action))
        state = env.state

        mrr.append(state.mrr)
        cash.append(state.cash)
        churn.append(business_logic.compute_churn_rate(state) * 100.0)
        rule40.append(info["rule_of_40"])

        # "Months to recover" only means something once MRR has actually fallen
        # below its pre-shock level. competitor_surge cuts price and lifts churn
        # rather than removing revenue outright, so a company growing fast enough
        # never dips - and reporting "recovered in 0 months" for a drawdown that
        # never happened would be a fabricated success. Track the dip first.
        if pre_shock_mrr is not None:
            if state.mrr < pre_shock_mrr:
                drawdown = True
            elif drawdown and recovery_month is None:
                recovery_month = month - SHOCK_MONTH

        if terminated:
            survived = False
            # Pad the remaining months so every path is the same length; a dead
            # company stays dead rather than being dropped from the median.
            remaining = horizon - len(mrr)
            mrr.extend([state.mrr] * remaining)
            cash.extend([state.cash] * remaining)
            churn.extend([churn[-1]] * remaining)
            rule40.extend([rule40[-1]] * remaining)
            break

    return {
        "mrr": mrr, "cash": cash, "churn": churn, "rule40": rule40,
        "survived": survived, "recovery_month": recovery_month,
        "drawdown": drawdown, "cascade_triggered": cascade_triggered,
    }


def _bands(series: list[list[float]]) -> dict[str, list[float]]:
    """Median and 25-75 interquartile band, per month, across seeds."""
    matrix = np.asarray(series, dtype=float)
    return {
        "median": np.median(matrix, axis=0).round(2).tolist(),
        "p25": np.percentile(matrix, 25, axis=0).round(2).tolist(),
        "p75": np.percentile(matrix, 75, axis=0).round(2).tolist(),
    }


def run_whatif(payload: dict[str, Any]) -> dict[str, Any]:
    """Roll the founder's state forward under all three policies."""
    horizon = int(payload.get("horizon_months") or HORIZON_MONTHS)
    n_seeds = int(payload.get("n_seeds") or N_SEEDS)
    shock = bool(payload.get("shock_mode", False))
    seeds = list(range(SEED_START, SEED_START + n_seeds))

    base_state = build_env_state(payload)

    margin = cal.gross_margin_pct()
    gross_margin = (float(margin.value) / 100.0) if margin.value is not None else None

    recommended = _clean_action(payload.get("recommended_action"))

    supplied_costs = (payload.get("config") or {}).get("monthly_costs")

    # The do-nothing arm holds what the founder is spending today. Marketing
    # spend is an optional onboarding field; when it is absent the arm holds at
    # zero and the response says so rather than inventing a level for it.
    current_marketing = payload.get("current_marketing_spend")
    hold = deepcopy(NOOP_ACTION)
    if current_marketing is not None:
        hold["marketing"]["spend"] = max(0.0, float(current_marketing))

    scale = absolute_scale(base_state.mrr)

    results: dict[str, Any] = {}
    cascade_seen = False

    for policy in POLICIES:
        paths = [
            _rollout(base_state, policy, seed, recommended, hold, horizon,
                     gross_margin, shock, scale)
            for seed in seeds
        ]
        cascade_seen = cascade_seen or any(p["cascade_triggered"] for p in paths)

        terminal_mrr = [p["mrr"][-1] for p in paths]
        terminal_cash = [p["cash"][-1] for p in paths]
        mean_rule40 = [float(np.mean(p["rule40"])) for p in paths]
        median_terminal_mrr = float(np.median(terminal_mrr))

        summary: dict[str, Any] = {
            "median_terminal_mrr": round(median_terminal_mrr, 2),
            "median_terminal_cash": round(float(np.median(terminal_cash)), 2),
            "survival_rate": round(sum(p["survived"] for p in paths) / len(paths), 3),
            "mean_rule_of_40": round(float(np.mean(mean_rule40)), 2),
            "months_to_recover": None,
            "drawdown_fraction": None,
            "shock_cost_pct": None,
        }

        if shock:
            drawdowns = [p for p in paths if p["drawdown"]]
            recoveries = [p["recovery_month"] for p in drawdowns if p["recovery_month"] is not None]
            summary["drawdown_fraction"] = round(len(drawdowns) / len(paths), 3)
            summary["months_to_recover"] = (
                round(float(np.median(recoveries)), 1) if recoveries else None
            )
            # What the shock actually costs: the same policy, the same seeds,
            # with and without it. This is the comparison that stays meaningful
            # whether or not MRR dips, and it is why the clean arm is re-run.
            clean = [
                _rollout(base_state, policy, seed, recommended, hold, horizon,
                         gross_margin, False, scale)
                for seed in seeds
            ]
            clean_median = float(np.median([p["mrr"][-1] for p in clean]))
            if clean_median > 0:
                summary["shock_cost_pct"] = round(
                    (median_terminal_mrr - clean_median) / clean_median * 100.0, 1
                )

        results[policy] = {
            "label": POLICY_LABELS[policy],
            "series": {
                "mrr": _bands([p["mrr"] for p in paths]),
                "cash": _bands([p["cash"] for p in paths]),
                "churn": _bands([p["churn"] for p in paths]),
                "rule_of_40": _bands([p["rule40"] for p in paths]),
            },
            "summary": summary,
        }

    return {
        "horizon_months": horizon,
        "n_seeds": n_seeds,
        "seeds": f"{seeds[0]}-{seeds[-1]}",
        "shock_mode": shock,
        "shock": (
            {"month": SHOCK_MONTH, "type": SHOCK_TYPE,
             "description": "3 new entrants, forced price cut, SMB churn +50%"}
            if shock else None
        ),
        "policies": results,
        "starting_state": {
            "mrr": base_state.mrr, "cash": base_state.cash,
            "price": base_state.price, "headcount": base_state.headcount,
            "months_elapsed": base_state.months_elapsed,
            # The number the whole projection turns on, and the one that used
            # to be silently replaced by a $8k salary slot. It travels back so
            # a founder can see which figure was actually charged.
            "monthly_burn": business_logic.monthly_burn(base_state),
            "monthly_burn_supplied": base_state.monthly_burn is not None,
        },
        "recommended_action": recommended,
        "caveat": CAVEAT,
        "assumptions": _assumptions(margin, current_marketing, shock,
                                    base_state, supplied_costs),
        # False means the policies stopped sharing a world and the comparison is
        # no longer clean. Surfaced, never silently swallowed.
        "shock_tape_shared": not cascade_seen,
    }


def _assumptions(
    margin: cal.Calibrated,
    current_marketing: float | None,
    shock: bool,
    base_state: EnvState,
    supplied_costs: float | None,
) -> list[dict[str, Any]]:
    """Every modelling choice this projection rests on, stated in full."""
    items: list[dict[str, Any]] = [
        {
            "field": "Monthly costs",
            "value": (
                f"${business_logic.monthly_burn(base_state):,.0f}/mo"
                + ("" if supplied_costs is not None else " (estimated)")
            ),
            "basis": "reported" if supplied_costs is not None else "assumption",
            "source": None,
            "detail": (
                "Your own figure, charged against cash every month."
                if supplied_costs is not None
                else "You did not supply monthly costs, so the engine charges its own "
                     "convention of $8,000 per person on the team. For a small team that "
                     "is usually far more than the truth and it dominates the result - "
                     "entering your real costs changes this projection more than any "
                     "other number."
            ),
        },
        {
            "field": "Gross margin",
            "value": f"{margin.value}%" if margin.value is not None else "not applied",
            "basis": margin.confidence,
            "source": margin.citation(),
            "detail": margin.page_or_figure,
        },
        {
            "field": "Price",
            "value": "held flat",
            "basis": "unidentified",
            "source": None,
            "detail": (
                "Price elasticity has no public dataset and is recorded as unidentified "
                "in calibration/bands.json, so the projection does not move price."
            ),
        },
        {
            "field": "Plan persistence",
            "value": "this month's plan repeated for the horizon",
            "basis": "assumption",
            "source": None,
            "detail": (
                "The board is asked once. A founder re-running the analysis monthly would "
                "get a different path; this shows the plan held constant."
            ),
        },
        {
            "field": "Scheduled research shocks",
            "value": "disabled",
            "basis": "assumption",
            "source": None,
            "detail": (
                "The engine's fixed shocks at months 24/48/72 belong to the 120-month "
                "research episode and would otherwise land inside a founder's horizon."
            ),
        },
    ]
    if current_marketing is None:
        items.append({
            "field": "Current marketing spend",
            "value": "$0",
            "basis": "assumption",
            "source": None,
            "detail": (
                "Not supplied at onboarding, so the do-nothing arm holds marketing at zero. "
                "Entering last month's marketing spend makes that comparison fairer."
            ),
        })
    if shock:
        items.append({
            "field": "Shock",
            "value": f"competitor surge at month {SHOCK_MONTH}",
            "basis": "assumption",
            "source": None,
            "detail": "Identical shock applied to all three policies at the same month.",
        })
    return items


def _verify_shared_shock_tape(result: dict[str, Any]) -> bool:
    return bool(result.get("shock_tape_shared"))
