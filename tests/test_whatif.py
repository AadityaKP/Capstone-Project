"""Tests for the what-if projection, the gated engine flags and the seed fix.

The bar these hold: every claim the projection panel makes to a founder should
fail a test if it stops being true. That includes the negative claims - that
turning a flag off changes nothing, and that a recovery figure is withheld when
no drawdown happened.
"""

from __future__ import annotations

import copy
import math
import random

import numpy as np
import pytest

import calibration as cal
from backend.advise_service import assumed_fields, build_env_state
from backend.whatif_service import (
    NOOP_ACTION,
    POLICIES,
    POLICY_HOLD,
    POLICY_RECOMMENDED,
    SHOCK_MONTH,
    _clean_action,
    _rule_based_action,
    run_whatif,
)
from env.startup_env import StartupEnv


FOUNDER = {
    "company_age_months": 20,
    "config": {
        "initial_mrr": 12000, "initial_cash": 60000, "average_price": 50, "cac": 300,
        "churn_enterprise": 0.05, "churn_smb": 0.05, "churn_b2c": 0.05,
        "competitors": 5, "product_quality": 0.5, "initial_headcount": 2,
    },
    "recommended_action": {
        "marketing": {"spend": 1800, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10000},
        "product": {"r_and_d_spend": 2100},
    },
    "current_marketing_spend": 1000,
    "n_seeds": 12,
}

FLAT_ACTION = {
    "marketing": {"spend": 5000.0, "channel": "ppc"},
    "hiring": {"hires": 0, "cost_per_employee": 10000.0},
    "product": {"r_and_d_spend": 3000.0},
    "pricing": {"price_change_pct": 0.0},
}


# --------------------------------------------------------------------------
# Gross margin: derived from printed components, and failing closed
# --------------------------------------------------------------------------
def test_gross_margin_is_derived_not_observed():
    """It must never claim to be printed: no source page prints a gross margin."""
    margin = cal.gross_margin_pct()
    assert margin.value == pytest.approx(83.5)
    assert margin.confidence == "derived"
    assert not margin.is_observed


def test_gross_margin_carries_its_own_composition():
    """A reader must be able to recompute it from the citation alone."""
    margin = cal.gross_margin_pct()
    assert margin.publisher == "SaaS Capital"
    for component in ("hosting", "devops", "pro_services_cogs", "other_cogs"):
        assert component in margin.page_or_figure


def test_gross_margin_equals_100_minus_printed_cogs():
    cogs = sum(
        float(cal.department_spend_pct_of_arr(name).value) for name in cal.COGS_COMPONENTS
    )
    assert cal.gross_margin_pct().value == pytest.approx(100.0 - cogs)


def test_gross_margin_withheld_when_a_component_is_missing(monkeypatch):
    """Half a CoGS sum overstates margin, which flatters cash. Fail closed."""
    real = cal.department_spend_pct_of_arr

    def missing_devops(name):
        if name == "devops":
            return cal.Calibrated(value=None, confidence="assumed")
        return real(name)

    monkeypatch.setattr(cal, "department_spend_pct_of_arr", missing_devops)
    assert cal.gross_margin_pct().value is None


# --------------------------------------------------------------------------
# The seed fix
# --------------------------------------------------------------------------
def _trajectory(seed: int) -> list[float]:
    env = StartupEnv()
    env.reset(seed=seed)
    return [round(env.step(FLAT_ACTION)[4]["state"]["mrr"], 4) for _ in range(6)]


def test_reset_seed_alone_is_reproducible():
    """business_logic draws from the global random module, which gym's
    super().reset() does not seed. Without the fix these differ."""
    assert _trajectory(42) == _trajectory(42)


def test_different_seeds_give_different_worlds():
    assert _trajectory(1) != _trajectory(2)


# --------------------------------------------------------------------------
# Gated flags change nothing when off
# --------------------------------------------------------------------------
def test_gross_margin_none_books_revenue_at_full_value():
    """The original behaviour, preserved exactly: cash += mrr, no CoGS."""
    env = StartupEnv()
    env.reset(seed=1)
    random.seed(1)
    before = env.state.cash
    _, _, _, _, info = env.step(NOOP_ACTION)
    salary = info["state"]["headcount"] * 8000.0
    assert info["state"]["cash"] == pytest.approx(before + info["state"]["mrr"] - salary)


def test_gross_margin_set_deducts_cost_of_revenue():
    env = StartupEnv(initial_config={"gross_margin": 0.835})
    env.reset(seed=1)
    random.seed(1)
    before = env.state.cash
    _, _, _, _, info = env.step(NOOP_ACTION)
    salary = info["state"]["headcount"] * 8000.0
    assert info["state"]["cash"] == pytest.approx(
        before + info["state"]["mrr"] * 0.835 - salary
    )


def test_scheduled_shocks_fire_by_default_and_can_be_disabled():
    """Month 24 carries a hard shock in research runs. A founder's projection
    must not inherit it."""
    def shock_at_month_24(scheduled: bool) -> str:
        env = StartupEnv(initial_config={"scheduled_shocks": scheduled})
        env.reset(seed=0)
        env.state.months_elapsed = 24
        return env.step(FLAT_ACTION)[4]["shock_label"]

    assert shock_at_month_24(True) != "NO_SHOCK"
    assert shock_at_month_24(False) == "NO_SHOCK"


# --------------------------------------------------------------------------
# Action handling
# --------------------------------------------------------------------------
def test_clean_action_never_invents_spend():
    action = _clean_action(None)
    assert action["marketing"]["spend"] == 0.0
    assert action["product"]["r_and_d_spend"] == 0.0
    assert action["hiring"]["hires"] == 0


def test_price_is_always_held_flat():
    """Elasticity is recorded as unidentified, so the projection must not move
    price even when asked to."""
    action = _clean_action({"pricing": {"price_change_pct": 0.25}})
    assert action["pricing"]["price_change_pct"] == 0.0


def test_rule_based_arm_is_scaled_to_the_company():
    """Unscaled, baseline_agents propose more marketing than a small company
    earns, making the comparator a strawman that wins by overspending."""
    state = build_env_state(FOUNDER)
    small = _rule_based_action(state, 12000 / 50000)
    unscaled = _rule_based_action(state, 1.0)
    assert small["marketing"]["spend"] < unscaled["marketing"]["spend"]
    assert small["product"]["r_and_d_spend"] < unscaled["product"]["r_and_d_spend"]


# --------------------------------------------------------------------------
# The projection itself
# --------------------------------------------------------------------------
def test_projection_returns_all_policies_with_full_bands():
    result = run_whatif(FOUNDER)
    assert set(result["policies"]) == set(POLICIES)
    for policy in POLICIES:
        series = result["policies"][policy]["series"]
        assert set(series) == {"mrr", "cash", "churn", "rule_of_40"}
        for panel in series.values():
            assert len(panel["median"]) == result["horizon_months"]
            assert len(panel["p25"]) == len(panel["p75"]) == result["horizon_months"]


def test_iqr_band_brackets_the_median():
    result = run_whatif(FOUNDER)
    mrr = result["policies"]["recommended"]["series"]["mrr"]
    for low, mid, high in zip(mrr["p25"], mrr["median"], mrr["p75"]):
        assert low <= mid <= high


def test_projection_is_reproducible():
    assert run_whatif(FOUNDER)["policies"] == run_whatif(FOUNDER)["policies"]


def test_policies_share_a_shock_tape_at_this_horizon():
    """The comparison is only clean while all three policies stay on the same
    RNG stream. Measured as true at 12 months; asserted so a change that breaks
    it is visible rather than silent."""
    assert run_whatif(FOUNDER)["shock_tape_shared"] is True


def test_recovery_is_withheld_when_no_drawdown_occurred():
    """competitor_surge cuts price and lifts churn but removes no revenue, so a
    growing company never dips. Reporting 'recovered in 0 months' for a drop
    that never happened would be a fabricated success."""
    result = run_whatif({**FOUNDER, "shock_mode": True})
    for policy in POLICIES:
        summary = result["policies"][policy]["summary"]
        if summary["drawdown_fraction"] == 0:
            assert summary["months_to_recover"] is None


def test_shock_mode_reports_a_cost_against_the_same_seeds():
    result = run_whatif({**FOUNDER, "shock_mode": True})
    assert result["shock"]["month"] == SHOCK_MONTH
    for policy in POLICIES:
        assert result["policies"][policy]["summary"]["shock_cost_pct"] is not None


def test_shock_only_hurts():
    """A shock that improved the outcome would mean the shock tape or the
    counterfactual is wired wrong."""
    result = run_whatif({**FOUNDER, "shock_mode": True})
    for policy in POLICIES:
        assert result["policies"][policy]["summary"]["shock_cost_pct"] <= 0


def test_caveat_and_assumptions_always_travel_with_the_numbers():
    result = run_whatif(FOUNDER)
    assert "not a forecast" in result["caveat"]
    fields = {a["field"] for a in result["assumptions"]}
    assert {"Gross margin", "Price", "Plan persistence"} <= fields
    margin = next(a for a in result["assumptions"] if a["field"] == "Gross margin")
    assert margin["basis"] == "derived"
    assert margin["source"] is not None


def test_missing_marketing_spend_is_declared_not_invented():
    result = run_whatif({k: v for k, v in FOUNDER.items() if k != "current_marketing_spend"})
    fields = {a["field"] for a in result["assumptions"]}
    assert "Current marketing spend" in fields


# --------------------------------------------------------------------------
# Assumed-value disclosure
# --------------------------------------------------------------------------
def test_assumed_fields_reports_the_macro_defaults_the_client_could_not_see():
    """These four used to fall to EnvState's pydantic defaults, invisible even
    to the code building the state, so nothing could report them."""
    fields = {a["field"] for a in assumed_fields(FOUNDER)}
    assert {"Interest rate", "Consumer confidence", "Unemployment",
            "Valuation multiple", "Innovation factor"} <= fields


def test_assumed_fields_omits_values_the_founder_supplied():
    supplied = {**FOUNDER, "config": {**FOUNDER["config"], "interest_rate": 5.0}}
    fields = {a["field"] for a in assumed_fields(supplied)}
    assert "Interest rate" not in fields


def test_every_assumed_field_explains_itself():
    for item in assumed_fields(FOUNDER):
        assert item["why"]
        assert item["value"] is not None


def test_build_env_state_sets_macro_fields_explicitly():
    """Set explicitly rather than inherited from the schema, so they are
    enumerable."""
    state = build_env_state(FOUNDER)
    assert state.unemployment == 4.0
    assert state.valuation_multiple == 10.0
    assert state.innovation_factor == 1.0


# --------------------------------------------------------------------------
# Founder scale: the case that used to die in month 0
#
# tests/test_whatif.py::FOUNDER above is $12k MRR with two $8k salary slots - a
# comfortable company that survives on the old constants, which is exactly why
# the failure below shipped green. Nothing here exercised a company small enough
# for the salary slot to matter until this fixture.
# --------------------------------------------------------------------------

FOUNDER_SMALL = {
    "company_age_months": 8,
    "config": {
        "initial_mrr": 2_500, "initial_cash": 5_000, "average_price": 25, "cac": 50,
        "churn_enterprise": 0.10, "churn_smb": 0.10, "churn_b2c": 0.10,
        "competitors": 5, "product_quality": 0.5,
        "monthly_costs": 500,          # the number that never used to arrive
        "initial_headcount": 1,
    },
    "recommended_action": {
        "marketing": {"spend": 250, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000},
        "product": {"r_and_d_spend": 100},
    },
    "current_marketing_spend": 10,
    "n_seeds": 12,
}


def test_a_founder_is_not_killed_by_the_engines_own_salary_slot():
    """The regression this change exists for.

    $2,500 MRR against $500/month of real costs. Every arm reported 0% survival,
    because the client encoded costs as one $8k salary slot and the physics
    charged it: the company was dead in month 0 whatever the plan said.
    """
    result = run_whatif(copy.deepcopy(FOUNDER_SMALL))
    for policy in (POLICY_RECOMMENDED, POLICY_HOLD):
        assert result["policies"][policy]["summary"]["survival_rate"] == 1.0


def test_the_projection_is_not_one_number_drawn_twelve_times():
    """Forward-filling a company that died in month 0 made every chart flat."""
    series = run_whatif(copy.deepcopy(FOUNDER_SMALL))["policies"][POLICY_HOLD]["series"]["mrr"]["median"]
    assert len(set(series)) > 1
    assert series[-1] > series[0]


def test_marketing_at_founder_scale_is_not_a_money_printer():
    """Under the absolute Hill constants $500 of ppc returned $1,255 of new MRR
    for a $2,500 company - half its revenue in one month - because beta is drawn
    as $10k-100k regardless of company size. Fixing burn without fixing this
    turned 0% survival into 4.3x growth on $250/month."""
    result = run_whatif(copy.deepcopy(FOUNDER_SMALL))
    median = result["policies"][POLICY_RECOMMENDED]["series"]["mrr"]["median"]
    gains = [(median[i] - median[i - 1]) / median[i - 1] for i in range(1, len(median))]
    assert max(gains) < 0.25, "10% of MRR on ppc must not compound faster than this"


def test_cac_cannot_run_away_under_the_scale_aware_curve():
    """marketing_curve_params places gamma from state.cac, so a month with a
    fractional response writes an enormous CAC, the CAC pushes gamma further
    right, and the next response is smaller still. Measured before the guard:
    cac 1.4e17 -> 9.0e43 in a single step, overflowing the float32 observation.

    A month that acquired a fraction of a customer has no cost-per-customer, so
    the previous estimate stands.
    """
    base = build_env_state(FOUNDER_SMALL)
    env = StartupEnv(initial_config={
        "max_months": 10_000, "scheduled_shocks": False,
        "scale_aware_marketing": True,
    })
    env.reset(seed=0)
    env.state = base.model_copy(deep=True)

    # $1 of brand spend sits far left of gamma: the response is a rounding error.
    starved = {
        "marketing": {"spend": 1.0, "channel": "brand"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 0.0},
        "pricing": {"price_change_pct": 0.0},
    }
    for _ in range(24):
        env.step(copy.deepcopy(starved))

    assert math.isfinite(env.state.cac)
    assert env.state.cac == base.cac


def test_without_the_guard_cac_is_the_runaway_this_prevents():
    """The negative half: same months, guard off. Pins that the guard is
    load-bearing rather than decorative.

    The divergence is super-exponential and does not settle at a large
    number - it leaves the float range. Left running it takes
    `hill_response` with it, because gamma ** alpha raises OverflowError once
    gamma passes ~1e103 at alpha 3. Either exit counts as the runaway; what
    is being pinned is that it happens at all.
    """
    base = build_env_state(FOUNDER_SMALL)
    env = StartupEnv(initial_config={
        "max_months": 10_000, "scheduled_shocks": False,
        "scale_aware_marketing": True, "stable_cac": False,
    })
    env.reset(seed=0)
    env.state = base.model_copy(deep=True)
    starved = {
        "marketing": {"spend": 1.0, "channel": "brand"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 0.0},
        "pricing": {"price_change_pct": 0.0},
    }
    diverged = False
    with np.errstate(over="ignore"):
        try:
            for _ in range(24):
                env.step(copy.deepcopy(starved))
                if env.state.cac > 1e30:
                    diverged = True
                    break
        except OverflowError:
            diverged = True
    assert diverged


def test_costs_reach_the_projection_and_change_the_answer():
    """Same company, same plan, costs supplied vs. left to the salary slot."""
    without = copy.deepcopy(FOUNDER_SMALL)
    without["config"].pop("monthly_costs")
    real = run_whatif(copy.deepcopy(FOUNDER_SMALL))
    slotted = run_whatif(without)
    assert real["policies"][POLICY_HOLD]["summary"]["survival_rate"] == 1.0
    assert slotted["policies"][POLICY_HOLD]["summary"]["survival_rate"] == 0.0
