"""Step 6 of docs/founder_scale_fix_plan.md: the two contracts that hold the
product together.

1. Determinism - every number the founder sees is a pure function of what they
   typed. Before the burn fix, more than half were not: costs were quantised
   into $8k salary slots on the client, and five macro fields were filled in by
   the server without appearing anywhere in the payload.

2. Non-contradiction - the frontend formula and the simulator cannot disagree
   about whether the company is dying. This is the executable form of "two
   engines, one screen", and it failed for every company below roughly $8,000
   MRR: Home printed runway ∞ while the projection reported 0% survival.
"""

from __future__ import annotations

import copy
import glob
import os
import re

import pytest

from backend import founder_view
from backend.advise_service import build_env_state
from backend.whatif_service import POLICY_HOLD, run_whatif
from env import business_logic


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def payload(mrr, cash, costs, price=25.0, churn=0.10, age=8, headcount=1):
    """Exactly the shape frontend/src/api.js buildAdvisePayload produces."""
    return {
        "company_id": "contract",
        "company_age_months": age,
        "config": {
            "company_name": "Contract Co",
            "initial_mrr": mrr,
            "initial_cash": cash,
            "average_price": price,
            "cac": 50,
            "churn_enterprise": churn,
            "churn_smb": churn,
            "churn_b2c": churn,
            "competitors": 5,
            "product_quality": 0.5,
            "monthly_costs": costs,
            "initial_headcount": headcount,
        },
        "history": [],
        "n_seeds": 12,
    }


# --------------------------------------------------------------------------
# 1. Determinism
# --------------------------------------------------------------------------

def test_every_engine_input_is_a_pure_function_of_what_the_founder_typed():
    state = build_env_state(payload(mrr=2_500, cash=5_000, costs=500))

    assert state.mrr == 2_500
    assert state.cash == 5_000
    assert state.price == 25.0
    assert state.monthly_burn == 500          # not round(500 / 8000) -> 1 -> $8,000
    assert state.headcount == 1
    assert state.churn_enterprise == state.churn_smb == state.churn_b2c == 0.10
    assert state.months_elapsed == 8
    assert state.cac == 50
    assert state.competitors == 5
    assert state.product_quality == 0.5
    # LTV is derived from the founder's own price and churn, never asked
    assert state.ltv == pytest.approx(25.0 / 0.10)


def test_building_the_state_twice_gives_the_same_state():
    base = payload(mrr=12_000, cash=60_000, costs=16_000)
    assert build_env_state(copy.deepcopy(base)) == build_env_state(copy.deepcopy(base))


@pytest.mark.parametrize("costs", [0, 250, 500, 4_000, 7_999, 8_000, 12_000, 30_000])
def test_costs_survive_the_trip_at_every_size(costs):
    """The salary-slot encoding mapped everything from $0 to $12,000 onto one
    $8,000 charge. Nothing is rounded now."""
    state = build_env_state(payload(mrr=2_500, cash=50_000, costs=costs))
    assert business_logic.monthly_burn(state) == costs


def test_omitting_costs_is_visible_rather_than_silent():
    """The fallback still exists for older clients, but it announces itself."""
    request = payload(mrr=2_500, cash=5_000, costs=500)
    request["config"].pop("monthly_costs")
    state = build_env_state(request)
    assert state.monthly_burn is None
    assert business_logic.monthly_burn(state) == 8_000

    result = run_whatif(request)
    assert result["starting_state"]["monthly_burn_supplied"] is False
    costs_note = next(a for a in result["assumptions"] if a["field"] == "Monthly costs")
    assert costs_note["basis"] == "assumption"


# --------------------------------------------------------------------------
# 2. Non-contradiction: the two engines cannot disagree about death
# --------------------------------------------------------------------------

@pytest.mark.parametrize("mrr,cash,costs", [
    (2_500, 5_000, 500),        # founder-only; failed before the burn fix
    (6_000, 20_000, 3_000),     # under the old $8k floor
    (12_000, 60_000, 9_000),    # over it
    (50_000, 500_000, 40_000),  # the calibration company
])
def test_a_company_that_is_not_burning_cash_is_not_killed_by_the_simulator(mrr, cash, costs):
    request = payload(mrr=mrr, cash=cash, costs=costs)
    state = build_env_state(request)

    # The frontend's own formula, via the shared translation layer.
    runway = founder_view.runway_months(
        state.cash, business_logic.monthly_burn(state), state.mrr
    )
    assert runway is None, "fixture must be cash-flow positive for this to mean anything"

    survival = run_whatif(request)["policies"][POLICY_HOLD]["summary"]["survival_rate"]
    assert survival > 0, (
        f"Home would print 'not burning cash' while the projection kills "
        f"{(1 - survival) * 100:.0f}% of runs"
    )


def test_a_company_that_really_is_dying_is_reported_as_dying_by_both():
    """The contract is agreement, not optimism."""
    request = payload(mrr=2_000, cash=3_000, costs=40_000)
    state = build_env_state(request)

    runway = founder_view.runway_months(
        state.cash, business_logic.monthly_burn(state), state.mrr
    )
    assert runway is not None and runway < 1

    summary = run_whatif(request)["policies"][POLICY_HOLD]["summary"]
    assert summary["survival_rate"] == 0.0
    assert summary["median_death_month"] is not None


# --------------------------------------------------------------------------
# 3. No engine vocabulary reaches a component
# --------------------------------------------------------------------------

# Fields that are simulator state and have no founder-facing meaning. Rule of 40
# is deliberately absent: above $1M ARR it is the right metric and founders in
# that band use the term, so founder_view gates it by size rather than banning it.
ENGINE_ONLY_FIELDS = [
    "innovation_factor", "valuation_multiple", "months_in_depression",
    "consumer_confidence", "product_quality", "months_elapsed",
    "churn_smb", "churn_b2c", "churn_enterprise",
    "unemployment", "interest_rate",
]


def _code_only(source: str) -> str:
    """Source with comments removed.

    Scanning raw text made a comment explaining that a field is no longer
    rendered count as rendering it, which is the opposite of what these tests
    are for. Handles // line comments, /* */ blocks and the {/* */} form JSX
    uses; the negative lookbehind keeps :// in a URL intact.
    """
    without_blocks = re.sub(r"/\*.*?\*/", " ", source, flags=re.DOTALL)
    return re.sub(r"(?<!:)//[^\n]*", " ", without_blocks)


def _frontend_sources(pattern: str) -> list[tuple[str, str]]:
    found = []
    for path in sorted(glob.glob(os.path.join(ROOT, "frontend", "src", "**", pattern),
                                 recursive=True)):
        with open(path, encoding="utf-8") as handle:
            found.append((os.path.basename(path), _code_only(handle.read())))
    return found


def test_no_component_renders_a_raw_engine_field():
    """The translation happens once, at the boundary. api.js is exempt: mapping
    founder inputs onto ScenarioConfig is its entire job."""
    offenders = [
        f"{name}: {field}"
        for name, source in _frontend_sources("*.jsx")
        for field in ENGINE_ONLY_FIELDS
        if field in source
    ]
    assert not offenders, "engine vocabulary leaked into components: " + ", ".join(offenders)


def test_infinity_is_not_renderable():
    """Runway ∞ was printed beside a projection that said the company died in
    month 1. The primitive no longer produces it, so no caller can."""
    offenders = [name for name, source in _frontend_sources("*.js*") if "\u221e" in source]
    assert not offenders, "infinity glyph still rendered by: " + ", ".join(offenders)
