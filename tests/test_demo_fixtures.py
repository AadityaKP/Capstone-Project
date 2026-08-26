"""The offline demonstration path (backend/demo_fixtures.py).

What these hold: the demo answers without Ollama or Neo4j, the recordings are
reachable by the inputs that produced them, and nothing on this path pretends a
model spoke when one did not.
"""

from __future__ import annotations

import copy
import json
import os

import pytest

from backend import demo_fixtures


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def payload(mrr, cash, costs, price, churn_pct, marketing, new_customers, age, team=2):
    """The shape frontend/src/api.js buildAdvisePayload sends."""
    return {
        "company_id": "demo-kettle",
        "company_age_months": age,
        "month_index": 0,
        "config": {
            "company_name": "Kettle Analytics",
            "initial_mrr": mrr, "initial_cash": cash, "average_price": price,
            "cac": round(marketing / new_customers, 2),
            "churn_enterprise": churn_pct / 100, "churn_smb": churn_pct / 100,
            "churn_b2c": churn_pct / 100, "competitors": 5, "product_quality": 0.5,
            "monthly_costs": costs, "initial_headcount": team,
        },
        "history": [],
    }


MONTH_1 = payload(11_000, 95_000, 15_500, 55, 3.4, 2_600, 20, age=15)
MONTH_2 = payload(12_800, 92_000, 16_400, 56, 2.9, 3_000, 24, age=16)


# --- the recordings exist and are reachable -------------------------------

def test_the_demo_dataset_replays():
    for name, request in (("month 1", MONTH_1), ("month 2", MONTH_2)):
        found = demo_fixtures.lookup(copy.deepcopy(request))
        assert found is not None, f"{name} has no recording"
        assert found["source"] == "recorded"
        assert found["llm_ok"] is True, (
            f"{name} was captured without the strategist; re-run "
            "record_demo_fixtures.py with Ollama up")
        assert found["trace"]["final_action"]["marketing"]["spend"] > 0


def test_the_two_months_are_different_analyses():
    """A signature collision would silently show month 1's advice twice."""
    one = demo_fixtures.lookup(copy.deepcopy(MONTH_1))
    two = demo_fixtures.lookup(copy.deepcopy(MONTH_2))
    assert one["trace"]["final_action"] != two["trace"]["final_action"]


def test_an_int_and_a_float_are_the_same_inputs():
    """The recorder builds its payload by hand, so initial_cash is the int
    95000; at runtime pydantic makes it the float 95000.0. Formatting them
    differently meant a recording could never match the request that made it."""
    as_int = copy.deepcopy(MONTH_1)
    as_float = copy.deepcopy(MONTH_1)
    as_float["config"] = {
        k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
        for k, v in as_float["config"].items()
    }
    assert demo_fixtures.signature(as_int) == demo_fixtures.signature(as_float)


def test_different_numbers_do_not_match_a_recording():
    """Near-miss inputs must fall through to the offline board rather than
    quietly showing advice computed for a different company."""
    other = copy.deepcopy(MONTH_1)
    other["config"]["initial_mrr"] = 11_500
    assert demo_fixtures.lookup(other) is None


def test_recordings_carry_the_inputs_that_made_them():
    """So a stale recording can be spotted and re-captured."""
    for filename in os.listdir(demo_fixtures.FIXTURE_DIR):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(demo_fixtures.FIXTURE_DIR, filename), encoding="utf-8") as fh:
            fixture = json.load(fh)
        assert fixture["inputs"]["initial_mrr"] > 0
        assert fixture["recorded_at"]
        assert fixture["signature"] == demo_fixtures.signature(
            {"company_age_months": fixture["company_age_months"],
             "config": fixture["inputs"]})


# --- the offline board ----------------------------------------------------

def test_unrecorded_numbers_still_get_a_real_analysis():
    result = demo_fixtures.offline_analysis(
        payload(7_400, 40_000, 9_800, 48, 5.1, 1_900, 15, age=12))
    assert result["source"] == "offline"
    assert result["trace"]["final_action"]["marketing"]["spend"] >= 0
    # $40,000 of cash against $9,800 of costs on $7,400 of revenue = 16.7 months.
    assert result["display"]["runway"] == "17 months of cash at current costs"


def test_the_offline_board_never_claims_a_model_spoke():
    result = demo_fixtures.offline_analysis(copy.deepcopy(MONTH_1))
    assert result["llm_ok"] is False
    assert result["brief"]["parse_ok"] is False


def test_the_offline_brief_is_derived_not_defaulted():
    """RiskChip falls back to MEDIUM for a missing risk_level, so an empty brief
    would have made the screen claim "Moderate risk" about a company nothing had
    assessed. Risk here follows runway."""
    healthy = demo_fixtures.offline_analysis(
        payload(20_000, 400_000, 22_000, 80, 2.0, 4_000, 30, age=24))
    dying = demo_fixtures.offline_analysis(
        payload(2_000, 6_000, 40_000, 40, 8.0, 1_000, 10, age=9))
    assert healthy["brief"]["risk_level"] == "LOW"
    assert dying["brief"]["risk_level"] == "CRITICAL"


def test_growth_outlook_needs_two_points():
    """One month of data cannot show a trend, so it does not claim one."""
    flat = demo_fixtures.offline_analysis(copy.deepcopy(MONTH_1))
    assert flat["brief"]["growth_outlook"] == "STABLE"

    grown = copy.deepcopy(MONTH_2)
    grown["history"] = [{"mrr": 11_000, "churn": 0.034, "entered_at": "2026-08-01T00:00:00Z"}]
    assert demo_fixtures.offline_analysis(grown)["brief"]["growth_outlook"] == "ACCELERATING"


def test_the_switch_turns_replay_off():
    """FOUNDER_DEMO_FIXTURES=0 restores the live stack."""
    os.environ["FOUNDER_DEMO_FIXTURES"] = "0"
    try:
        assert demo_fixtures.enabled() is False
        assert demo_fixtures.lookup(copy.deepcopy(MONTH_1)) is None
    finally:
        os.environ.pop("FOUNDER_DEMO_FIXTURES", None)
    assert demo_fixtures.enabled() is True
