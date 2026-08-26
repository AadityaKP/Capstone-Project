"""Tests for the product guards added on the founder path.

Every guard here is opt-in: research runs must reproduce byte-identically with
the flags at their defaults, and each test pins both sides of that.
"""

import copy
import hashlib
import json

import pytest

from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent, merge_actions
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
from boardroom.boardroom import Boardroom
from env import business_logic
from env.schemas import EnvState
from env.startup_env import StartupEnv
from oracle.oracle import Oracle
from oracle.parser import parse_llm_response
from oracle.prompt_builder import build_prompt


def state(mrr=12_000, cash=90_000, costs=24_000, price=40, cac=91, churn=0.05):
    return EnvState(
        mrr=mrr, cash=cash, cac=cac, ltv=price / max(churn, 1e-6),
        churn_enterprise=churn, churn_smb=churn, churn_b2c=churn,
        interest_rate=3, consumer_confidence=100, competitors=5,
        product_quality=0.5, price=price, months_elapsed=12,
        headcount=1, monthly_burn=costs,
    )


def agents(scale=1.0):
    return [CFOProposalAgent(scale=scale), CMOProposalAgent(scale=scale), CPOProposalAgent(scale=scale)]


def action(hires=1, marketing=3_000.0, rnd=4_000.0):
    return {
        "marketing": {"spend": marketing, "channel": "ppc"},
        "hiring": {"hires": hires, "cost_per_employee": 10_000},
        "product": {"r_and_d_spend": rnd},
        "pricing": {"price_change_pct": 0.0},
    }


# --- G3: honest failure flag ----------------------------------------------

def test_parsed_brief_is_flagged_parseable():
    brief = parse_llm_response('{"risk_level":"HIGH","growth_outlook":"DECLINING","confidence":0.8}')
    assert brief.parse_ok is True
    assert brief.risk_level.value == "HIGH"


@pytest.mark.parametrize("raw", ["", "not json", "{unclosed", "null"])
def test_fallback_brief_is_flagged_unparseable(raw):
    """A neutral fallback must never be indistinguishable from a real read."""
    brief = parse_llm_response(raw)
    assert brief.parse_ok is False
    assert brief.risk_level.value == "MEDIUM"  # safe default, carrying no signal


# --- G11: calibration scaling ---------------------------------------------

def test_agent_dollar_constants_scale():
    small = CMOAgent(scale=0.24).act(state())["marketing"]["spend"]
    full = CMOAgent(scale=1.0).act(state())["marketing"]["spend"]
    assert small == pytest.approx(full * 0.24)


def test_research_defaults_are_unchanged():
    """The original constants, reproduced exactly at default scale."""
    merged = merge_actions(state())
    assert merged["marketing"]["spend"] == pytest.approx(20_000)
    assert merged["hiring"]["cost_per_employee"] == pytest.approx(10_000)
    assert CPOAgent().act(state())["product"]["r_and_d_spend"] == pytest.approx(7_500)


def test_boardroom_floors_scale_but_default_to_originals():
    st = state()
    empty = {"marketing": {"spend": 0}, "hiring": {"hires": 0}, "product": {"r_and_d_spend": 0}}

    research = Boardroom(agents())._apply_dynamic_minimums(copy.deepcopy(empty), st, 0.5)
    assert research["product"]["r_and_d_spend"] == pytest.approx(20_000)
    assert research["marketing"]["spend"] == pytest.approx(5_000)

    product = Boardroom(agents(), scale_absolutes=0.24)._apply_dynamic_minimums(
        copy.deepcopy(empty), st, 0.5
    )
    assert product["product"]["r_and_d_spend"] == pytest.approx(20_000 * 0.24)
    assert product["marketing"]["spend"] == pytest.approx(5_000 * 0.24)


# --- hiring runway guard ---------------------------------------------------

def test_hiring_guard_is_off_by_default():
    st = state()  # ~3.8 months of engine runway
    kept = Boardroom(agents())._apply_sanity_bounds(action(hires=1), st)
    assert kept["hiring"]["hires"] == 1


def test_hiring_guard_blocks_when_runway_is_short():
    board = Boardroom(agents(), hiring_runway_guard_months=24.0)
    blocked = board._apply_sanity_bounds(action(hires=1), state())
    assert blocked["hiring"]["hires"] == 0


def test_hiring_guard_allows_when_runway_is_long():
    board = Boardroom(agents(), hiring_runway_guard_months=24.0)
    allowed = board._apply_sanity_bounds(action(hires=1), state(cash=900_000))
    assert allowed["hiring"]["hires"] == 1


def test_guard_runway_counts_revenue():
    """_estimate_runway_months ignores MRR and reads 8.9 months for a company
    with 100; a guard using it would block hiring at healthy companies."""
    st = state(mrr=200_000, cash=2_000_000, costs=220_000, price=150, cac=140, churn=0.02)
    board = Boardroom(agents(), hiring_runway_guard_months=24.0)
    assert board._estimate_runway_months(st) < 12          # the flawed measure
    assert board._net_runway_months(st) > 24               # the correct one
    assert board._apply_sanity_bounds(action(hires=1), st)["hiring"]["hires"] == 1


def test_net_runway_is_infinite_when_cash_flow_positive():
    st = state(mrr=100_000, cash=50_000, costs=24_000)
    assert Boardroom(agents())._net_runway_months(st) == float("inf")


# --- burn context in the Oracle prompt ------------------------------------

def test_research_prompt_carries_no_burn_context():
    prompt = build_prompt(state(), mode="oracle_v4_causal")
    assert "Runway" not in prompt
    assert "Monthly burn" not in prompt


def test_product_prompt_states_runway_and_burn():
    prompt = build_prompt(state(mrr=30_000, cash=45_000, costs=70_000), include_burn_context=True)
    assert "Monthly burn" in prompt
    assert "Runway" in prompt


def test_churn_benchmark_line_is_omitted_when_absent():
    assert "Median monthly churn" not in build_prompt(state(), include_burn_context=True)


def test_churn_benchmark_line_appears_when_supplied():
    prompt = build_prompt(state(), include_burn_context=True, churn_benchmark_pct=3.4)
    assert "Median monthly churn for this price point: 3.4%" in prompt
    assert "this company is at 5.0%" in prompt  # the comparison, not just the benchmark


# --- scale-aware marketing physics ----------------------------------------

def test_marketing_curve_anchors_scale_with_the_company():
    from env.business_logic import marketing_curve_params

    small = marketing_curve_params(state(mrr=12_000, price=40, cac=91), "ppc")
    large = marketing_curve_params(state(mrr=200_000, price=150, cac=140), "ppc")
    _, beta_small, gamma_small = small
    _, beta_large, gamma_large = large
    assert gamma_large > gamma_small
    assert beta_large > beta_small


def _relative_response(mrr, price, cac, spend_share=0.25, draws=40):
    import random
    import statistics as st_
    from env.business_logic import compute_new_mrr
    from env.schemas import MarketingAction

    s = state(mrr=mrr, price=price, cac=cac, cash=mrr * 8, costs=mrr)
    random.seed(3)
    got = st_.mean(
        compute_new_mrr(s, MarketingAction(spend=mrr * spend_share, channel="brand"), scale_aware=True)
        for _ in range(draws)
    )
    return got / mrr


def test_marketing_response_is_size_invariant_at_equal_unit_economics():
    """With CAC/price held constant, equal relative spend buys equal relative
    growth across a 17x range of company sizes. That invariance is the whole
    point of the reparameterisation - before it, the same relative spend
    returned 33% at $12k MRR and 22% at $200k purely because of scale.
    """
    shares = [_relative_response(mrr, price, price * 2.3)
              for mrr, price in ((12_000, 40), (50_000, 80), (200_000, 150))]
    assert max(shares) - min(shares) < 0.01, f"response still varies by size: {shares}"


def test_marketing_response_still_tracks_unit_economics():
    """Size invariance must not flatten CAC away: a company paying more to
    acquire each customer should get less back per marketing dollar."""
    cheap = _relative_response(50_000, 80, 80 * 0.9)
    costly = _relative_response(50_000, 80, 80 * 2.3)
    assert cheap > costly


# --- the burn contract ----------------------------------------------------
#
# `headcount * 8000` had seven implementations across five subsystems, and the
# moment a real cost figure arrived they would have disagreed. These pin that
# there is one of them, that the fallback is exact, and that the model is told
# the truth.

def test_monthly_burn_none_reproduces_the_headcount_convention():
    """The fallback is what keeps every recorded research run valid."""
    st = state()
    st.monthly_burn = None
    st.headcount = 3
    assert business_logic.monthly_burn(st) == 3 * business_logic.SALARY_SLOT_USD


def test_real_costs_replace_the_headcount_convention():
    st = state(costs=500)                       # a founder-only company
    assert business_logic.monthly_burn(st) == 500
    # what the salary-slot encoding would have charged instead: 16x over
    assert st.headcount * business_logic.SALARY_SLOT_USD == 8_000


def test_every_burn_consumer_reads_the_same_number():
    st = state(mrr=0, cash=6_000, costs=500)
    expected = 6_000 / 500
    assert Boardroom(agents())._estimate_runway_months(st) == pytest.approx(expected)
    assert Oracle._estimate_runway_months(st) == pytest.approx(expected)
    assert f"{expected:.1f} months" in build_prompt(st, include_burn_context=True)


def test_prompt_states_the_real_burn_not_a_salary_slot():
    """The prompt used to tell the model a $500/month founder burned $8,000,
    then ask it why the company was in trouble."""
    prompt = build_prompt(state(mrr=2_500, cash=5_000, costs=500), include_burn_context=True)
    assert "- Monthly burn: 500" in prompt
    assert "8,000" not in prompt


def test_runway_clause_reads_as_a_sentence_when_cash_flow_positive():
    prompt = build_prompt(state(mrr=2_500, cash=5_000, costs=500), include_burn_context=True)
    assert "cash-flow positive - revenue covers the current burn" in prompt
    assert "positive months of cash" not in prompt


def test_hiring_adds_ongoing_payroll_once_burn_is_real():
    """Under the headcount convention a hire raised burn for free, because burn
    was derived from headcount. With a real figure it has to be added."""
    env = StartupEnv(initial_config={
        "initial_mrr": 12_000, "initial_cash": 500_000,
        "monthly_burn": 20_000, "scheduled_shocks": False,
    })
    env.reset(seed=0)
    env.step({
        "marketing": {"spend": 0.0, "channel": "ppc"},
        "hiring": {"hires": 2, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 0.0},
        "pricing": {"price_change_pct": 0.0},
    })
    assert env.state.headcount == 3
    assert env.state.monthly_burn == 20_000 + 2 * business_logic.SALARY_SLOT_USD


def test_headcount_alone_still_drives_burn_when_no_costs_are_supplied():
    """The same hire, with monthly_burn left at None: research behaviour."""
    env = StartupEnv(initial_config={
        "initial_mrr": 12_000, "initial_cash": 500_000, "scheduled_shocks": False,
    })
    env.reset(seed=0)
    before = business_logic.monthly_burn(env.state)
    env.step({
        "marketing": {"spend": 0.0, "channel": "ppc"},
        "hiring": {"hires": 2, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 0.0},
        "pricing": {"price_change_pct": 0.0},
    })
    assert env.state.monthly_burn is None
    assert business_logic.monthly_burn(env.state) == before + 2 * business_logic.SALARY_SLOT_USD


# --- research comparability -----------------------------------------------

# Fingerprint of four research-mode episodes: every StartupEnv default, 60
# months, a fixed action cycle. Recorded on the commit before monthly_burn
# existed and unchanged by it.
RESEARCH_FINGERPRINT = "aff36e6589a6c1b4c12257004a6faad31006ea17d69a8a2bb74798eddc9a7bbb"

_RESEARCH_ACTIONS = [
    {"marketing": {"spend": 12_000.0, "channel": "ppc"},
     "hiring": {"hires": 1, "cost_per_employee": 10_000.0},
     "product": {"r_and_d_spend": 9_000.0},
     "pricing": {"price_change_pct": 0.02}},
    {"marketing": {"spend": 0.0, "channel": "brand"},
     "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
     "product": {"r_and_d_spend": 0.0},
     "pricing": {"price_change_pct": 0.0}},
    {"marketing": {"spend": 30_000.0, "channel": "brand"},
     "hiring": {"hires": 2, "cost_per_employee": 12_000.0},
     "product": {"r_and_d_spend": 15_000.0},
     "pricing": {"price_change_pct": -0.05}},
]


def test_research_episodes_are_byte_identical_to_the_recorded_physics():
    """The guarantee the whole burn change rests on.

    Everything in results/ was produced by StartupEnv at its defaults. This
    pins that those defaults still produce exactly the same trajectories:
    monthly_burn=None falls back to the headcount slot, scale_aware_marketing
    and stable_cac are off, and no flag added here leaks into a research run.

    A failure means recorded results are no longer comparable to new ones. That
    is a decision to take deliberately on founder-calibration, not a test to
    update casually - if you meant it, re-record the fingerprint and say so in
    the commit.
    """
    digest = hashlib.sha256()
    for seed in (0, 1, 7, 42):
        env = StartupEnv()
        env.reset(seed=seed)
        for month in range(60):
            _, reward, terminated, truncated, info = env.step(
                _RESEARCH_ACTIONS[month % len(_RESEARCH_ACTIONS)]
            )
            snapshot = dict(info["state"])
            snapshot.pop("monthly_burn", None)   # the new field; absent before
            digest.update(json.dumps({
                "seed": seed, "m": month, "r": round(reward, 9),
                "r40": round(info["rule_of_40"], 9), "shock": info["shock_label"],
                "s": {k: (round(v, 9) if isinstance(v, float) else v)
                      for k, v in sorted(snapshot.items())},
            }, sort_keys=True).encode())
            if terminated or truncated:
                break
    assert digest.hexdigest() == RESEARCH_FINGERPRINT


def test_research_mode_never_carries_a_burn_figure():
    """Belt and braces: the fallback is reached because the field is unset,
    not because a default happened to equal the slot."""
    env = StartupEnv()
    env.reset(seed=0)
    assert env.state.monthly_burn is None
    assert env.scale_aware_marketing is False
    assert env.stable_cac is False


# --- the cash-safety resolver --------------------------------------------
#
# Boardroom._resolve_conflicts was the eighth place the engine reimplemented
# "burn = headcount x a per-head constant", and the one a grep for
# `headcount * 8000` misses because the constant is a variable - and the wrong
# variable. cost_per_employee is the ONE-TIME recruiting cost of a new hire; used
# again as a monthly salary it charged a founder-only company $10,000 a month,
# invented a shortfall, and zeroed the entire plan to cover it.

def _proposed():
    return {
        "marketing": {"spend": 500.0, "channel": "ppc"},
        "hiring": {"hires": 1, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 600.0},
        "pricing": {"price_change_pct": 0.0},
    }


def _resolved(st):
    return Boardroom(agents())._resolve_conflicts(copy.deepcopy(_proposed()), st, 0.5)


def test_a_founder_who_can_afford_the_plan_is_given_the_plan():
    """$2,500 MRR, $5,000 cash, $500/month of real costs. $500 of marketing and
    $600 of R&D are comfortably affordable; the $10,000 hire is not."""
    out = _resolved(state(mrr=2_500, cash=5_000, costs=500, price=25, cac=25, churn=0.10))
    assert out["marketing"]["spend"] == 500.0
    assert out["product"]["r_and_d_spend"] == 600.0
    assert out["hiring"]["hires"] == 0


def test_an_unaffordable_hire_is_cut_rather_than_rounded_away():
    """Rounding the cut down meant a hire costing more than the remaining
    shortfall could never be cut: floor($6,100 / $10,000) is zero, so the plan
    shed every dollar of marketing and R&D to protect it."""
    out = _resolved(state(mrr=2_500, cash=5_000, costs=500, price=25, cac=25, churn=0.10))
    assert out["hiring"]["hires"] == 0


def test_a_hire_the_company_can_afford_survives():
    out = _resolved(state(mrr=2_500, cash=200_000, costs=500, price=25, cac=25, churn=0.10))
    assert out["hiring"]["hires"] == 1
    assert out["marketing"]["spend"] == 500.0


def test_the_resolver_is_untouched_without_a_real_burn_figure():
    """Research runs reach the same code and their recorded trajectories were
    produced by the original rounding and the original base_burn expression."""
    st = state(mrr=2_500, cash=5_000, costs=500, price=25, cac=25, churn=0.10)
    st.monthly_burn = None
    st.headcount = 1
    out = _resolved(st)
    # base_burn = headcount * cost_per_employee = $10,000; nothing survives.
    assert out["marketing"]["spend"] == 0.0
    assert out["product"]["r_and_d_spend"] == 0.0
    assert out["hiring"]["hires"] == 0
