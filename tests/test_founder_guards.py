"""Tests for the product guards added on the founder path.

Every guard here is opt-in: research runs must reproduce byte-identically with
the flags at their defaults, and each test pins both sides of that.
"""

import copy

import pytest

from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent, merge_actions
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
from boardroom.boardroom import Boardroom
from env.schemas import EnvState
from oracle.parser import parse_llm_response
from oracle.prompt_builder import build_prompt


def state(mrr=12_000, cash=90_000, costs=24_000, price=40, cac=91, churn=0.05):
    return EnvState(
        mrr=mrr, cash=cash, cac=cac, ltv=price / max(churn, 1e-6),
        churn_enterprise=churn, churn_smb=churn, churn_b2c=churn,
        interest_rate=3, consumer_confidence=100, competitors=5,
        product_quality=0.5, price=price, months_elapsed=12,
        headcount=max(1, round(costs / 8000)),
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
