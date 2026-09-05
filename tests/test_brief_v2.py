"""Round-2 brief v2 tests (BRIEF_V2_SPEC.md; plan session S2).

Asserts: flags default off leave the prompt byte-identical (no v2 text);
runway uses the engine's real burn; guardrail floors only ever raise
severity; modifier_bound is a no-op under the legacy corridor and binds under
scale_aware; normalized memory documents/queries carry no absolute dollar
figure; runway_estimator="burn" equals the legacy estimator on this branch.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from env import business_logic
from env.schemas import EnvState
from boardroom.boardroom import Boardroom
from agents.proposal_agents import (CFOProposalAgent, CMOProposalAgent,
                                    CPOProposalAgent)
from oracle.levels import apply_brief_floors, compute_level_assessment
from oracle.memory import (build_memory_query_normalized,
                           format_memory_document_normalized)
from oracle.context import snapshot_state
from oracle.prompt_builder import build_prompt
from oracle.schemas import (ExpectedOutcome, GraphContext, OracleBrief,
                            TrendContext)


def make_state(**over):
    d = dict(mrr=50_000.0, cash=600_000.0, cac=100.0, ltv=1_500.0,
             churn_enterprise=0.01, churn_smb=0.03, churn_b2c=0.05,
             interest_rate=3.0, consumer_confidence=100.0, competitors=5,
             product_quality=0.5, price=50.0, months_elapsed=24, headcount=4,
             valuation_multiple=10.0, unemployment=4.0, innovation_factor=1.0,
             months_in_depression=0)
    d.update(over)
    return EnvState(**d)


# ---------------------------------------------------------------- prompt

def test_default_prompt_has_no_v2_text():
    p_default = build_prompt(make_state())
    p_v1 = build_prompt(make_state(), brief_version="v1")
    assert p_default == p_v1
    assert "LEVEL ASSESSMENT" not in p_default
    assert "at least as severe" not in p_default


def test_v2_prompt_contains_level_block_and_instruction():
    p = build_prompt(make_state(), brief_version="v2")
    assert "--- LEVEL ASSESSMENT" in p
    assert "risk_level must be at least as severe" in p
    assert "Runway:" in p and "LTV:CAC:" in p and "Macro regime:" in p


def test_runway_uses_real_burn_not_headcount_slots():
    st = make_state(monthly_burn=100_000.0, cash=600_000.0, headcount=4)
    levels = compute_level_assessment(st)
    assert levels["runway_months"] == pytest.approx(6.0)
    assert levels["runway_band"] == "HIGH"
    # headcount*8000 would have said 18.75 months (MEDIUM)


# ---------------------------------------------------------------- floors

@pytest.mark.parametrize("risk_in", ["LOW", "MEDIUM", "HIGH", "CRITICAL"])
def test_floors_only_raise_severity(risk_in):
    sev = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    for cash in (100_000.0, 600_000.0, 3_000_000.0):
        for ltv in (50.0, 250.0, 1_500.0):
            st = make_state(cash=cash, ltv=ltv, monthly_burn=25_000.0)
            levels = compute_level_assessment(st)
            brief = OracleBrief(risk_level=risk_in)
            out, log = apply_brief_floors(brief, levels)
            out_risk = out.risk_level.value if hasattr(out.risk_level, "value") else out.risk_level
            assert sev[out_risk] >= sev[risk_in], "floor lowered severity"
            if sev[out_risk] > sev[risk_in]:
                assert log, "override not logged"


def test_macro_floor_only_forces_recession():
    st = make_state(consumer_confidence=45.0)  # computed regime RECESSION
    levels = compute_level_assessment(st)
    out, log = apply_brief_floors(OracleBrief(macro_condition="EXPANSION"), levels)
    macro = out.macro_condition.value if hasattr(out.macro_condition, "value") else out.macro_condition
    assert macro == "RECESSION" and any("macro" in x for x in log)
    st2 = make_state(consumer_confidence=120.0, unemployment=4.0)  # EXPANSION regime
    out2, log2 = apply_brief_floors(OracleBrief(macro_condition="NEUTRAL"),
                                    compute_level_assessment(st2))
    macro2 = out2.macro_condition.value if hasattr(out2.macro_condition, "value") else out2.macro_condition
    assert macro2 == "NEUTRAL" and not log2  # non-recession regime never overrides


# ---------------------------------------------------------------- modifier bound

class _StubOracle:
    """Minimal oracle: fixed aggressive brief (LOW/ACCELERATING/LOW -> 1.638x)."""
    last_floor_applied: list = []

    def __init__(self):
        self.latest_snapshot = None

    def start_episode(self, episode_seed=None):
        pass

    def observe_state(self, state):
        pass

    def get_context(self, state, active_shock_label=None):
        return TrendContext(), [], 0, GraphContext()

    def build_cache_key(self, state, trend_context=None, memories=None):
        return ("stub", str(state.months_elapsed))

    def generate_brief(self, state, trend_context=None, memories=None,
                       shock_label=None):
        return OracleBrief(risk_level="LOW", growth_outlook="ACCELERATING",
                           efficiency_pressure="LOW", innovation_urgency="LOW")


def _board(corridor, modifier_bound):
    return Boardroom(
        [CFOProposalAgent(corridor=corridor), CMOProposalAgent(corridor=corridor),
         CPOProposalAgent(corridor=corridor)],
        use_oracle=True, oracle_mode="oracle_v1", oracle_instance=_StubOracle(),
        corridor=corridor, modifier_bound=modifier_bound)


def _post_modifier_mkt(board, state):
    board.start_episode(0)
    board.decide(state)
    return board.last_decision_trace["post_modifier_action"]["marketing"]["spend"]


def test_modifier_bound_noop_under_legacy_corridor():
    st = make_state(ltv=500.0, cac=100.0)  # ratio 5 -> top CMO tier
    unbounded = _post_modifier_mkt(_board("legacy", "none"), st)
    bounded = _post_modifier_mkt(_board("legacy", "tier"), st)
    assert unbounded == bounded  # no-op by construction under legacy


def test_modifier_bound_binds_under_scale_aware():
    st = make_state(ltv=500.0, cac=100.0)  # top tier: 40% MRR = 20,000
    unbounded = _post_modifier_mkt(_board("scale_aware", "none"), st)
    bounded = _post_modifier_mkt(_board("scale_aware", "tier"), st)
    assert unbounded == pytest.approx(20_000 * 1.638, rel=1e-9)
    assert bounded == pytest.approx(0.40 * st.mrr, rel=1e-9)
    assert bounded < unbounded


# ---------------------------------------------------------------- runway flag

def test_runway_estimator_burn_equals_legacy_on_this_branch():
    st = make_state(monthly_burn=40_000.0)
    legacy = Boardroom([], runway_estimator="legacy")
    burn = Boardroom([], runway_estimator="burn")
    assert legacy._estimate_runway_months(st) == burn._estimate_runway_months(st)
    assert legacy._estimate_runway_months(st) == pytest.approx(
        st.cash / business_logic.monthly_burn(st))


# ---------------------------------------------------------------- normalized memory

def test_normalized_memory_has_no_absolute_dollars():
    st = make_state(mrr=3_456_789.0, cash=987_654_321.0, monthly_burn=1_234_567.0)
    snap = snapshot_state(st, global_month=10, episode_seed=1,
                          episode_start_mrr=1_000_000.0, prev_mrr=3_300_000.0)
    doc = format_memory_document_normalized(snap, TrendContext(),
                                            ExpectedOutcome.GROWTH)
    query = build_memory_query_normalized(st, TrendContext(),
                                          episode_start_mrr=1_000_000.0)
    for text in (doc, query):
        digits = re.findall(r"\d[\d,]{4,}", text)
        assert not digits, f"absolute figure leaked: {digits} in {text!r}"
    assert "3.5x episode start" in doc
    assert snap.runway_band is not None


def test_legacy_snapshot_unchanged():
    st = make_state()
    snap = snapshot_state(st, global_month=5, episode_seed=2)
    assert snap.mrr_rel_start is None and snap.runway_band is None
