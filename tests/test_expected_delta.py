"""Phase 1A of docs/oefa_loop_plan.md: the agents make a prediction, and they
see their own track record.

  - expected_delta is on every proposal in the product path, and absent on
    every research arm (which must keep reproducing byte-identically);
  - the prediction is deterministic and leaves the process-global RNG alone;
  - an agent whose last prediction missed on direction changes its proposal
    and says so.
"""

from __future__ import annotations

import random

import pytest

from agents.causal_proposal_agents import BatchedCausalProposalGenerator
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
from boardroom.boardroom import Boardroom
from boardroom import expectation as ex
from env.schemas import EnvState
from oracle.schemas import CausalGraphContext


def state(mrr=30_000, cash=200_000, costs=40_000, churn=0.05, month=8):
    return EnvState(
        mrr=mrr, cash=cash, cac=90, ltv=85 / churn,
        churn_enterprise=churn, churn_smb=churn, churn_b2c=churn,
        interest_rate=3, consumer_confidence=100, competitors=5,
        product_quality=0.5, price=85, months_elapsed=month,
        headcount=4, monthly_burn=costs,
    )


ACTION = {
    "marketing": {"spend": 6_000.0, "channel": "ppc"},
    "hiring": {"hires": 0, "cost_per_employee": 10_000},
    "product": {"r_and_d_spend": 9_000.0},
    "pricing": {"price_change_pct": 0.0},
}

FOUNDER_ENV = {"max_months": 10_000, "scheduled_shocks": False,
               "scale_aware_marketing": True, "scale_aware_rnd": True}


def predictor():
    return ex.make_expectation(env_kwargs=FOUNDER_ENV, horizon_months=2, n_seeds=4)


# --------------------------------------------------------------------------
# the prediction itself
# --------------------------------------------------------------------------

def test_prediction_has_every_kpi_a_horizon_and_a_band():
    delta = predictor()(state(), ACTION)
    for key in ex.KPI_KEYS:
        assert key in delta
    assert delta["horizon_months"] == 2
    assert delta["basis"] == "simulated"
    assert set(delta["band"]) == {"mrr", "cash", "churn_pct"}
    assert delta["band"]["mrr"]["p25"] <= delta["band"]["mrr"]["p75"]
    assert isinstance(delta["mrr_pct"], float)


def test_prediction_hires_once_across_its_horizon(monkeypatch):
    """Same rule as the what-if: a hire is a one-off, spend repeats."""
    from env.startup_env import StartupEnv

    seen = []
    original = StartupEnv.step

    def spy(self, action):
        seen.append(int(action["hiring"]["hires"]))
        return original(self, action)

    monkeypatch.setattr(StartupEnv, "step", spy)
    hiring = {**ACTION, "hiring": {"hires": 1, "cost_per_employee": 10_000}}
    ex.predict_expected_delta(state(cash=600_000), hiring, horizon_months=3, n_seeds=2, env_kwargs=FOUNDER_ENV)
    assert seen == [1, 0, 0, 1, 0, 0]


def test_prediction_is_deterministic():
    assert predictor()(state(), ACTION) == predictor()(state(), ACTION)


def test_prediction_leaves_the_global_rng_untouched():
    """A cycle steps its own environment on the global stream; a prediction
    made mid-cycle must not move it."""
    random.seed(7)
    before = [random.random() for _ in range(3)]
    random.seed(7)
    predictor()(state(), ACTION)
    after = [random.random() for _ in range(3)]
    assert before == after


def test_kpi_delta_sign_conventions():
    before = state(mrr=30_000, cash=200_000, churn=0.05)
    after = state(mrr=33_000, cash=190_000, churn=0.045)
    delta = ex.kpi_delta_between(before, after)
    assert delta["mrr_pct"] == pytest.approx(10.0)
    assert delta["cash_pct"] == pytest.approx(-5.0)
    assert delta["churn_pp"] == pytest.approx(-0.5)


def test_runway_delta_is_none_when_not_burning():
    burning = state(mrr=30_000, costs=40_000)
    profitable = state(mrr=50_000, costs=40_000)
    assert ex.kpi_delta_between(burning, profitable)["runway_months"] is None


def test_prediction_error_scores_direction_and_tolerance():
    expected = {"mrr_pct": 4.0, "cash_pct": -6.0, "churn_pp": -0.3, "runway_months": -0.4, "horizon_months": 2}
    realized = {"mrr_pct": -1.5, "cash_pct": -6.5, "churn_pp": -0.25, "runway_months": None}
    error = ex.prediction_error(expected, realized)
    assert error["mrr_pct"]["error"] == pytest.approx(-5.5)
    assert error["mrr_pct"]["sign_agrees"] is False
    assert error["cash_pct"]["within_tolerance"] is True
    assert error["churn_pp"]["sign_agrees"] is True
    assert error["runway_months"] is None
    assert error["summary"] == {"kpis_scored": 3, "within_tolerance": 2, "sign_agrees": 2, "horizon_months": 2}


def test_founder_marketing_curve_flag_switches_the_prediction(monkeypatch):
    """Decision 9 (open): the shipped curve assumes a saturation rate of 0.20;
    v2 uses the rate physics_v2 fitted. The flag must reach the physics, and
    the fitted rate must predict less growth from the same marketing spend."""
    monkeypatch.setenv("SIM_PROFILE", "founder")
    from backend import sim_profile

    monkeypatch.setattr(sim_profile, "FOUNDER_MARKETING_CURVE", "scale_aware")
    shipped = sim_profile.get_env_kwargs()
    assert shipped["marketing_curve"] == "scale_aware" and shipped["scale_aware_marketing"] is True
    monkeypatch.setattr(sim_profile, "FOUNDER_MARKETING_CURVE", "v2")
    fitted = sim_profile.get_env_kwargs()
    assert fitted["marketing_curve"] == "v2"

    assumed = ex.predict_expected_delta(state(), ACTION, env_kwargs=shipped)
    calibrated = ex.predict_expected_delta(state(), ACTION, env_kwargs=fitted)
    assert calibrated["mrr_pct"] < assumed["mrr_pct"]

    monkeypatch.setattr(sim_profile, "FOUNDER_MARKETING_CURVE", "nope")
    with pytest.raises(ValueError):
        sim_profile.get_env_kwargs()


# --------------------------------------------------------------------------
# present in the product path, absent on research arms
# --------------------------------------------------------------------------

def test_every_product_proposal_carries_expected_delta_and_the_trace_scores_the_final_action():
    board = Boardroom(
        [CMOProposalAgent(expectation=predictor()), CFOProposalAgent(expectation=predictor()),
         CPOProposalAgent(expectation=predictor())],
        use_oracle=False, expectation=predictor(),
    )
    board.decide(state())
    trace = board.get_last_decision_trace()
    assert all(p["expected_delta"] is not None for p in trace["proposals"])
    assert trace["expected_delta"] is not None
    assert trace["expected_delta"]["horizon_months"] == 2


def test_research_arms_carry_no_prediction():
    board = Boardroom([CMOProposalAgent(), CFOProposalAgent(), CPOProposalAgent()], use_oracle=False)
    action = board.decide(state())
    trace = board.get_last_decision_trace()
    assert all(p["expected_delta"] is None for p in trace["proposals"])
    assert all(p["adaptation"] is None for p in trace["proposals"])
    assert trace["expected_delta"] is None
    assert trace["track_record"] is None
    # And the action is the one the heuristics always produced.
    assert action["marketing"]["spend"] == pytest.approx(
        Boardroom([CMOProposalAgent(), CFOProposalAgent(), CPOProposalAgent()], use_oracle=False).decide(state())["marketing"]["spend"]
    )


# --------------------------------------------------------------------------
# the track record changes the proposal
# --------------------------------------------------------------------------

def record_with(kpi, expected, realized):
    exp = {"mrr_pct": 1.0, "cash_pct": -3.0, "churn_pp": -0.1, "runway_months": None, "horizon_months": 2}
    got = {"mrr_pct": 1.0, "cash_pct": -3.0, "churn_pp": -0.1, "runway_months": None}
    exp[kpi] = expected
    got[kpi] = realized
    return ex.build_track_record(ACTION, exp, got, month_label="last month")


def test_cmo_pulls_marketing_back_after_a_wrong_direction_call():
    cold = CMOProposalAgent().propose(state())
    agent = CMOProposalAgent()
    agent.set_track_record(record_with("mrr_pct", expected=5.0, realized=-3.0))
    warm = agent.propose(state())
    assert warm.actions["marketing"]["spend"] == pytest.approx(cold.actions["marketing"]["spend"] * 0.75)
    assert "expected MRR +5.0%" in warm.adaptation
    assert "held back 25%" in warm.adaptation


def test_a_size_miss_in_the_right_direction_changes_nothing():
    agent = CMOProposalAgent()
    agent.set_track_record(record_with("mrr_pct", expected=5.0, realized=2.0))
    cold = CMOProposalAgent().propose(state())
    warm = agent.propose(state())
    assert warm.actions == cold.actions
    assert warm.adaptation is None


def test_cpo_and_cfo_have_their_own_kpis():
    cpo = CPOProposalAgent()
    cpo.set_track_record(record_with("churn_pp", expected=-0.4, realized=0.6))
    assert cpo.propose(state()).adaptation.startswith("Last month's plan expected churn -0.4pp")

    cfo = CFOProposalAgent()
    cfo.set_track_record(record_with("cash_pct", expected=2.0, realized=-9.0))
    rich = state(cash=2_000_000, mrr=60_000, costs=40_000)  # runway > 24 months, so the CFO would hire
    assert CFOProposalAgent().propose(rich).actions["hiring"]["hires"] == 1
    warm = cfo.propose(rich)
    assert warm.actions["hiring"]["hires"] == 0
    assert "hiring waits" in warm.adaptation


def test_boardroom_fans_the_record_out_and_records_it_in_the_trace():
    agents = [CMOProposalAgent(), CFOProposalAgent(), CPOProposalAgent()]
    board = Boardroom(agents, use_oracle=False)
    record = record_with("mrr_pct", expected=5.0, realized=-3.0)
    board.set_track_record(record)
    board.decide(state())
    trace = board.get_last_decision_trace()
    assert trace["track_record"]["prediction_error"]["mrr_pct"]["sign_agrees"] is False
    cmo = next(p for p in trace["proposals"] if p["agent"] == "CMO")
    assert cmo["adaptation"] is not None
    board.set_track_record(None)
    assert all(a.track_record is None for a in agents)


def test_causal_prompt_shows_the_track_record_only_when_there_is_one():
    contexts = {
        role: CausalGraphContext(role=role, stress_node="Churn_Spike", chain_summary="x -MAY_CAUSE-> y",
                                 confidence=0.6, raw_triples=[["x", "MAY_CAUSE", "y"]])
        for role in ("CFO", "CMO", "CPO")
    }
    generator = BatchedCausalProposalGenerator(llm_client=None)
    cold = generator._build_prompt(state(), contexts)
    assert "Board track record" not in cold
    generator.set_track_record(record_with("mrr_pct", expected=5.0, realized=-3.0))
    warm = generator._build_prompt(state(), contexts)
    assert "Board track record" in warm
    assert "WRONG DIRECTION" in warm
    assert "predicted +5.00, actual -3.00" in warm


def test_causal_fallback_proposals_also_carry_the_prediction():
    class Down:
        def complete(self, *_):
            raise ConnectionError("ollama down")

    generator = BatchedCausalProposalGenerator(llm_client=Down(), expectation=predictor())
    contexts = {
        role: CausalGraphContext(role=role, stress_node="Steady_State", chain_summary="s",
                                 confidence=0.5, raw_triples=[["a", "MAY_CAUSE", "b"]])
        for role in ("CFO", "CMO", "CPO")
    }
    proposals = generator.propose_all(state(), contexts)
    assert generator.last_source == "fallback_llm_error"
    assert all(p.expected_delta is not None for p in proposals)
