import json
import os
import sys
from copy import deepcopy

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.causal_proposal_agents import BatchedCausalProposalGenerator
from boardroom.boardroom import Boardroom
from boardroom.schemas import Proposal
from env.schemas import EnvState
from oracle.graph_store import CausalGraphStore
from oracle.oracle import Oracle
from oracle.schemas import CausalGraphContext, GraphContext
from simulation_runner import BoardroomAgent, _build_agent_for_policy, run_simulation


def make_state(month: int = 0) -> EnvState:
    return EnvState(
        mrr=80_000,
        cash=3_000_000,
        cac=500,
        ltv=5_000,
        churn_enterprise=0.01,
        churn_smb=0.02,
        churn_b2c=0.03,
        interest_rate=4.0,
        consumer_confidence=95.0,
        competitors=5,
        product_quality=0.8,
        price=100.0,
        months_elapsed=month,
        headcount=10,
        valuation_multiple=10.0,
        unemployment=4.0,
        innovation_factor=0.9,
        months_in_depression=0,
    )


def make_context(
    role: str,
    stress_node: str = "Growth_Stall",
    confidence: float = 0.8,
) -> CausalGraphContext:
    return CausalGraphContext(
        role=role,
        stress_node=stress_node,
        chain_summary=f"{stress_node} -AFFECTS-> {role}_Lever",
        root_cause_node=f"{role}_Lever",
        confidence=confidence,
        raw_triples=[[stress_node, "AFFECTS", f"{role}_Lever"]],
    )


def make_proposal(role: str, causal_confidence: float | None = None) -> Proposal:
    actions = {
        "CFO": {
            "hiring": {"hires": 0, "cost_per_employee": 10_000},
            "pricing": {"price_change_pct": 0.01},
        },
        "CMO": {"marketing": {"spend": 12_000, "channel": "ppc"}},
        "CPO": {"product": {"r_and_d_spend": 15_000}},
    }
    return Proposal(
        agent=role,
        objective=f"{role} objective",
        actions=actions[role],
        expected_impact=f"{role} impact",
        risks=[],
        confidence=0.75,
        rationale=f"{role} rationale",
        causal_confidence=causal_confidence,
    )


class FakeBriefLLM:
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return json.dumps(
            {
                "risk_level": "MEDIUM",
                "growth_outlook": "STABLE",
                "efficiency_pressure": "MEDIUM",
                "innovation_urgency": "MEDIUM",
                "macro_condition": "NEUTRAL",
                "key_risks": [],
                "key_opportunities": [],
                "recommended_focus": [],
                "confidence": 0.5,
            }
        )


class FakeGraphStore:
    enabled = True

    def __init__(self):
        self.context_calls = []
        self.outcome_writes = []

    def build_graph_context(self, shock_type=None, mrr_tier=None):
        return GraphContext()

    def query_role_causal_context(self, stress_node: str, role: str, limit: int = 3):
        self.context_calls.append((stress_node, role, limit))
        return make_context(role, stress_node)

    def write_action_outcome(self, **kwargs):
        self.outcome_writes.append(kwargs)

    def write_shock_event(self, **kwargs):
        pass

    def write_outcome(self, **kwargs):
        pass

    def write_episode(self, episode_metrics):
        pass


class FakeProposalGenerator:
    def __init__(self):
        self.calls = 0
        self.llm_calls = 0
        self.last_source = "none"
        self.last_error = None

    def propose_all(self, state, causal_contexts):
        self.calls += 1
        self.llm_calls += 1
        self.last_source = "llm"
        self.last_error = None
        return [make_proposal(role, 0.8) for role in ("CFO", "CMO", "CPO")]


class StaticAgent:
    def __init__(self, proposal: Proposal):
        self.proposal = proposal

    def propose(self, state):
        return self.proposal.model_copy(deep=True)


def test_proposal_accepts_optional_causal_confidence():
    without_confidence = make_proposal("CFO")
    with_confidence = make_proposal("CFO", 0.7)

    assert without_confidence.causal_confidence is None
    assert with_confidence.causal_confidence == 0.7


def test_oracle_causal_graph_context_is_v4_only():
    non_v4 = Oracle(
        mode="oracle_v1",
        graph_store=FakeGraphStore(),
        enable_memory_retrieval=False,
        llm=FakeBriefLLM(),
    )
    assert non_v4.get_causal_graph_context(make_state()) == {}

    graph_store = FakeGraphStore()
    oracle = Oracle(
        mode="oracle_v4_causal",
        graph_store=graph_store,
        enable_memory_retrieval=False,
        llm=FakeBriefLLM(),
    )

    contexts = oracle.get_causal_graph_context(make_state())

    assert set(contexts) == {"CFO", "CMO", "CPO"}
    assert all(context.raw_triples for context in contexts.values())
    assert graph_store.context_calls[0][0] == "Steady_State"


def test_graph_store_seeds_cash_shortage_role_edges():
    graph_store = object.__new__(CausalGraphStore)
    calls = []

    def fake_run(cypher, params):
        calls.append((cypher, params))

    graph_store._run = fake_run

    graph_store._ensure_seed_edges()

    assert calls
    edges = calls[0][1]["edges"]
    cash_shortage_edges = [
        edge for edge in edges if edge["stress_name"] == "Cash_Shortage"
    ]
    assert {edge["role"] for edge in cash_shortage_edges} == {"CFO", "CMO", "CPO"}
    assert {edge["lever_name"] for edge in cash_shortage_edges} >= {
        "Hiring_Freeze_Recommended",
        "Marketing_Spend_Cut",
        "Product_Investment_Delay",
    }
    assert len({edge["confidence"] for edge in cash_shortage_edges}) > 1


def test_batched_generator_parses_three_role_payload():
    class FakeLLM:
        def complete(self, system_prompt, user_prompt):
            return json.dumps(
                {
                    "CFO": {
                        "objective": "Preserve runway",
                        "actions": {
                            "hiring": {"hires": 1, "cost_per_employee": 10_000},
                            "pricing": {"price_change_pct": 0.02},
                        },
                        "expected_impact": "Better cash efficiency",
                        "risks": ["Slower hiring"],
                        "rationale": "Cash evidence supports restraint.",
                        "confidence": 0.7,
                        "causal_confidence": 0.9,
                    },
                    "CMO": {
                        "objective": "Improve efficient growth",
                        "actions": {"marketing": {"spend": 15_000, "channel": "ppc"}},
                        "expected_impact": "More pipeline",
                        "risks": ["CAC pressure"],
                        "rationale": "Growth evidence supports spend.",
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                    "CPO": {
                        "objective": "Reduce churn",
                        "actions": {"product": {"r_and_d_spend": 18_000}},
                        "expected_impact": "Better retention",
                        "risks": ["Burn"],
                        "rationale": "Churn evidence supports R&D.",
                        "confidence": 0.7,
                        "causal_confidence": 0.85,
                    },
                }
            )

    generator = BatchedCausalProposalGenerator(FakeLLM())
    contexts = {role: make_context(role) for role in ("CFO", "CMO", "CPO")}

    proposals = generator.propose_all(make_state(), contexts)

    assert [proposal.agent for proposal in proposals] == ["CFO", "CMO", "CPO"]
    assert proposals[0].actions.keys() == {"hiring", "pricing"}
    assert proposals[1].actions.keys() == {"marketing"}
    assert proposals[2].actions.keys() == {"product"}
    assert generator.llm_calls == 1
    assert generator.last_source == "llm"


def test_batched_generator_prompt_contains_dynamic_bounds():
    class CapturingLLM:
        def __init__(self):
            self.user_prompt = ""

        def complete(self, system_prompt, user_prompt):
            self.user_prompt = user_prompt
            return json.dumps(
                {
                    "CFO": {
                        "objective": "Preserve runway",
                        "actions": {
                            "hiring": {"hires": 1, "cost_per_employee": 10_000},
                            "pricing": {"price_change_pct": 0.0},
                        },
                        "expected_impact": "Better efficiency",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                    "CMO": {
                        "objective": "Efficient growth",
                        "actions": {"marketing": {"spend": 10_000, "channel": "ppc"}},
                        "expected_impact": "More pipeline",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                    "CPO": {
                        "objective": "Improve retention",
                        "actions": {"product": {"r_and_d_spend": 10_000}},
                        "expected_impact": "Better product",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                }
            )

    llm = CapturingLLM()
    generator = BatchedCausalProposalGenerator(llm)
    contexts = {role: make_context(role) for role in ("CFO", "CMO", "CPO")}

    generator.propose_all(make_state(), contexts)

    assert "CFO hiring.hires: 0 to 2" in llm.user_prompt
    assert "CFO hiring.cost_per_employee: 8000 to 12000" in llm.user_prompt
    assert "CFO pricing.price_change_pct: -0.05 to 0.05" in llm.user_prompt
    assert "CMO marketing.spend: 0 to 16000" in llm.user_prompt
    assert "CPO product.r_and_d_spend: 0 to 12000" in llm.user_prompt
    assert "Price increases are valid and often correct" in llm.user_prompt
    assert "LTV/CAC > 3" in llm.user_prompt
    assert '"price_change_pct": 0.02' in llm.user_prompt


def test_batched_generator_prompt_contains_persistent_stress_signal():
    generator = BatchedCausalProposalGenerator(object())
    contexts = {
        role: make_context(role, stress_node="Churn_Spike")
        for role in ("CFO", "CMO", "CPO")
    }

    prompt = generator._build_prompt(
        make_state(),
        contexts,
        stress_persistence_months=12,
        recent_action_pattern={
            "marketing": {"spend": 5000},
            "hiring": {"hires": 0},
            "product": {"r_and_d_spend": 30000},
            "pricing": {"price_change_pct": -0.01},
        },
    )

    assert "Current stress node: Churn_Spike" in prompt
    assert "Consecutive months in this stress node: 12" in prompt
    assert "Previous final action pattern:" in prompt
    assert "switch or escalate levers" in prompt


def test_batched_generator_causal_confidence_is_graph_grounded():
    class UniformConfidenceLLM:
        def complete(self, system_prompt, user_prompt):
            return json.dumps(
                {
                    "CFO": {
                        "objective": "Preserve runway",
                        "actions": {
                            "hiring": {"hires": 0, "cost_per_employee": 10_000},
                            "pricing": {"price_change_pct": 0.0},
                        },
                        "expected_impact": "Better efficiency",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.5,
                    },
                    "CMO": {
                        "objective": "Efficient growth",
                        "actions": {"marketing": {"spend": 10_000, "channel": "ppc"}},
                        "expected_impact": "More pipeline",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.5,
                    },
                    "CPO": {
                        "objective": "Improve retention",
                        "actions": {"product": {"r_and_d_spend": 10_000}},
                        "expected_impact": "Better product",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.5,
                    },
                }
            )

    generator = BatchedCausalProposalGenerator(UniformConfidenceLLM())
    contexts = {
        "CFO": make_context("CFO", stress_node="Cash_Shortage", confidence=0.82),
        "CMO": make_context("CMO", stress_node="Cash_Shortage", confidence=0.68),
        "CPO": make_context("CPO", stress_node="Cash_Shortage", confidence=0.60),
    }

    proposals = generator.propose_all(make_state(), contexts)
    confidences = {proposal.agent: proposal.causal_confidence for proposal in proposals}

    assert confidences["CFO"] > confidences["CMO"] > confidences["CPO"]
    assert confidences["CFO"] - confidences["CPO"] >= 0.05


def test_batched_generator_clamps_actions_to_dynamic_bounds():
    class ExtremeLLM:
        def complete(self, system_prompt, user_prompt):
            return json.dumps(
                {
                    "CFO": {
                        "objective": "Preserve runway",
                        "actions": {
                            "hiring": {"hires": 99, "cost_per_employee": 50_000},
                            "pricing": {"price_change_pct": 1.0},
                        },
                        "expected_impact": "Better efficiency",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                    "CMO": {
                        "objective": "Efficient growth",
                        "actions": {"marketing": {"spend": 999_999, "channel": "invalid"}},
                        "expected_impact": "More pipeline",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                    "CPO": {
                        "objective": "Improve retention",
                        "actions": {"product": {"r_and_d_spend": 999_999}},
                        "expected_impact": "Better product",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                }
            )

    generator = BatchedCausalProposalGenerator(ExtremeLLM())
    contexts = {role: make_context(role) for role in ("CFO", "CMO", "CPO")}

    proposals = generator.propose_all(make_state(), contexts)

    assert proposals[0].actions["hiring"]["hires"] == 2
    assert proposals[0].actions["hiring"]["cost_per_employee"] == 12_000
    assert proposals[0].actions["pricing"]["price_change_pct"] == 0.05
    assert proposals[1].actions["marketing"] == {"spend": 16_000, "channel": "ppc"}
    assert proposals[2].actions["product"]["r_and_d_spend"] == 12_000


def test_action_bounds_churn_spike_widens_price_max_at_sp10():
    b = BatchedCausalProposalGenerator._action_bounds(make_state(), "Churn_Spike", 10)
    assert b["price_max"] == pytest.approx(0.08)


def test_action_bounds_churn_spike_widens_price_max_at_sp30():
    b = BatchedCausalProposalGenerator._action_bounds(make_state(), "Churn_Spike", 30)
    assert b["price_max"] == pytest.approx(0.12)


def test_action_bounds_non_churn_stress_no_widening():
    for stress_node, sp in [("Steady_State", 40), ("Cash_Shortage", 15)]:
        b = BatchedCausalProposalGenerator._action_bounds(make_state(), stress_node, sp)
        assert b["price_max"] == pytest.approx(0.05), f"Expected 0.05 for {stress_node} sp={sp}, got {b['price_max']}"


def test_action_bounds_churn_spike_sp0_baseline():
    b = BatchedCausalProposalGenerator._action_bounds(make_state(), "Churn_Spike", 0)
    assert b["price_max"] == pytest.approx(0.05)


# --------------------------------------------------------------------------- #
# CFO persistence escalation in _normalize_actions
# --------------------------------------------------------------------------- #

def _cfo_actions(price_change_pct: float) -> dict:
    return {
        "hiring": {"hires": 0, "cost_per_employee": 10_000},
        "pricing": {"price_change_pct": price_change_pct},
    }


def _normalize_cfo(price_change_pct: float, stress_node: str, sp: int) -> float:
    gen = BatchedCausalProposalGenerator.__new__(BatchedCausalProposalGenerator)
    ctx = make_context("CFO", stress_node=stress_node)
    result = gen._normalize_actions(
        "CFO", _cfo_actions(price_change_pct), make_state(),
        context=ctx, stress_persistence_months=sp,
    )
    return result["pricing"]["price_change_pct"]


def test_cfo_persistence_no_override_below_sp10():
    assert _normalize_cfo(-0.01, "Churn_Spike", 9) == pytest.approx(-0.01)


def test_cfo_persistence_escalates_at_sp10():
    assert _normalize_cfo(-0.01, "Churn_Spike", 10) == pytest.approx(0.02)


def test_cfo_persistence_escalates_at_sp30():
    assert _normalize_cfo(-0.01, "Churn_Spike", 30) == pytest.approx(0.04)


def test_cfo_persistence_respects_positive_llm_choice():
    # LLM already chose positive price — escalation must not override
    assert _normalize_cfo(0.03, "Churn_Spike", 10) == pytest.approx(0.03)


def test_cfo_persistence_no_override_steady_state():
    assert _normalize_cfo(-0.01, "Steady_State", 40) == pytest.approx(-0.01)


def test_cfo_persistence_no_override_cash_shortage():
    assert _normalize_cfo(-0.01, "Cash_Shortage", 40) == pytest.approx(-0.01)


def test_batched_generator_nudges_zero_pricing_when_demand_is_healthy():
    class ZeroPricingLLM:
        def complete(self, system_prompt, user_prompt):
            return json.dumps(
                {
                    "CFO": {
                        "objective": "Preserve runway",
                        "actions": {
                            "hiring": {"hires": 0, "cost_per_employee": 10_000},
                            "pricing": {"price_change_pct": 0.0},
                        },
                        "expected_impact": "Better efficiency",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                    "CMO": {
                        "objective": "Efficient growth",
                        "actions": {"marketing": {"spend": 10_000, "channel": "ppc"}},
                        "expected_impact": "More pipeline",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                    "CPO": {
                        "objective": "Improve retention",
                        "actions": {"product": {"r_and_d_spend": 10_000}},
                        "expected_impact": "Better product",
                        "risks": [],
                        "confidence": 0.7,
                        "causal_confidence": 0.8,
                    },
                }
            )

    generator = BatchedCausalProposalGenerator(ZeroPricingLLM())
    contexts = {role: make_context(role) for role in ("CFO", "CMO", "CPO")}

    proposals = generator.propose_all(make_state(), contexts)

    assert proposals[0].actions["pricing"]["price_change_pct"] == pytest.approx(0.01)


def test_batched_generator_falls_back_on_invalid_output():
    class EmptyLLM:
        def complete(self, system_prompt, user_prompt):
            return ""

    generator = BatchedCausalProposalGenerator(EmptyLLM())
    contexts = {role: make_context(role) for role in ("CFO", "CMO", "CPO")}

    proposals = generator.propose_all(make_state(), contexts)

    assert [proposal.agent for proposal in proposals] == ["CFO", "CMO", "CPO"]
    assert proposals[0].causal_confidence is None
    assert generator.llm_calls == 1
    assert generator.last_source == "fallback_llm_empty"


def test_boardroom_refresh_cache_and_reuse_batched_proposals():
    graph_store = FakeGraphStore()
    oracle = Oracle(
        mode="oracle_v4_causal",
        graph_store=graph_store,
        enable_memory_retrieval=False,
        llm=FakeBriefLLM(),
    )
    generator = FakeProposalGenerator()
    boardroom = Boardroom(
        agents=[],
        use_oracle=True,
        oracle_mode="oracle_v4_causal",
        oracle_frequency=5,
        oracle_instance=oracle,
        proposal_generator=generator,
    )
    boardroom.start_episode(episode_seed=1)

    boardroom.decide(make_state(0))
    assert generator.calls == 1
    assert boardroom.last_decision_trace["proposal_source"] == "llm"
    assert boardroom.last_decision_trace["stress_persistence_months"] == 1

    boardroom.decide(make_state(1))
    assert generator.calls == 1
    assert boardroom.last_decision_trace["proposal_source"] == "reuse"
    assert boardroom.last_decision_trace["stress_persistence_months"] == 2

    boardroom.decide(make_state(5))
    assert generator.calls == 1
    assert boardroom.last_decision_trace["proposal_source"] == "cache_hit"
    assert boardroom.last_decision_trace["stress_persistence_months"] == 3
    assert boardroom.get_episode_stats()["proposal_llm_calls"] == 1
    assert boardroom.get_episode_stats()["proposal_cache_hits"] == 1


def test_boardroom_passes_stress_persistence_to_compatible_generator():
    class PersistenceAwareGenerator(FakeProposalGenerator):
        def __init__(self):
            super().__init__()
            self.received = []

        def propose_all(
            self,
            state,
            causal_contexts,
            stress_persistence_months=None,
            recent_action_pattern=None,
        ):
            self.received.append(
                {
                    "stress_persistence_months": stress_persistence_months,
                    "recent_action_pattern": deepcopy(recent_action_pattern),
                }
            )
            return super().propose_all(state, causal_contexts)

    generator = PersistenceAwareGenerator()
    boardroom = Boardroom([], use_oracle=False, proposal_generator=generator)
    previous_action = {
        "marketing": {"spend": 5000, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000},
        "pricing": {"price_change_pct": 0.02},
        "product": {"r_and_d_spend": 30_000},
    }

    boardroom._call_proposal_generator(
        make_state(),
        {role: make_context(role) for role in ("CFO", "CMO", "CPO")},
        stress_persistence_months=7,
        recent_action_pattern=previous_action,
    )

    assert generator.received == [
        {
            "stress_persistence_months": 7,
            "recent_action_pattern": previous_action,
        }
    ]


def test_causal_confidence_modulates_only_v4_scores():
    proposal = make_proposal("CFO", 1.0)
    base_score = 0.5

    non_v4 = Boardroom([StaticAgent(proposal)], use_oracle=False)
    assert non_v4._apply_causal_confidence_score(base_score, proposal) == base_score

    oracle = Oracle(
        mode="oracle_v4_causal",
        graph_store=FakeGraphStore(),
        enable_memory_retrieval=False,
        llm=FakeBriefLLM(),
    )
    v4 = Boardroom(
        [StaticAgent(proposal)],
        use_oracle=True,
        oracle_mode="oracle_v4_causal",
        oracle_instance=oracle,
    )
    assert v4._apply_causal_confidence_score(base_score, proposal) == pytest.approx(0.575)


def test_v4_causal_rd_cap_applies_after_dynamic_minimums_only_for_v4():
    state = make_state().model_copy(
        update={
            "mrr": 1_000_000,
            "cash": 100_000,
            "innovation_factor": 0.0,
        }
    )
    action = {
        "marketing": {"spend": 0.0, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000},
        "pricing": {"price_change_pct": 0.0},
        "product": {"r_and_d_spend": 500_000},
    }
    oracle = Oracle(
        mode="oracle_v4_causal",
        graph_store=FakeGraphStore(),
        enable_memory_retrieval=False,
        llm=FakeBriefLLM(),
    )
    v4 = Boardroom(
        [],
        use_oracle=True,
        oracle_mode="oracle_v4_causal",
        oracle_instance=oracle,
    )

    bounded = v4._apply_sanity_bounds(deepcopy(action), state)
    assert bounded["product"]["r_and_d_spend"] == 30_000

    bounded["product"]["r_and_d_spend"] = 1
    bounded = v4._apply_dynamic_minimums(bounded, state, innov_score=1.0)
    assert bounded["product"]["r_and_d_spend"] > 30_000
    bounded = v4._apply_v4_causal_rd_cap(bounded, state)
    assert bounded["product"]["r_and_d_spend"] == 30_000

    non_v4 = Boardroom([], use_oracle=False)
    unchanged = non_v4._apply_sanity_bounds(deepcopy(action), state)
    assert unchanged["product"]["r_and_d_spend"] == 500_000


def test_v4_cash_shortage_relaxes_rd_protection_but_other_stress_keeps_it():
    state = make_state().model_copy(update={"cash": 50_000, "headcount": 5})
    action = {
        "marketing": {"spend": 0.0, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000},
        "pricing": {"price_change_pct": 0.0},
        "product": {"r_and_d_spend": 100_000},
    }
    oracle = Oracle(
        mode="oracle_v4_causal",
        graph_store=FakeGraphStore(),
        enable_memory_retrieval=False,
        llm=FakeBriefLLM(),
    )
    v4 = Boardroom(
        [],
        use_oracle=True,
        oracle_mode="oracle_v4_causal",
        oracle_instance=oracle,
    )

    cash_shortage = v4._resolve_conflicts(
        deepcopy(action),
        state,
        innov_score=0.9,
        stress_node="Cash_Shortage",
    )
    assert cash_shortage["product"]["r_and_d_spend"] == 0

    steady_state = v4._resolve_conflicts(
        deepcopy(action),
        state,
        innov_score=0.9,
        stress_node="Steady_State",
    )
    assert steady_state["product"]["r_and_d_spend"] == 80_000


def test_oracle_v4_causal_hetero_policy_builds_boardroom_agent():
    oracle = Oracle(
        mode="oracle_v4_causal",
        graph_store=FakeGraphStore(),
        enable_memory_retrieval=False,
        llm=FakeBriefLLM(),
    )
    generator = FakeProposalGenerator()

    agent = _build_agent_for_policy(
        "oracle_v4_causal_hetero",
        oracle_frequency=5,
        oracle_overrides={
            "oracle_instance": oracle,
            "proposal_generator": generator,
        },
    )

    assert isinstance(agent, BoardroomAgent)
    assert agent.boardroom.oracle_mode == "oracle_v4_causal"
    assert agent.boardroom.proposal_generator is generator


def test_one_episode_fake_llm_smoke_run_records_traces_and_stats():
    graph_store = FakeGraphStore()
    oracle = Oracle(
        mode="oracle_v4_causal",
        graph_store=graph_store,
        enable_memory_retrieval=False,
        llm=FakeBriefLLM(),
    )
    generator = FakeProposalGenerator()

    df, monthly_trace = run_simulation(
        policy="oracle_v4_causal_hetero",
        num_episodes=1,
        seed_start=0,
        oracle_frequency=5,
        oracle_overrides={
            "oracle_instance": oracle,
            "proposal_generator": generator,
        },
        return_monthly_trace=True,
    )

    assert len(df) == 1
    assert "proposal_llm_calls" in df.columns
    assert df.loc[0, "proposal_llm_calls"] >= 1
    assert monthly_trace
    assert "proposals" in monthly_trace[0]["decision_trace"]
    assert graph_store.outcome_writes
