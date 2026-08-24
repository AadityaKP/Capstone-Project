import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.llm_client import DummyLLMClient
from agents.proposal_agents import CFOProposalAgent
from boardroom.boardroom import Boardroom
from boardroom.schemas import Proposal
from env.schemas import EnvState
from experiments.thesis_analysis import ABLATION_SCENARIOS, PRIMARY_SCENARIOS
from simulation_runner import BoardroomAgent, _build_agent_for_policy


def make_state() -> EnvState:
    return EnvState(
        mrr=75_000,
        cash=250_000,
        cac=500,
        ltv=5_000,
        churn_enterprise=0.02,
        churn_smb=0.05,
        churn_b2c=0.12,
        interest_rate=5.5,
        consumer_confidence=72,
        competitors=5,
        product_quality=0.78,
        price=150,
        months_elapsed=12,
        headcount=15,
        unemployment=4.2,
        innovation_factor=0.85,
        months_in_depression=0,
    )


class RecordingLLMClient:
    def __init__(self, response: str = "Role-specific rationale"):
        self.response = response
        self.calls = 0

    def complete_text(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return self.response


class SleepAgent:
    def __init__(self, agent_name: str, delay_seconds: float):
        self.agent_name = agent_name
        self.delay_seconds = delay_seconds
        self.use_llm = True

    def propose(self, state: EnvState) -> Proposal:
        time.sleep(self.delay_seconds)
        actions_by_agent = {
            "CFO": {
                "hiring": {"hires": 1, "cost_per_employee": 10_000},
                "pricing": {"price_change_pct": 0.02},
            },
            "CMO": {
                "marketing": {"spend": 15_000, "channel": "ppc"},
            },
            "CPO": {
                "product": {"r_and_d_spend": 20_000},
            },
        }
        return Proposal(
            agent=self.agent_name,
            objective=f"{self.agent_name} objective",
            actions=actions_by_agent[self.agent_name],
            expected_impact=f"{self.agent_name} static impact",
            risks=[],
            confidence=0.8,
        )


def test_boardroom_agent_accepts_injected_agents():
    injected_agents = [CFOProposalAgent(use_llm=False)]
    boardroom_agent = BoardroomAgent(oracle_mode="none", agents=injected_agents)

    assert boardroom_agent.boardroom.agents is injected_agents


def test_oracle_v3_hetero_policy_builds_boardroom_agent():
    agent = _build_agent_for_policy("oracle_v3_hetero", oracle_frequency=3)

    assert isinstance(agent, BoardroomAgent)
    assert hasattr(agent, "get_action")
    assert all(getattr(proposal_agent, "use_llm", False) for proposal_agent in agent.boardroom.agents)


def test_llm_rationale_does_not_override_expected_impact():
    agent = CFOProposalAgent(llm_client=RecordingLLMClient("Custom CFO rationale"), use_llm=True)

    proposal = agent.propose(make_state())

    assert proposal.expected_impact == "Lower burn, improved survival probability"
    assert proposal.rationale == "Custom CFO rationale"


def test_llm_rationale_cache_reuses_same_result():
    llm_client = RecordingLLMClient("Cached rationale")
    agent = CFOProposalAgent(llm_client=llm_client, use_llm=True)
    state = make_state()

    first = agent.propose(state)
    second = agent.propose(state)

    assert first.rationale == "Cached rationale"
    assert second.rationale == "Cached rationale"
    assert llm_client.calls == 1


def test_dummy_llm_client_preserves_static_expected_impact():
    agent = CFOProposalAgent(llm_client=DummyLLMClient(), use_llm=True)

    proposal = agent.propose(make_state())

    assert proposal.expected_impact == "Lower burn, improved survival probability"
    assert proposal.rationale is None


def test_boardroom_parallelizes_llm_enabled_proposals():
    boardroom = Boardroom(
        [
            SleepAgent("CFO", 0.2),
            SleepAgent("CMO", 0.2),
            SleepAgent("CPO", 0.2),
        ],
        use_oracle=False,
    )

    start = time.perf_counter()
    final_action = boardroom.decide(make_state())
    elapsed = time.perf_counter() - start

    assert elapsed < 0.45
    assert final_action["marketing"]["spend"] >= 0
    assert final_action["hiring"]["hires"] >= 0
    assert final_action["product"]["r_and_d_spend"] >= 0


def test_default_thesis_scenarios_do_not_include_hetero_policy():
    policies = [scenario["policy"] for scenario in PRIMARY_SCENARIOS + ABLATION_SCENARIOS]

    assert "oracle_v3_hetero" not in policies
