import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from boardroom.boardroom import Boardroom
from boardroom.schemas import Proposal
from env.schemas import EnvState
from oracle.oracle import Oracle
from oracle.schemas import (
    GrowthOutlook,
    MacroCondition,
    OracleBrief,
    RetrievedMemoryCandidate,
    TrendContext,
    TrendDirection,
    UrgencyLevel,
)
import simulation_runner


class FakeProposalAgent:
    def __init__(self, agent: str, actions: dict):
        self.agent = agent
        self.actions = actions

    def propose(self, state: EnvState) -> Proposal:
        return Proposal(
            agent=self.agent,
            objective=f"{self.agent} objective",
            actions=self.actions,
            expected_impact="steady",
            risks=[],
            confidence=0.8,
        )


class FakeOracle:
    def __init__(self, mode: str = "oracle_v1", key_builder=None):
        self.mode = mode
        self.key_builder = key_builder or (lambda state, trend_context, memories: ("stable",))
        self.trend_context = TrendContext(
            mrr_trend=TrendDirection.FLAT,
            innovation_trend=TrendDirection.FLAT,
            churn_trend=TrendDirection.FLAT,
            history_points=5,
        )
        self.memories = []
        self.generate_calls = []

    def start_episode(self, episode_seed=None):
        return None

    def observe_state(self, state: EnvState, episode_seed=None):
        return None

    def get_context(self, state: EnvState):
        return self.trend_context, self.memories, state.months_elapsed

    def build_cache_key(self, state: EnvState, trend_context=None, memories=None):
        return self.key_builder(state, trend_context, memories)

    def generate_brief(self, state: EnvState, trend_context=None, memories=None):
        self.generate_calls.append((state.months_elapsed, self.build_cache_key(state, trend_context, memories)))
        return make_brief()


class StaticLLM:
    def complete(self, system_prompt, user_prompt):
        return (
            '{"risk_level":"MEDIUM","growth_outlook":"STABLE","efficiency_pressure":"MEDIUM",'
            '"innovation_urgency":"MEDIUM","macro_condition":"NEUTRAL","confidence":0.8}'
        )


def make_brief():
    return OracleBrief(
        risk_level=UrgencyLevel.MEDIUM,
        growth_outlook=GrowthOutlook.STABLE,
        efficiency_pressure=UrgencyLevel.MEDIUM,
        innovation_urgency=UrgencyLevel.MEDIUM,
        macro_condition=MacroCondition.NEUTRAL,
        confidence=0.8,
    )


def make_state(
    months_elapsed: int,
    mrr: float = 100_000.0,
    cash: float = 1_000_000.0,
    churn_enterprise: float = 0.02,
    churn_smb: float = 0.03,
    churn_b2c: float = 0.04,
    consumer_confidence: float = 100.0,
    unemployment: float = 4.0,
    innovation_factor: float = 0.6,
    price: float = 50.0,
    headcount: int = 5,
) -> EnvState:
    return EnvState(
        mrr=mrr,
        cash=cash,
        cac=100.0,
        ltv=700.0,
        churn_enterprise=churn_enterprise,
        churn_smb=churn_smb,
        churn_b2c=churn_b2c,
        interest_rate=3.0,
        consumer_confidence=consumer_confidence,
        competitors=5,
        product_quality=0.5,
        price=price,
        months_elapsed=months_elapsed,
        headcount=headcount,
        valuation_multiple=10.0,
        unemployment=unemployment,
        innovation_factor=innovation_factor,
        months_in_depression=0,
    )


def make_boardroom(fake_oracle: FakeOracle, oracle_cache_max_size: int = 5000) -> Boardroom:
    agents = [
        FakeProposalAgent("CFO", {"hiring": {"hires": 0, "cost_per_employee": 10_000}, "pricing": {"price_change_pct": 0.0}}),
        FakeProposalAgent("CMO", {"marketing": {"spend": 5_000.0, "channel": "ppc"}}),
        FakeProposalAgent("CPO", {"product": {"r_and_d_spend": 6_000.0}}),
    ]
    return Boardroom(
        agents,
        use_oracle=True,
        oracle_frequency=10,
        oracle_mode=fake_oracle.mode,
        oracle_instance=fake_oracle,
        oracle_cache_max_size=oracle_cache_max_size,
    )


def test_no_refresh_before_month_10_when_state_is_stable():
    fake_oracle = FakeOracle(key_builder=lambda state, trend_context, memories: ("month", str(state.months_elapsed)))
    boardroom = make_boardroom(fake_oracle)

    boardroom.start_episode(episode_seed=1)
    boardroom.decide(make_state(0))
    boardroom.decide(make_state(5))

    stats = boardroom.get_episode_stats()
    assert len(fake_oracle.generate_calls) == 1
    assert stats["oracle_refresh_requests"] == 1
    assert stats["cadence_refreshes"] == 0
    assert stats["event_refreshes"] == 0


def test_cadence_refresh_fires_at_month_10():
    fake_oracle = FakeOracle(key_builder=lambda state, trend_context, memories: ("month", str(state.months_elapsed)))
    boardroom = make_boardroom(fake_oracle)

    boardroom.start_episode(episode_seed=1)
    boardroom.decide(make_state(0))
    boardroom.decide(make_state(10))

    stats = boardroom.get_episode_stats()
    assert len(fake_oracle.generate_calls) == 2
    assert stats["oracle_refresh_requests"] == 2
    assert stats["cadence_refreshes"] == 1
    assert stats["event_refreshes"] == 0


def test_event_refresh_fires_for_critical_changes():
    scenarios = [
        make_state(5, cash=300_000.0),
        make_state(5, mrr=85_000.0),
        make_state(5, churn_enterprise=0.035, churn_smb=0.045, churn_b2c=0.055),
        make_state(5, consumer_confidence=85.0),
        make_state(5, unemployment=6.0),
    ]

    for scenario in scenarios:
        fake_oracle = FakeOracle(key_builder=lambda state, trend_context, memories: ("month", str(state.months_elapsed)))
        boardroom = make_boardroom(fake_oracle)
        boardroom.start_episode(episode_seed=1)
        boardroom.decide(make_state(0))
        boardroom.decide(scenario)

        stats = boardroom.get_episode_stats()
        assert len(fake_oracle.generate_calls) == 2
        assert stats["event_refreshes"] == 1


def test_sub_threshold_changes_do_not_trigger_event_refresh():
    fake_oracle = FakeOracle(key_builder=lambda state, trend_context, memories: ("month", str(state.months_elapsed)))
    boardroom = make_boardroom(fake_oracle)

    boardroom.start_episode(episode_seed=1)
    boardroom.decide(make_state(0))
    boardroom.decide(
        make_state(
            5,
            mrr=86_000.0,
            churn_enterprise=0.034,
            churn_smb=0.044,
            churn_b2c=0.054,
            consumer_confidence=86.0,
            unemployment=5.9,
            cash=500_000.0,
        )
    )

    stats = boardroom.get_episode_stats()
    assert len(fake_oracle.generate_calls) == 1
    assert stats["event_refreshes"] == 0


def test_missing_brief_forces_refresh_even_off_cadence():
    fake_oracle = FakeOracle()
    boardroom = make_boardroom(fake_oracle)

    boardroom.start_episode(episode_seed=1)
    boardroom.decide(make_state(7))

    stats = boardroom.get_episode_stats()
    assert len(fake_oracle.generate_calls) == 1
    assert stats["oracle_refresh_requests"] == 1


def test_identical_context_reuses_cached_brief_without_new_llm_call():
    fake_oracle = FakeOracle(key_builder=lambda state, trend_context, memories: ("same-context",))
    boardroom = make_boardroom(fake_oracle)

    boardroom.start_episode(episode_seed=1)
    boardroom.decide(make_state(0))
    assert len(fake_oracle.generate_calls) == 1

    boardroom.start_episode(episode_seed=2)
    boardroom.decide(make_state(0))

    stats = boardroom.get_episode_stats()
    assert len(fake_oracle.generate_calls) == 1
    assert stats["oracle_refresh_requests"] == 1
    assert stats["cache_hits"] == 1
    assert stats["llm_calls"] == 0


def test_cache_key_changes_when_bucket_changes():
    oracle = Oracle(mode="oracle_v1", llm=StaticLLM())
    trend_context = TrendContext(history_points=5)

    key_a = oracle.build_cache_key(make_state(1, mrr=90_000.0), trend_context=trend_context, memories=[])
    key_b = oracle.build_cache_key(make_state(1, mrr=120_000.0), trend_context=trend_context, memories=[])

    assert key_a != key_b


def test_oracle_v3_memory_signature_uses_top_two_memories_only():
    oracle = Oracle(mode="oracle_v3", llm=StaticLLM())
    trend_context = TrendContext(history_points=5)
    state = make_state(1)

    memories_a = [
        RetrievedMemoryCandidate(document="m1", metadata={"source_month": 1, "realized_outcome": "DECLINE"}),
        RetrievedMemoryCandidate(document="m2", metadata={"source_month": 2, "realized_outcome": "GROWTH"}),
        RetrievedMemoryCandidate(document="m3", metadata={"source_month": 3, "realized_outcome": "STAGNATION"}),
    ]
    memories_b = [
        RetrievedMemoryCandidate(document="m1", metadata={"source_month": 1, "realized_outcome": "DECLINE"}),
        RetrievedMemoryCandidate(document="m2", metadata={"source_month": 2, "realized_outcome": "GROWTH"}),
        RetrievedMemoryCandidate(document="m4", metadata={"source_month": 4, "realized_outcome": "DECLINE"}),
    ]
    memories_c = [
        RetrievedMemoryCandidate(document="m1", metadata={"source_month": 1, "realized_outcome": "DECLINE"}),
        RetrievedMemoryCandidate(document="m2", metadata={"source_month": 2, "realized_outcome": "DECLINE"}),
        RetrievedMemoryCandidate(document="m3", metadata={"source_month": 3, "realized_outcome": "STAGNATION"}),
    ]

    key_a = oracle.build_cache_key(state, trend_context=trend_context, memories=memories_a)
    key_b = oracle.build_cache_key(state, trend_context=trend_context, memories=memories_b)
    key_c = oracle.build_cache_key(state, trend_context=trend_context, memories=memories_c)

    assert key_a == key_b
    assert key_a != key_c


def test_cache_evicts_oldest_entries_when_capacity_is_exceeded():
    fake_oracle = FakeOracle(
        key_builder=lambda state, trend_context, memories: (f"key_{int(state.consumer_confidence)}",)
    )
    boardroom = make_boardroom(fake_oracle, oracle_cache_max_size=2)

    boardroom.start_episode(episode_seed=1)
    boardroom.decide(make_state(0, consumer_confidence=101.0))
    boardroom.start_episode(episode_seed=2)
    boardroom.decide(make_state(0, consumer_confidence=102.0))
    boardroom.start_episode(episode_seed=3)
    boardroom.decide(make_state(0, consumer_confidence=103.0))

    assert len(boardroom.oracle_cache) == 2
    assert ("key_101",) not in boardroom.oracle_cache

    boardroom.start_episode(episode_seed=4)
    boardroom.decide(make_state(0, consumer_confidence=101.0))

    stats = boardroom.get_episode_stats()
    assert len(fake_oracle.generate_calls) == 4
    assert stats["cache_hits"] == 0
    assert stats["llm_calls"] == 1


def test_cache_isolated_between_separate_boardroom_runs():
    first_oracle = FakeOracle(key_builder=lambda state, trend_context, memories: ("shared",))
    first_boardroom = make_boardroom(first_oracle)
    first_boardroom.start_episode(episode_seed=1)
    first_boardroom.decide(make_state(0))

    second_oracle = FakeOracle(key_builder=lambda state, trend_context, memories: ("shared",))
    second_boardroom = make_boardroom(second_oracle)
    second_boardroom.start_episode(episode_seed=1)
    second_boardroom.decide(make_state(0))

    stats = second_boardroom.get_episode_stats()
    assert len(second_oracle.generate_calls) == 1
    assert stats["cache_hits"] == 0
    assert stats["llm_calls"] == 1


def test_run_simulation_logs_episode_end_and_includes_oracle_stat_columns(monkeypatch, capsys):
    class FakeEnv:
        def __init__(self):
            self.state = None

        def reset(self, seed=None):
            self.state = make_state(0)
            return np.zeros(16), {}

        def step(self, action):
            self.state.months_elapsed += 1
            return np.zeros(16), 1.0, True, False, {"rule_of_40": 10.0, "state": self.state.model_dump()}

    monkeypatch.setattr(simulation_runner, "StartupEnv", FakeEnv)

    df = simulation_runner.run_simulation(policy="random", num_episodes=2, seed_start=11)
    output = capsys.readouterr().out

    assert output.count("EPISODE_END | policy=random") == 2
    for column in ["oracle_refresh_requests", "cadence_refreshes", "event_refreshes", "cache_hits", "llm_calls"]:
        assert column in df.columns
        assert (df[column] == 0).all()
