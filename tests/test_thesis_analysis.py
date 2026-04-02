import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.schemas import EnvState
from experiments.thesis_analysis import compute_decision_divergence, compute_recovery_events, significance_test
import simulation_runner


class TinyOracleEnv:
    def __init__(self):
        self.state = None

    def reset(self, seed=None):
        self.state = EnvState(
            mrr=80_000.0,
            cash=2_000_000.0,
            cac=100.0,
            ltv=700.0,
            churn_enterprise=0.02,
            churn_smb=0.03,
            churn_b2c=0.04,
            interest_rate=8.0,
            consumer_confidence=65.0,
            competitors=9,
            product_quality=0.4,
            price=50.0,
            months_elapsed=0,
            headcount=5,
            valuation_multiple=8.0,
            unemployment=7.0,
            innovation_factor=0.5,
            months_in_depression=4,
        )
        return [0.0] * 16, {}

    def step(self, action):
        reward = 1.5
        rule_40 = 12.0 if self.state.months_elapsed == 0 else 8.0
        shock_label = "CUSTOM_SHOCK" if self.state.months_elapsed == 0 else "NO_SHOCK"
        self.state.months_elapsed += 1
        terminated = False
        truncated = self.state.months_elapsed >= 2
        return [0.0] * 16, reward, terminated, truncated, {
            "rule_of_40": rule_40,
            "state": self.state.model_dump(),
            "shock_label": shock_label,
        }


def test_compute_recovery_events_uses_pre_shock_rule_40():
    monthly_df = pd.DataFrame(
        [
            {
                "scenario_id": "boardroom",
                "scenario_label": "Boardroom Baseline",
                "policy": "boardroom",
                "episode": 0,
                "seed": 0,
                "month": 23,
                "rule_of_40": 10.0,
                "shock_label": "NO_SHOCK",
            },
            {
                "scenario_id": "boardroom",
                "scenario_label": "Boardroom Baseline",
                "policy": "boardroom",
                "episode": 0,
                "seed": 0,
                "month": 24,
                "rule_of_40": -5.0,
                "shock_label": "RATE_HIKE",
            },
            {
                "scenario_id": "boardroom",
                "scenario_label": "Boardroom Baseline",
                "policy": "boardroom",
                "episode": 0,
                "seed": 0,
                "month": 25,
                "rule_of_40": 2.0,
                "shock_label": "NO_SHOCK",
            },
            {
                "scenario_id": "boardroom",
                "scenario_label": "Boardroom Baseline",
                "policy": "boardroom",
                "episode": 0,
                "seed": 0,
                "month": 26,
                "rule_of_40": 11.0,
                "shock_label": "NO_SHOCK",
            },
        ]
    )

    recovery_df = compute_recovery_events(monthly_df)

    assert len(recovery_df) == 1
    assert recovery_df.iloc[0]["pre_shock_rule_40"] == 10.0
    assert recovery_df.iloc[0]["recovery_month"] == 26.0
    assert recovery_df.iloc[0]["recovery_time_months"] == 2.0
    assert bool(recovery_df.iloc[0]["recovered"]) is True


def test_compute_decision_divergence_reports_policy_difference():
    action_df = pd.DataFrame(
        [
            {
                "scenario_id": "boardroom",
                "scenario_label": "Boardroom Baseline",
                "seed": 0,
                "month": 24,
                "marketing_spend_final": 10_000.0,
                "rd_spend_final": 30_000.0,
                "hires_final": 2,
            },
            {
                "scenario_id": "oracle_v1",
                "scenario_label": "Oracle v1",
                "seed": 0,
                "month": 24,
                "marketing_spend_final": 7_000.0,
                "rd_spend_final": 45_000.0,
                "hires_final": 0,
            },
        ]
    )

    summary_df, detail_df = compute_decision_divergence(action_df, reference_scenario_id="boardroom")

    assert len(summary_df) == 1
    assert summary_df.iloc[0]["decision_difference_rate_pct"] == 100.0
    assert bool(detail_df.iloc[0]["decision_diff"]) is True


def test_run_simulation_can_return_richer_action_and_monthly_traces(monkeypatch):
    monkeypatch.setattr(simulation_runner, "StartupEnv", TinyOracleEnv)
    monkeypatch.setattr(
        "agents.llm_client.LLMClient.complete",
        lambda self, system_prompt, user_prompt: (
            '{"risk_level":"HIGH","growth_outlook":"DECLINING","efficiency_pressure":"HIGH",'
            '"innovation_urgency":"HIGH","macro_condition":"RECESSION","confidence":0.9}'
        ),
    )

    _, trace_payload = simulation_runner.run_simulation(
        policy="oracle_v1",
        num_episodes=1,
        seed_start=0,
        oracle_frequency=10,
        return_action_trace=True,
        return_monthly_trace=True,
    )

    assert set(trace_payload.keys()) == {"action_trace", "monthly_trace"}
    assert len(trace_payload["action_trace"]) == 2
    assert len(trace_payload["monthly_trace"]) == 2

    first_action = trace_payload["action_trace"][0]
    first_monthly = trace_payload["monthly_trace"][0]

    assert first_action["decision_trace"]["brief_source"] in {"llm", "cache_hit", "reuse"}
    assert "pre_modifier_action" in first_action["decision_trace"]
    assert "post_modifier_action" in first_action["decision_trace"]
    assert first_monthly["shock_label"] == "CUSTOM_SHOCK"
    assert first_monthly["decision_trace"]["final_action"] is not None


def test_significance_test_detects_distribution_gap():
    baseline_df = pd.DataFrame({"post_shock_avg_rule40": [-40.0, -38.0, -36.0, -35.0, -34.0]})
    oracle_df = pd.DataFrame({"post_shock_avg_rule40": [-12.0, -10.0, -8.0, -7.0, -5.0]})

    result = significance_test(baseline_df, oracle_df, metric="post_shock_avg_rule40")

    assert result["metric"] == "post_shock_avg_rule40"
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["n_a"] == 5
    assert result["n_b"] == 5


def test_ablation_policy_strings_build_expected_agent_configs():
    oracle_v1_no_modifier = simulation_runner._build_agent_for_policy("oracle_v1_no_modifier", oracle_frequency=10)
    oracle_v3_no_memory = simulation_runner._build_agent_for_policy("oracle_v3_no_memory", oracle_frequency=10)

    assert oracle_v1_no_modifier.boardroom.enable_action_modifier is False
    assert oracle_v1_no_modifier.boardroom.action_modifier.__class__.__name__ == "NoOpActionModifier"
    assert oracle_v3_no_memory.boardroom.oracle.enable_memory_retrieval is False
    assert oracle_v3_no_memory.boardroom.oracle.memory_store is None
