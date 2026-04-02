import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import simulation_runner
from env.schemas import EnvState


class ShockTestEnv:
    def __init__(self):
        self.state = None

    def reset(self, seed=None):
        self.state = EnvState(
            mrr=50_000.0,
            cash=2_000_000.0,
            cac=100.0,
            ltv=700.0,
            churn_enterprise=0.03,
            churn_smb=0.05,
            churn_b2c=0.06,
            interest_rate=8.5,
            consumer_confidence=60.0,
            competitors=10,
            product_quality=0.3,
            price=50.0,
            months_elapsed=0,
            headcount=5,
            valuation_multiple=8.0,
            unemployment=9.0,
            innovation_factor=0.4,
            months_in_depression=4,
        )
        return [0.0] * 16, {}

    def step(self, action):
        self.state.months_elapsed += 1
        terminated = False
        truncated = self.state.months_elapsed >= 25
        info = {
            "rule_of_40": -5.0,
            "state": self.state.model_dump(),
            "shock_label": "CUSTOM_SHOCK",
        }
        return [0.0] * 16, 0.0, terminated, truncated, info


def _find_month_action(trace, target_month):
    for row in trace:
        if row["month"] == target_month:
            return row
    raise AssertionError(f"No action captured for month {target_month}")


def test_oracle_has_nonzero_effect_on_actions(monkeypatch):
    monkeypatch.setattr(simulation_runner, "StartupEnv", ShockTestEnv)
    monkeypatch.setattr(
        "agents.llm_client.LLMClient.complete",
        lambda self, system_prompt, user_prompt: (
            '{"risk_level":"HIGH","growth_outlook":"COLLAPSING","efficiency_pressure":"HIGH",'
            '"innovation_urgency":"HIGH","macro_condition":"RECESSION","confidence":0.9}'
        ),
    )

    _, baseline_trace = simulation_runner.run_simulation(
        policy="boardroom",
        num_episodes=1,
        seed_start=0,
        return_action_trace=True,
    )
    _, oracle_trace = simulation_runner.run_simulation(
        policy="oracle_v1",
        num_episodes=1,
        seed_start=0,
        oracle_frequency=10,
        return_action_trace=True,
    )

    baseline_row = _find_month_action(baseline_trace, 24)
    oracle_row = _find_month_action(oracle_trace, 24)

    baseline_action = baseline_row["action"]
    oracle_action = oracle_row["action"]

    marketing_diff = abs(oracle_action["marketing"]["spend"] - baseline_action["marketing"]["spend"]) / max(
        baseline_action["marketing"]["spend"], 1.0
    )
    rd_diff = abs(oracle_action["product"]["r_and_d_spend"] - baseline_action["product"]["r_and_d_spend"]) / max(
        baseline_action["product"]["r_and_d_spend"], 1.0
    )
    hires_diff = abs(oracle_action["hiring"]["hires"] - baseline_action["hiring"]["hires"])

    if not (marketing_diff > 0.05 or rd_diff > 0.05 or hires_diff > 0):
        diagnostic = (
            f"brief={oracle_row['brief']} | baseline={baseline_action} | oracle={oracle_action}"
        )
        raise AssertionError(
            "Oracle has zero effect on actions — check ActionModifier integration\n" + diagnostic
        )
