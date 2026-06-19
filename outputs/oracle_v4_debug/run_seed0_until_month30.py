import json
import os
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from agents.adapter import ActionAdapter
from env.startup_env import StartupEnv
from simulation_runner import (
    _build_agent_for_policy,
    _capture_causal_metrics,
    _write_causal_step_outcome,
)


OUT_DIR = Path("outputs/oracle_v4_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

env = StartupEnv()
agent = _build_agent_for_policy("oracle_v4_causal_hetero", oracle_frequency=5)

episode_seed = 0
env.reset(seed=episode_seed)
agent.start_episode(episode_seed)
agent.set_shock_label(None)
random.seed(episode_seed)
np.random.seed(episode_seed)

terminated = False
truncated = False
rows = []

while not (terminated or truncated) and env.state.months_elapsed <= 30:
    current_month = env.state.months_elapsed
    cash_before = float(env.state.cash)
    raw_action = agent.get_action(env.state)
    decision_trace = agent.get_last_decision_trace()
    clean_action = ActionAdapter.translate_action(raw_action)
    before_metrics = _capture_causal_metrics(env.state, agent)

    _, reward, terminated, truncated, info = env.step(clean_action)
    _write_causal_step_outcome(
        agent=agent,
        clean_action=clean_action,
        before_metrics=before_metrics,
        after_state=env.state,
        episode_seed=episode_seed,
        month=current_month,
    )
    agent.set_shock_label(info.get("shock_label"))

    trace = decision_trace or {}
    rows.append(
        {
            "month": current_month,
            "cash_before": cash_before,
            "cash_after": float(env.state.cash),
            "mrr_after": float(env.state.mrr),
            "reward": reward,
            "rule_of_40": info.get("rule_of_40"),
            "terminated": terminated,
            "truncated": truncated,
            "proposal_source": trace.get("proposal_source"),
            "causal_stress_node": trace.get("causal_stress_node"),
            "pre_modifier_action": trace.get("pre_modifier_action"),
            "post_modifier_action": trace.get("post_modifier_action"),
            "final_action": trace.get("final_action"),
            "proposals": [
                {
                    "agent": proposal.get("agent"),
                    "actions": proposal.get("actions"),
                    "causal_confidence": proposal.get("causal_confidence"),
                    "base_score": proposal.get("base_score"),
                    "final_confidence": proposal.get("final_confidence"),
                }
                for proposal in trace.get("proposals") or []
            ],
        }
    )
    print(json.dumps(rows[-1], sort_keys=True), flush=True)

survived_past_30 = not terminated and env.state.months_elapsed > 30
payload = {
    "survived_past_30": survived_past_30,
    "months_elapsed": int(env.state.months_elapsed),
    "terminated": terminated,
    "truncated": truncated,
    "final_cash": float(env.state.cash),
    "final_mrr": float(env.state.mrr),
    "trace": rows,
    "episode_stats": agent.get_episode_stats(),
}
(OUT_DIR / "seed0_after_fix_until_month30.json").write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
)
print("SEED0_MONTH30_RESULT", json.dumps(payload, sort_keys=True), flush=True)
