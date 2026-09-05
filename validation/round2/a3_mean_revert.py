"""S11 stretch: A3 under recoverable shocks (LLM job - queued).

boardroom + oracle_v3, 20 matched seeds, legacy physics +
shock_recovery="mean_revert" (3-month half-life on hard-shock price/churn
damage), deterministic_rng, freq 10. Writes a3_oracle_value_mr.csv (episodes)
and a3_mr_monthly.csv (MRR series for the E6 recomputation). Arm-level
resume like the RS runner.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.adapter import ActionAdapter  # noqa: E402
from env.startup_env import StartupEnv  # noqa: E402
from simulation_runner import _build_agent_for_policy  # noqa: E402

RESULTS = ROOT / "validation/results"
N_EPISODES = 20
FREQ = 10
ENV = {"deterministic_rng": True, "shock_recovery": "mean_revert"}
POLICIES = ["boardroom", "oracle_v3"]


def main() -> None:
    ep_rows, monthly_rows = [], []
    done: set[str] = set()
    ep_path = RESULTS / "a3_oracle_value_mr.csv"
    mo_path = RESULTS / "a3_mr_monthly.csv"
    if ep_path.exists():
        prev = pd.read_csv(ep_path)
        done = {p for p, g in prev.groupby("policy")
                if g.seed.nunique() >= N_EPISODES}
        ep_rows = prev[prev.policy.isin(done)].to_dict("records")
        if mo_path.exists():
            pm = pd.read_csv(mo_path)
            monthly_rows = pm[pm.policy.isin(done)].to_dict("records")
        print(f"resuming; complete arms kept: {sorted(done)}", flush=True)
    for policy in POLICIES:
        if policy in done:
            continue
        t0 = time.time()
        agent = _build_agent_for_policy(policy, FREQ)
        for seed in range(N_EPISODES):
            env = StartupEnv(initial_config=dict(ENV))
            env.reset(seed=seed)
            if hasattr(agent, "start_episode"):
                agent.start_episode(seed)
            if hasattr(agent, "set_shock_label"):
                agent.set_shock_label(None)
            random.seed(seed)
            np.random.seed(seed)
            terminated = truncated = False
            while not (terminated or truncated):
                month = env.state.months_elapsed
                action = ActionAdapter.translate_action(agent.get_action(env.state))
                _, _, terminated, truncated, info = env.step(action)
                if hasattr(agent, "set_shock_label"):
                    agent.set_shock_label(info.get("shock_label"))
                monthly_rows.append(dict(policy=policy, seed=seed, month=month,
                                         mrr=env.state.mrr,
                                         rule_of_40=info["rule_of_40"]))
            ep_rows.append(dict(policy=policy, seed=seed,
                                steps=env.state.months_elapsed,
                                survived=int(not terminated),
                                final_mrr=env.state.mrr,
                                final_cash=env.state.cash))
            print(f"{policy} seed {seed}: mrr={env.state.mrr:,.0f}", flush=True)
            pd.DataFrame(ep_rows).to_csv(ep_path, index=False)
        pd.DataFrame(monthly_rows).to_csv(mo_path, index=False)
        print(f"DONE {policy} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
