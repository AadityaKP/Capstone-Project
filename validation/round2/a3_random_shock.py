"""S7: random-shock-timing ablation (LLM job - queued after A3 v2phys).

boardroom / oracle_v3 / oracle_v3_no_memory, 20 matched seeds, legacy physics,
deterministic_rng, freq 10, shock_schedule="random" (3 months drawn per
episode from the world RNG: equal seeds -> equal schedules across arms).
Writes validation/results/a3_oracle_value_rs.csv (episode rows with the
schedule) and a3_rs_monthly.csv (r40 series for the recovery-rate gate).
"""
from __future__ import annotations

import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import random  # noqa: E402

from agents.adapter import ActionAdapter  # noqa: E402
from env.startup_env import StartupEnv  # noqa: E402
from simulation_runner import _build_agent_for_policy  # noqa: E402

RESULTS = ROOT / "validation/results"
N_EPISODES = 20
FREQ = 10
ENV = {"deterministic_rng": True, "shock_schedule": "random"}
POLICIES = ["boardroom", "oracle_v3", "oracle_v3_no_memory"]


def main() -> None:
    # Arm-level resume: a policy arm already complete in the episodes CSV is
    # skipped (an arm must be whole - oracle memory accrues across episodes
    # within an arm, so partial arms are discarded and re-run). Lets the job
    # survive an interrupted session without redoing finished arms.
    ep_rows, monthly_rows = [], []
    done_policies: set[str] = set()
    ep_path = RESULTS / "a3_oracle_value_rs.csv"
    mo_path = RESULTS / "a3_rs_monthly.csv"
    if ep_path.exists():
        prev = pd.read_csv(ep_path)
        done_policies = {p for p, g in prev.groupby("policy")
                         if g.seed.nunique() >= N_EPISODES}
        prev = prev[prev.policy.isin(done_policies)]
        ep_rows = prev.to_dict("records")
        if mo_path.exists():
            prev_mo = pd.read_csv(mo_path)
            monthly_rows = prev_mo[prev_mo.policy.isin(done_policies)].to_dict("records")
        print(f"resuming; complete arms kept: {sorted(done_policies)}", flush=True)
    for policy in POLICIES:
        if policy in done_policies:
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
            schedule = list(env.shock_months)
            terminated = truncated = False
            llm0 = (agent.get_episode_stats().get("llm_calls", 0)
                    if hasattr(agent, "get_episode_stats") else 0)
            while not (terminated or truncated):
                month = env.state.months_elapsed
                action = ActionAdapter.translate_action(agent.get_action(env.state))
                _, _, terminated, truncated, info = env.step(action)
                if hasattr(agent, "set_shock_label"):
                    agent.set_shock_label(info.get("shock_label"))
                monthly_rows.append(dict(policy=policy, seed=seed, month=month,
                                         rule_of_40=info["rule_of_40"],
                                         mrr=env.state.mrr,
                                         shock_label=info["shock_label"]))
            llm1 = (agent.get_episode_stats().get("llm_calls", 0)
                    if hasattr(agent, "get_episode_stats") else 0)
            ep_rows.append(dict(policy=policy, seed=seed,
                                steps=env.state.months_elapsed,
                                survived=int(not terminated),
                                final_mrr=env.state.mrr,
                                final_cash=env.state.cash,
                                shock_m1=schedule[0], shock_m2=schedule[1],
                                shock_m3=schedule[2],
                                llm_calls=max(0, llm1 - llm0)))
            print(f"{policy} seed {seed}: mrr={env.state.mrr:,.0f} "
                  f"schedule={schedule} llm={ep_rows[-1]['llm_calls']}",
                  flush=True)
            pd.DataFrame(ep_rows).to_csv(ep_path, index=False)
        # monthly rows land after each completed ARM so an interrupted session
        # keeps whole arms only (matching the episode-CSV resume contract)
        pd.DataFrame(monthly_rows).to_csv(mo_path, index=False)
        print(f"DONE {policy} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
