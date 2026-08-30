"""Verify the legacy-RNG shared-world claim (system_audit.md section 1).

Claim: in legacy mode (global RNG), policies that never draw from the global
`random` module (heuristic, boardroom, oracle modes) experience identical
exogenous macro paths (interest rate, unemployment, consumer confidence) at
equal seed, because per-step draw counts are constant except the recession
cascade, whose trigger depends only on action-independent macro state.
The `random` policy draws globally and does desynchronise the world.

Writes validation/results/shared_world_check.csv.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.adapter import ActionAdapter
from agents.baseline_agents import merge_actions
from env.startup_env import StartupEnv
from simulation_runner import BoardroomAgent, RandomBundleAgent


def macro_path(policy: str, seed: int, months: int = 120):
    env = StartupEnv()          # legacy mode: global RNG
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = None
    if policy == "boardroom":
        agent = BoardroomAgent(oracle_mode="none")
        agent.start_episode(seed)
    elif policy == "random":
        agent = RandomBundleAgent()
    path = []
    for _ in range(months):
        if policy == "heuristic":
            action = merge_actions(env.state)
        else:
            action = agent.get_action(env.state)
        _, _, term, trunc, _ = env.step(ActionAdapter.translate_action(action))
        s = env.state
        path.append((round(s.interest_rate, 6), round(s.unemployment, 6),
                     round(s.consumer_confidence, 6)))
        if term or trunc:
            break
    return path


rows = []
for seed in range(10):
    h = macro_path("heuristic", seed)
    b = macro_path("boardroom", seed)
    r = macro_path("random", seed)
    n = min(len(h), len(b))
    hb_same = h[:n] == b[:n]
    n2 = min(len(h), len(r))
    hr_same = h[:n2] == r[:n2]
    rows.append(dict(seed=seed, months_compared_hb=n, heuristic_eq_boardroom=hb_same,
                     months_compared_hr=n2, heuristic_eq_random=hr_same))

df = pd.DataFrame(rows)
df.to_csv(ROOT / "validation/results/shared_world_check.csv", index=False)
print(df.to_string(index=False))
print(f"\nheuristic==boardroom macro paths (legacy RNG): {df.heuristic_eq_boardroom.all()}")
print(f"heuristic==random macro paths (legacy RNG):    {df.heuristic_eq_random.any()}")
