"""A7: robustness of the A2 policy ranking across initial conditions.

Grid: initial_mrr {25k, 50k, 100k} x initial_cash {0.5M, 1M, 2M}, 20 matched
seeds per cell, deterministic_rng. Same four arms as A2. The question is not
"how big is the effect in each cell" but "does the ranking (and the paired
boardroom advantage) survive across cells".

Writes validation/results/a7_robustness_grid.csv (per cell x policy) and
a7_robustness_pairs.csv (paired boardroom-vs-baseline effects per cell).
"""
from __future__ import annotations

import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.adapter import ActionAdapter
from agents.baseline_agents import merge_actions
from env.startup_env import StartupEnv
from simulation_runner import BoardroomAgent, RandomBundleAgent

OUT = ROOT / "validation/results"
N_SEEDS = 20
MRRS = [25_000, 50_000, 100_000]
CASHES = [500_000, 1_000_000, 2_000_000]
POLICIES = ["noop", "random", "heuristic", "boardroom"]

NOOP = {"marketing": {"spend": 0.0, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10000.0},
        "product": {"r_and_d_spend": 0.0},
        "pricing": {"price_change_pct": 0.0}}


def run_episode(policy, seed, mrr0, cash0):
    env = StartupEnv(initial_config={"deterministic_rng": True,
                                     "initial_mrr": mrr0, "initial_cash": cash0})
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = None
    if policy == "random":
        agent = RandomBundleAgent()
    elif policy == "boardroom":
        agent = BoardroomAgent(oracle_mode="none")
        agent.start_episode(seed)
    terminated = truncated = False
    while not (terminated or truncated):
        if policy == "noop":
            action = deepcopy(NOOP)
        elif policy == "heuristic":
            action = merge_actions(env.state)
        else:
            action = agent.get_action(env.state)
        _, _, terminated, truncated, _ = env.step(ActionAdapter.translate_action(action))
    return dict(policy=policy, seed=seed, initial_mrr=mrr0, initial_cash=cash0,
                survived=int(not terminated), final_mrr=env.state.mrr,
                steps=env.state.months_elapsed)


def hedges_g_paired(diff):
    n = len(diff)
    sd = diff.std(ddof=1)
    if sd == 0 or n < 2:
        return float("nan")
    d = diff.mean() / sd
    return d * (1 - 3 / (4 * n - 9)) if n > 3 else d


rows = []
for mrr0 in MRRS:
    for cash0 in CASHES:
        for policy in POLICIES:
            for seed in range(N_SEEDS):
                rows.append(run_episode(policy, seed, mrr0, cash0))
        print(f"cell mrr={mrr0} cash={cash0}: done")

df = pd.DataFrame(rows)
cell = (df.groupby(["initial_mrr", "initial_cash", "policy"])
          .agg(survival=("survived", "mean"), median_final_mrr=("final_mrr", "median"))
          .reset_index())
cell["rank_in_cell"] = (cell.groupby(["initial_mrr", "initial_cash"]).median_final_mrr
                            .rank(ascending=False).astype(int))
cell.to_csv(OUT / "a7_robustness_grid.csv", index=False)

pairs = []
for (mrr0, cash0), g in df.groupby(["initial_mrr", "initial_cash"]):
    piv = g.pivot(index="seed", columns="policy", values="final_mrr")
    for base in ["noop", "random", "heuristic"]:
        diff = (piv["boardroom"] - piv[base]).dropna().to_numpy()
        p = stats.wilcoxon(diff).pvalue if not np.allclose(diff, 0) else 1.0
        pairs.append(dict(initial_mrr=mrr0, initial_cash=cash0, baseline=base,
                          n=len(diff), mean_diff=float(diff.mean()),
                          hedges_g_paired=hedges_g_paired(diff),
                          wilcoxon_p=float(p),
                          positive_seeds=int((diff > 0).sum())))
pairs_df = pd.DataFrame(pairs)
pairs_df.to_csv(OUT / "a7_robustness_pairs.csv", index=False)

print("\nranking by median final MRR per cell (1 = best):")
print(cell.pivot_table(index=["initial_mrr", "initial_cash"], columns="policy",
                       values="rank_in_cell").to_string())
bd = cell[cell.policy == "boardroom"]
print(f"\nboardroom ranks 1st in {(bd.rank_in_cell == 1).sum()}/{len(bd)} cells")
print("\npaired boardroom advantage per cell:")
print(pairs_df.to_string(index=False))
