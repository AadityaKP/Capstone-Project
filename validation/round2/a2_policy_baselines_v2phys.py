"""A2 robustness rows under v2 physics (plan S3; decision D1).

Same design as the recorded A2 (validation/analysis/policy_baselines.py):
noop/random/heuristic/boardroom, 50 matched seeds x 120 months,
deterministic_rng, paired Wilcoxon / Hedges-g / Holm - but with the round-1
v2 physics flags (marketing_curve="v2", corridor="scale_aware",
competitive_entry="scale_neutral"; financing off at research scale).

Writes ONLY new files: policy_comparison_v2phys.csv,
policy_comparison_episodes_v2phys.csv, statistical_tests_policy_baselines_v2phys.csv.
"""
from __future__ import annotations

import random
import sys
from copy import deepcopy
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.adapter import ActionAdapter  # noqa: E402
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent  # noqa: E402
from agents.proposal_agents import (CFOProposalAgent, CMOProposalAgent,  # noqa: E402
                                    CPOProposalAgent)
from boardroom.boardroom import Boardroom  # noqa: E402
from env.startup_env import StartupEnv  # noqa: E402
from simulation_runner import RandomBundleAgent  # noqa: E402

sys.path.insert(0, str(ROOT / "validation/analysis"))
from policy_baselines import NOOP, bootstrap_ci, hedges_g_paired  # noqa: E402

OUT = ROOT / "validation/results"
N_SEEDS = 50
V2_CONFIG = {"deterministic_rng": True, "marketing_curve": "v2",
             "competitive_entry": "scale_neutral"}


def run_episode(policy: str, seed: int) -> dict:
    env = StartupEnv(initial_config=dict(V2_CONFIG))
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = None
    if policy == "random":
        agent = RandomBundleAgent()
    elif policy == "boardroom":
        agent = Boardroom(
            [CFOProposalAgent(corridor="scale_aware"),
             CMOProposalAgent(corridor="scale_aware"),
             CPOProposalAgent(corridor="scale_aware")],
            use_oracle=False, corridor="scale_aware")
        agent.start_episode(seed)
    elif policy == "heuristic":
        agent = (CFOAgent(corridor="scale_aware"), CMOAgent(corridor="scale_aware"),
                 CPOAgent(corridor="scale_aware"))

    total_reward, r40_hist, post_shock = 0.0, [], []
    terminated = truncated = False
    while not (terminated or truncated):
        month = env.state.months_elapsed
        if policy == "noop":
            action = deepcopy(NOOP)
        elif policy == "heuristic":
            action = {}
            for a in agent:
                action.update(a.act(env.state))
        elif policy == "random":
            action = agent.get_action(env.state)
        else:
            action = agent.decide(env.state)
        _, reward, terminated, truncated, info = env.step(
            ActionAdapter.translate_action(action))
        total_reward += reward
        r40_hist.append(info["rule_of_40"])
        if 25 <= month <= 60:
            post_shock.append(info["rule_of_40"])
    s = env.state
    return dict(physics_version="v2", policy=policy, seed=seed,
                steps=s.months_elapsed, survived=int(not terminated),
                final_mrr=s.mrr, final_cash=s.cash,
                avg_rule40=float(np.mean(r40_hist)),
                post_shock_avg_rule40=float(np.mean(post_shock)) if post_shock else np.nan,
                total_reward=total_reward)


def main() -> None:
    rows = []
    for policy in ["noop", "random", "heuristic", "boardroom"]:
        for seed in range(N_SEEDS):
            rows.append(run_episode(policy, seed))
        print(f"{policy}: done ({N_SEEDS} seeds)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "policy_comparison_episodes_v2phys.csv", index=False)

    summary = df.groupby("policy").agg(
        survival=("survived", "mean"), median_final_mrr=("final_mrr", "median"),
        mean_final_mrr=("final_mrr", "mean"),
        mean_post_shock_r40=("post_shock_avg_rule40", "mean"),
        mean_steps=("steps", "mean")).reset_index()
    summary["physics_version"] = "v2"
    summary.to_csv(OUT / "policy_comparison_v2phys.csv", index=False)
    print("\n", summary.to_string(index=False))

    tests = []
    metrics = ["final_mrr", "post_shock_avg_rule40", "survived"]
    pivot = {m: df.pivot(index="seed", columns="policy", values=m) for m in metrics}
    for a, b in combinations(["noop", "random", "heuristic", "boardroom"], 2):
        for m in metrics:
            pa, pb = pivot[m][a], pivot[m][b]
            mask = pa.notna() & pb.notna()
            diff = (pb - pa)[mask].to_numpy()
            p = 1.0 if np.allclose(diff, 0) else stats.wilcoxon(diff).pvalue
            lo, hi = bootstrap_ci(diff)
            tests.append(dict(test="A2_paired_v2phys", metric=m, arm_a=a, arm_b=b,
                              n=int(mask.sum()),
                              mean_diff_b_minus_a=float(diff.mean()),
                              ci95_lo=lo, ci95_hi=hi,
                              hedges_g_paired=hedges_g_paired(diff),
                              wilcoxon_p=float(p)))
    tdf = pd.DataFrame(tests)
    fam = tdf[(tdf.arm_b == "boardroom")].copy()
    order = fam.wilcoxon_p.sort_values().index
    adj, prev = {}, 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (len(order) - rank) * fam.loc[idx, "wilcoxon_p"])
        prev = max(prev, val)
        adj[idx] = prev
    tdf["holm_p_vs_boardroom"] = pd.Series(adj)
    tdf.to_csv(OUT / "statistical_tests_policy_baselines_v2phys.csv", index=False)
    print("\n", tdf[tdf.arm_b == "boardroom"][["metric", "arm_a",
          "mean_diff_b_minus_a", "hedges_g_paired", "holm_p_vs_boardroom"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
