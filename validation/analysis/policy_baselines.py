"""A2: trivial-baseline policy comparison under matched randomness.

Arms: noop / random / heuristic / boardroom. 50 seeds x 120 months,
deterministic_rng=True (env owns a private stream; per-step draw count fixed),
so every arm faces the identical shock tape at equal seed. Paired analysis:
per-seed differences, Wilcoxon signed-rank, Hedges-corrected paired effect size,
Holm adjustment across the metric family.

Also writes the decision-level log for the no-LLM arms:
validation/agents/decision_log.csv (policy, seed, month, state-before, action, state-after).
"""
from __future__ import annotations

import math
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

from agents.adapter import ActionAdapter
from agents.baseline_agents import merge_actions
from env.startup_env import StartupEnv
from simulation_runner import BoardroomAgent, RandomBundleAgent

OUT = ROOT / "validation/results"
AGENTS_OUT = ROOT / "validation/agents"
OUT.mkdir(parents=True, exist_ok=True)
AGENTS_OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 50
MAX_MONTHS = 120
NOOP = {
    "marketing": {"spend": 0.0, "channel": "ppc"},
    "hiring": {"hires": 0, "cost_per_employee": 10000.0},
    "product": {"r_and_d_spend": 0.0},
    "pricing": {"price_change_pct": 0.0},
}

STATE_COLS = ["mrr", "cash", "cac", "ltv", "product_quality", "innovation_factor",
              "consumer_confidence", "unemployment", "interest_rate", "competitors",
              "months_in_depression", "headcount"]


def get_action(policy: str, state, agent):
    if policy == "noop":
        return deepcopy(NOOP)
    if policy == "random":
        return agent.get_action(state)
    if policy == "heuristic":
        return merge_actions(state)
    return agent.get_action(state)  # boardroom


def run_episode(policy: str, seed: int, log_rows: list | None):
    env = StartupEnv(initial_config={"deterministic_rng": True})
    env.reset(seed=seed)
    random.seed(seed)          # policies' own draws (random arm) - isolated from physics
    np.random.seed(seed)
    agent = None
    if policy == "random":
        agent = RandomBundleAgent()
    elif policy == "boardroom":
        agent = BoardroomAgent(oracle_mode="none")
        agent.start_episode(seed)

    total_reward, r40_hist, post_shock = 0.0, [], []
    terminated = truncated = False
    while not (terminated or truncated):
        month = env.state.months_elapsed
        before = {c: getattr(env.state, c) for c in STATE_COLS}
        action = ActionAdapter.translate_action(get_action(policy, env.state, agent))
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        r40 = info["rule_of_40"]
        r40_hist.append(r40)
        if 25 <= month <= 60:
            post_shock.append(r40)
        if log_rows is not None:
            log_rows.append({
                "policy": policy, "seed": seed, "month": month,
                **{f"pre_{c}": before[c] for c in STATE_COLS},
                "mkt_spend": action["marketing"]["spend"],
                "mkt_channel": action["marketing"]["channel"],
                "rd_spend": action["product"]["r_and_d_spend"],
                "hires": action["hiring"]["hires"],
                "price_change_pct": action["pricing"]["price_change_pct"],
                "shock_label": info.get("shock_label", "NO_SHOCK"),
                "post_mrr": env.state.mrr, "post_cash": env.state.cash,
                "rule_of_40": r40, "reward": reward,
            })
    s = env.state
    return dict(policy=policy, seed=seed, steps=s.months_elapsed,
                survived=int(not terminated), final_mrr=s.mrr, final_cash=s.cash,
                avg_rule40=float(np.mean(r40_hist)),
                post_shock_avg_rule40=float(np.mean(post_shock)) if post_shock else np.nan,
                total_reward=total_reward)


def hedges_g_paired(diff):
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    sd = diff.std(ddof=1)
    if sd == 0 or n < 2:
        return float("nan")
    d = diff.mean() / sd
    return d * (1 - 3 / (4 * n - 9)) if n > 3 else d


def bootstrap_ci(diff, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    rows, log_rows = [], []
    for policy in ["noop", "random", "heuristic", "boardroom"]:
        for seed in range(N_SEEDS):
            rows.append(run_episode(policy, seed, log_rows))
        print(f"{policy}: done ({N_SEEDS} seeds)")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "policy_comparison_episodes.csv", index=False)
    pd.DataFrame(log_rows).to_csv(AGENTS_OUT / "decision_log.csv", index=False)

    summary = df.groupby("policy").agg(
        survival=("survived", "mean"), median_final_mrr=("final_mrr", "median"),
        mean_final_mrr=("final_mrr", "mean"), mean_post_shock_r40=("post_shock_avg_rule40", "mean"),
        mean_steps=("steps", "mean")).reset_index()
    summary.to_csv(OUT / "policy_comparison.csv", index=False)
    print("\n", summary.to_string(index=False))

    tests = []
    metrics = ["final_mrr", "post_shock_avg_rule40", "survived"]
    pivot = {m: df.pivot(index="seed", columns="policy", values=m) for m in metrics}
    for a, b in combinations(["noop", "random", "heuristic", "boardroom"], 2):
        for m in metrics:
            pa, pb = pivot[m][a], pivot[m][b]
            mask = pa.notna() & pb.notna()
            diff = (pb - pa)[mask].to_numpy()
            if m == "survived" or np.allclose(diff, 0):
                p = stats.wilcoxon(diff).pvalue if not np.allclose(diff, 0) else 1.0
            else:
                p = stats.wilcoxon(diff).pvalue
            lo, hi = bootstrap_ci(diff)
            tests.append(dict(test="A2_paired", metric=m, arm_a=a, arm_b=b, n=int(mask.sum()),
                              mean_diff_b_minus_a=float(diff.mean()),
                              ci95_lo=lo, ci95_hi=hi,
                              hedges_g_paired=hedges_g_paired(diff),
                              wilcoxon_p=float(p)))
    tdf = pd.DataFrame(tests)
    # Holm within each baseline family vs boardroom
    fam = tdf[(tdf.arm_b == "boardroom")].copy()
    order = fam.wilcoxon_p.sort_values().index
    m_count = len(order)
    adj = {}
    prev = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m_count - rank) * fam.loc[idx, "wilcoxon_p"])
        prev = max(prev, val)
        adj[idx] = prev
    tdf["holm_p_vs_boardroom"] = pd.Series(adj)
    tdf.to_csv(OUT / "statistical_tests_policy_baselines.csv", index=False)
    print("\n", tdf[["metric", "arm_a", "arm_b", "mean_diff_b_minus_a",
                     "hedges_g_paired", "wilcoxon_p", "holm_p_vs_boardroom"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
