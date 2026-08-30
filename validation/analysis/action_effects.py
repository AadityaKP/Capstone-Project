"""A1: do actions causally move outcomes? One-dimension-at-a-time ladders.

Same base state, same seed (deterministic_rng), one action dimension varied,
12-month horizon, 20 seeds. Base state: the default $50k-MRR research start,
and a mid-game state produced by rolling the boardroom policy to month 18 at
seed 0 (so ladders are tested at two different operating points).

Writes validation/agents/action_effects.csv and a summary to stdout.
"""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.adapter import ActionAdapter
from env.startup_env import StartupEnv
from simulation_runner import BoardroomAgent

OUT = ROOT / "validation/agents"
OUT.mkdir(parents=True, exist_ok=True)

N_SEEDS = 20
HORIZON = 12
BASE = {
    "marketing": {"spend": 10_000.0, "channel": "ppc"},
    "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
    "product": {"r_and_d_spend": 8_000.0},
    "pricing": {"price_change_pct": 0.0},
}
LADDERS = {
    "marketing.spend": [0.0, 2_000.0, 10_000.0, 20_000.0, 50_000.0],
    "product.r_and_d_spend": [0.0, 5_000.0, 20_000.0, 50_000.0, 100_000.0],
    "hiring.hires": [0, 1, 2, 5],
    "pricing.price_change_pct": [-0.10, -0.05, 0.0, 0.05, 0.10],  # month 1 only
}


def make_states():
    env = StartupEnv(initial_config={"deterministic_rng": True})
    env.reset(seed=0)
    start_state = env.state.model_copy(deep=True)

    env2 = StartupEnv(initial_config={"deterministic_rng": True})
    env2.reset(seed=0)
    agent = BoardroomAgent(oracle_mode="none")
    agent.start_episode(0)
    for _ in range(18):
        action = ActionAdapter.translate_action(agent.get_action(env2.state))
        env2.step(action)
    mid_state = env2.state.model_copy(deep=True)
    return {"start_month0": start_state, "midgame_month18": mid_state}


def rollout(base_state, bundle, seed):
    env = StartupEnv(initial_config={"deterministic_rng": True})
    env.reset(seed=seed)
    env.state = base_state.model_copy(deep=True)
    survived = True
    for month in range(HORIZON):
        action = deepcopy(bundle)
        if month > 0:
            action["pricing"]["price_change_pct"] = 0.0  # one-time price move
        _, _, terminated, _, _ = env.step(ActionAdapter.translate_action(action))
        if terminated:
            survived = False
            break
    return env.state.mrr, env.state.cash, survived


def set_dim(bundle, dim, value):
    b = deepcopy(bundle)
    group, key = dim.split(".")
    b[group][key] = value
    return b


def main():
    states = make_states()
    rows = []
    for sname, st in states.items():
        print(f"state {sname}: mrr={st.mrr:,.0f} cash={st.cash:,.0f} pq={st.product_quality:.2f}")
        for dim, values in LADDERS.items():
            for v in values:
                bundle = set_dim(BASE, dim, v)
                for seed in range(N_SEEDS):
                    mrr, cash, surv = rollout(st, bundle, seed)
                    rows.append(dict(state=sname, dimension=dim, value=v, seed=seed,
                                     final_mrr=mrr, final_cash=cash, survived=int(surv)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "action_effects.csv", index=False)

    print("\nMedian 12-month terminal MRR by ladder rung (paired seeds):")
    summary = []
    for (sname, dim), g in df.groupby(["state", "dimension"]):
        med = g.groupby("value").final_mrr.median()
        base_val = BASE[dim.split(".")[0]][dim.split(".")[1]]
        mid = med.loc[base_val] if base_val in med.index else med.iloc[len(med) // 2]
        spread = (med.max() - med.min()) / max(abs(mid), 1.0)
        # monotonicity: Spearman of rung value vs per-seed-mean outcome
        per_rung = g.groupby("value").final_mrr.mean()
        from scipy.stats import spearmanr
        rho = spearmanr(per_rung.index.to_numpy(dtype=float), per_rung.to_numpy()).statistic
        summary.append(dict(state=sname, dimension=dim,
                            min_median=med.min(), max_median=med.max(),
                            spread_vs_base=spread, spearman_rung_vs_outcome=rho))
        print(f"  {sname:16s} {dim:26s} " +
              " | ".join(f"{v}:{m:,.0f}" for v, m in med.items()) +
              f"   spread/base={spread:.2f} rho={rho:+.2f}")
    pd.DataFrame(summary).to_csv(OUT / "action_effects_summary.csv", index=False)


if __name__ == "__main__":
    main()
