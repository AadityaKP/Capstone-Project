"""D5: attribute E2 (persistence) / E5 (volatility) to mechanisms. Diagnose only.

Research scale (default init, boardroom no-oracle arm, deterministic_rng,
10 seeds x 120 months), four variants:

  baseline      as recorded runs (scheduled shocks on, all draws live)
  hill_pinned   marketing Hill draws consumed but pinned to channel midpoints
                (removes demand noise, keeps macro randomness)
  macro_frozen  random macro shocks consume draws but do not act; scheduled
                shocks off (removes macro variation, keeps demand noise)
  both          hill_pinned + macro_frozen

E5 metric: within-episode std of quarterly growth. E2: lag-1 autocorr of
quarterly growth. Medians across episodes, exactly as environment_battery.py
computes them. Writes d5_volatility_attribution.csv + .png.
"""
from __future__ import annotations

import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import ROOT, CAL_DIR  # noqa: E402

from agents.adapter import ActionAdapter  # noqa: E402
from env import business_logic  # noqa: E402
from env.business_logic import hill_response  # noqa: E402
from env.startup_env import StartupEnv  # noqa: E402
from simulation_runner import BoardroomAgent  # noqa: E402

SEEDS = range(10)
MONTHS = 120


@contextmanager
def variant_patches(hill_pinned: bool, macro_frozen: bool):
    orig_new_mrr = business_logic.compute_new_mrr
    orig_shocks = (business_logic.interest_rate_shock,
                   business_logic.consumer_confidence_shock,
                   business_logic.competitive_entry_shock)
    try:
        if hill_pinned:
            def pinned(state, action, scale_aware=False, rng=None,
                       saturation_rate=None):
                draw = business_logic._stream(rng)
                # consume exactly the legacy branch's three draws
                draw.uniform(0.5, 1.0); draw.uniform(15_000, 50_000)
                if action.channel == "ppc":
                    draw.uniform(10_000, 50_000)
                    alpha, gamma, beta = 0.75, 32_500.0, 30_000.0
                else:
                    draw.uniform(50_000, 100_000)
                    alpha, gamma, beta = 2.25, 32_500.0, 75_000.0
                response = hill_response(action.spend, alpha, beta, gamma)
                if state.consumer_confidence < 80:
                    response *= 0.85
                elif state.consumer_confidence > 120:
                    response *= 1.08
                if state.competitors >= 10:
                    response *= 0.6
                elif state.competitors >= 4:
                    response *= 0.8
                return response
            business_logic.compute_new_mrr = pinned
        if macro_frozen:
            def frozen(state, prob=0.1, rng=None, **kwargs):
                business_logic._stream(rng).random()  # consume, no effect
            business_logic.interest_rate_shock = frozen
            business_logic.consumer_confidence_shock = frozen
            business_logic.competitive_entry_shock = frozen
        yield
    finally:
        business_logic.compute_new_mrr = orig_new_mrr
        (business_logic.interest_rate_shock,
         business_logic.consumer_confidence_shock,
         business_logic.competitive_entry_shock) = orig_shocks


def run_episode(seed: int, scheduled_shocks: bool) -> pd.DataFrame:
    env = StartupEnv(initial_config={"deterministic_rng": True,
                                     "scheduled_shocks": scheduled_shocks})
    env.reset(seed=seed)
    agent = BoardroomAgent(oracle_mode="none")
    agent.start_episode(seed)
    rows = []
    for _ in range(MONTHS):
        action = ActionAdapter.translate_action(agent.get_action(env.state))
        _, _, terminated, truncated, _ = env.step(action)
        rows.append(dict(month=env.state.months_elapsed - 1, mrr=env.state.mrr))
        if terminated or truncated:
            break
    return pd.DataFrame(rows)


def episode_metrics(df: pd.DataFrame) -> tuple[float, float]:
    df = df.assign(quarter=df.month // 3)
    q = df.groupby("quarter").agg(qrev=("mrr", "sum"), n=("mrr", "size"))
    q = q[q.n == 3]
    g = (q.qrev / q.qrev.shift(1) - 1.0).dropna().to_numpy()
    if len(g) < 3:
        return np.nan, np.nan
    ac = (np.corrcoef(g[:-1], g[1:])[0, 1]
          if np.std(g[:-1]) > 0 and np.std(g[1:]) > 0 else np.nan)
    return float(np.std(g, ddof=1)), float(ac)


def main() -> None:
    variants = dict(
        baseline=dict(hill_pinned=False, macro_frozen=False, scheduled=True),
        hill_pinned=dict(hill_pinned=True, macro_frozen=False, scheduled=True),
        macro_frozen=dict(hill_pinned=False, macro_frozen=True, scheduled=False),
        both=dict(hill_pinned=True, macro_frozen=True, scheduled=False))
    rows = []
    for name, v in variants.items():
        with variant_patches(v["hill_pinned"], v["macro_frozen"]):
            for seed in SEEDS:
                df = run_episode(seed, v["scheduled"])
                vol, ac = episode_metrics(df)
                rows.append(dict(variant=name, seed=seed,
                                 growth_std=vol, lag1_autocorr=ac,
                                 months=len(df)))
        print(f"{name} done")
    out = pd.DataFrame(rows)
    out.to_csv(CAL_DIR / "d5_volatility_attribution.csv", index=False)

    summ = out.groupby("variant")[["growth_std", "lag1_autocorr"]].median()
    summ = summ.reindex(["baseline", "hill_pinned", "macro_frozen", "both"])
    print("\nmedians (EDGAR: growth_std 0.046, lag1_autocorr 0.46):")
    print(summ.to_string(float_format="%.3f"))

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, col, edgar, title in [
            (axes[0], "growth_std", 0.046, "E5 within-episode growth std"),
            (axes[1], "lag1_autocorr", 0.46, "E2 lag-1 autocorr")]:
        data = [out[out.variant == v][col].dropna() for v in summ.index]
        ax.boxplot(data, tick_labels=summ.index)
        ax.axhline(edgar, color="red", ls="--", lw=1, label=f"EDGAR {edgar}")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(CAL_DIR / "d5_volatility_attribution.png", dpi=150)


if __name__ == "__main__":
    main()
