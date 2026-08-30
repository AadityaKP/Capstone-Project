"""Tier-1 figures from saved validation CSVs -> validation/figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "validation/results"
AG = ROOT / "validation/agents"
FIG = ROOT / "validation/figures"
FIG.mkdir(parents=True, exist_ok=True)
FULL = ROOT / "results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background"

plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})

# ---- F1: QoQ growth distribution, sim vs EDGAR ---------------------------
edgar = pd.read_csv(ROOT / "data/edgar_ratios.csv").qoq_growth.dropna()
mt = pd.read_csv(FULL / "primary_monthly_trace.csv",
                 usecols=["policy", "episode", "month", "mrr"])
mt["quarter"] = mt.month // 3
q = (mt[mt.policy == "boardroom"].groupby(["episode", "quarter"])
       .agg(qrev=("mrr", "sum"), n=("mrr", "size")).reset_index())
q = q[q.n == 3].sort_values(["episode", "quarter"])
q["g"] = q.groupby("episode").qrev.pct_change()
simg = q.g.dropna()

fig, ax = plt.subplots(figsize=(7, 4))
bins = np.linspace(-0.3, 0.5, 60)
ax.hist(edgar.clip(-0.3, 0.5), bins=bins, density=True, alpha=0.55,
        label=f"EDGAR panel (n={len(edgar)})", color="#2b6cb0")
ax.hist(simg.clip(-0.3, 0.5), bins=bins, density=True, alpha=0.55,
        label=f"Simulator, boardroom arm (n={len(simg)})", color="#dd6b20")
ax.set_xlabel("Quarter-over-quarter revenue growth")
ax.set_ylabel("Density")
ax.set_title("E1 - QoQ revenue growth: simulator vs 39-company EDGAR panel")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "E1_growth_distribution.png")
plt.close(fig)

# ---- F2: A2 policy baselines --------------------------------------------
ep = pd.read_csv(RES / "policy_comparison_episodes.csv")
order = ["noop", "random", "heuristic", "boardroom"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
data = [ep[ep.policy == p].final_mrr / 1e6 for p in order]
axes[0].boxplot(data, tick_labels=order, showfliers=False)
axes[0].set_yscale("log")
axes[0].set_ylabel("Final MRR ($M, log)")
axes[0].set_title("Final MRR by policy (50 matched seeds)")
surv = [ep[ep.policy == p].survived.mean() for p in order]
axes[1].bar(order, surv, color=["#718096", "#e53e3e", "#38a169", "#2b6cb0"])
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("Survival rate (120 months)")
axes[1].set_title("Survival by policy")
fig.suptitle("A2 - trivial baselines, deterministic RNG, paired seeds")
fig.tight_layout()
fig.savefig(FIG / "A2_policy_baselines.png")
plt.close(fig)

# ---- F3: A1 action ladders ----------------------------------------------
ae = pd.read_csv(AG / "action_effects.csv")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, sname in zip(axes, ["start_month0", "midgame_month18"]):
    for dim, g in ae[ae.state == sname].groupby("dimension"):
        med = g.groupby("value").final_mrr.median()
        x = np.arange(len(med))
        ax.plot(x, med / med.iloc[len(med) // 2], marker="o", label=dim.split(".")[-1])
        ax.set_xticks(x)
    ax.set_title(sname)
    ax.set_xlabel("ladder rung (low - high)")
    ax.set_ylabel("median 12-mo MRR / mid-rung")
    ax.legend(fontsize=8)
fig.suptitle("A1 - one-dimension action ladders, matched seeds")
fig.tight_layout()
fig.savefig(FIG / "A1_action_ladders.png")
plt.close(fig)

# ---- F4: C1 backtest retrodiction ---------------------------------------
bt = pd.read_csv(RES / "real_company_backtest.csv")
sub = bt[bt.price_assumed == 250.0].dropna(subset=["sim_hold_median"])
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ax.scatter(sub.actual_4q_growth, sub.sim_hold_median, label="hold (company's own spend)",
           color="#dd6b20")
ax.scatter(sub.actual_4q_growth, sub.sim_boardroom_median, label="boardroom (scaled)",
           color="#2b6cb0", marker="s")
lim = [-0.1, 1.3]
ax.plot(lim, lim, "k--", lw=1, label="perfect retrodiction")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Actual next-4-quarter revenue growth (EDGAR)")
ax.set_ylabel("Simulated 4-quarter growth (median over 30 seeds)")
ax.set_title("C1 - real-company backtest: simulated vs actual growth")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "C1_backtest_retrodiction.png")
plt.close(fig)

print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
