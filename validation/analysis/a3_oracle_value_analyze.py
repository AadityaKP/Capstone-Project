"""A3 analysis: paired oracle-value comparison on matched seeds (run after
a3_oracle_value_run.py). Writes validation/results/a3_oracle_value.csv and
a3_retrieval_decision_delta.csv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
A3 = ROOT / "validation/results/a3"
OUT = ROOT / "validation/results"

ARMS = ["boardroom", "oracle_v1", "oracle_v3", "oracle_v3_no_memory"]
eps = {a: pd.read_csv(A3 / f"episodes_{a}.csv").set_index("seed") for a in ARMS}


def hedges_g_paired(diff):
    n = len(diff)
    sd = diff.std(ddof=1)
    if sd == 0 or n < 2:
        return float("nan")
    d = diff.mean() / sd
    return d * (1 - 3 / (4 * n - 9)) if n > 3 else d


def boot_ci(diff, n_boot=10_000):
    rng = np.random.default_rng(0)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


rows = []
for arm in ARMS[1:]:
    for metric in ["final_mrr", "post_shock_avg_rule40_25_60"]:
        a = eps["boardroom"][metric]
        b = eps[arm][metric]
        seeds = a.index.intersection(b.index)
        diff = (b.loc[seeds] - a.loc[seeds]).dropna().to_numpy()
        lo, hi = boot_ci(diff)
        try:
            p = stats.wilcoxon(diff).pvalue if not np.allclose(diff, 0) else 1.0
        except ValueError:
            p = float("nan")
        rows.append(dict(comparison=f"{arm} - boardroom", metric=metric, n=len(diff),
                         mean_diff=float(diff.mean()), median_diff=float(np.median(diff)),
                         ci95_lo=lo, ci95_hi=hi, hedges_g_paired=hedges_g_paired(diff),
                         wilcoxon_p=float(p),
                         positive_seeds=int((diff > 0).sum())))

# retrieval value: v3 vs v3_no_memory head-to-head
for metric in ["final_mrr", "post_shock_avg_rule40_25_60"]:
    a = eps["oracle_v3_no_memory"][metric]
    b = eps["oracle_v3"][metric]
    seeds = a.index.intersection(b.index)
    diff = (b.loc[seeds] - a.loc[seeds]).dropna().to_numpy()
    lo, hi = boot_ci(diff)
    try:
        p = stats.wilcoxon(diff).pvalue if not np.allclose(diff, 0) else 1.0
    except ValueError:
        p = float("nan")
    rows.append(dict(comparison="oracle_v3 - oracle_v3_no_memory", metric=metric,
                     n=len(diff), mean_diff=float(diff.mean()),
                     median_diff=float(np.median(diff)), ci95_lo=lo, ci95_hi=hi,
                     hedges_g_paired=hedges_g_paired(diff), wilcoxon_p=float(p),
                     positive_seeds=int((diff > 0).sum())))

df = pd.DataFrame(rows)
df.to_csv(OUT / "a3_oracle_value.csv", index=False)
print(df.to_string(index=False))

# does retrieval change decisions? per (seed, month) action deltas
try:
    av3 = pd.read_csv(A3 / "actions_oracle_v3.csv")
    avn = pd.read_csv(A3 / "actions_oracle_v3_no_memory.csv")
    m = av3.merge(avn, on=["seed", "month"], suffixes=("_mem", "_nomem"))
    m["mkt_differs"] = (m.mkt_spend_mem - m.mkt_spend_nomem).abs() > 1.0
    m["rd_differs"] = (m.rd_spend_mem - m.rd_spend_nomem).abs() > 1.0
    m["brief_differs"] = (m.risk_level_mem.fillna("") != m.risk_level_nomem.fillna("")) | \
                         (m.growth_outlook_mem.fillna("") != m.growth_outlook_nomem.fillna(""))
    summary = pd.DataFrame([dict(
        months_compared=len(m),
        share_marketing_differs=m.mkt_differs.mean(),
        share_rd_differs=m.rd_differs.mean(),
        share_brief_differs=m.brief_differs.mean())])
    summary.to_csv(OUT / "a3_retrieval_decision_delta.csv", index=False)
    print("\n", summary.to_string(index=False))
except FileNotFoundError as exc:
    print(f"decision-delta skipped: {exc}")

# survival note
print("\nsurvival by arm:",
      {a: float((eps[a].cause == "Time Limit").mean()) for a in ARMS})
