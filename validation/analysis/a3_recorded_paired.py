"""Paired re-analysis of the recorded FULL run (n=75 per arm).

Justification: verify_shared_world.py shows that in legacy RNG mode the
non-drawing policies (boardroom, oracle_v1, oracle_v3) experience identical
exogenous macro paths at equal seed (10/10 seeds, 120 months). The FULL run's
arms therefore form a matched-seed design and paired statistics are valid.
(Divergence of competitors/price via MRR is a consequence of policy, not a
confound.) The original analysis used unpaired Mann-Whitney tests.

Writes validation/results/a3_recorded_paired.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background"
OUT = ROOT / "validation/results"

ep = pd.read_csv(FULL / "primary_episode_metric_summary.csv")


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
piv = {m: ep.pivot(index="seed", columns="policy", values=m)
       for m in ["final_mrr", "post_shock_avg_rule40"]}
for arm in ["oracle_v1", "oracle_v3"]:
    for metric in ["final_mrr", "post_shock_avg_rule40"]:
        p = piv[metric]
        diff = (p[arm] - p["boardroom"]).dropna().to_numpy()
        lo, hi = boot_ci(diff)
        w = stats.wilcoxon(diff)
        rows.append(dict(comparison=f"{arm} - boardroom", metric=metric, n=len(diff),
                         mean_diff=float(diff.mean()), median_diff=float(np.median(diff)),
                         ci95_lo=lo, ci95_hi=hi,
                         hedges_g_paired=hedges_g_paired(diff),
                         wilcoxon_p=float(w.pvalue),
                         positive_seeds=int((diff > 0).sum()),
                         design="paired (recorded FULL run, shared-world verified)"))

df = pd.DataFrame(rows)
df.to_csv(OUT / "a3_recorded_paired.csv", index=False)
print(df.to_string(index=False))
