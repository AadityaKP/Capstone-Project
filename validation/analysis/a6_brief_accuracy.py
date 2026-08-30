"""A6(i): does the memory-conditioned oracle_v3 brief predict outcomes?

Recorded FULL-run data only. At each fresh brief (brief_source == 'llm' or
'cache_hit') the v3 brief commits to expected_outcome in {GROWTH, STAGNATION,
DECLINE} for the next 6-12 months. Realized outcome: MRR at t+6 vs t with the
repo's own +/-10 percent thresholds (oracle/memory.py). Accuracy is compared
with the majority-class base rate; uncertainty via episode-clustered bootstrap.

Writes validation/results/brief_accuracy.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background"
OUT = ROOT / "validation/results"

at = pd.read_csv(FULL / "primary_action_trace.csv",
                 usecols=["policy", "episode", "month", "brief_source",
                          "brief_expected_outcome"])
mt = pd.read_csv(FULL / "primary_monthly_trace.csv",
                 usecols=["policy", "episode", "month", "mrr"])

v3 = at[(at.policy == "oracle_v3")
        & at.brief_source.isin(["llm", "cache_hit"])
        & at.brief_expected_outcome.notna()].copy()

mrr = mt[mt.policy == "oracle_v3"].set_index(["episode", "month"]).mrr

def realized(ep, m):
    try:
        now = mrr.loc[(ep, m)]
        fut = mrr.loc[(ep, m + 6)]
    except KeyError:
        return None
    pct = (fut - now) / max(abs(now), 1.0)
    if pct > 0.10:
        return "GROWTH"
    if pct < -0.10:
        return "DECLINE"
    return "STAGNATION"

v3["realized"] = [realized(e, m) for e, m in zip(v3.episode, v3.month)]
v3 = v3.dropna(subset=["realized"])
v3["correct"] = v3.brief_expected_outcome == v3.realized

n = len(v3)
acc = v3.correct.mean()
base_rate = v3.realized.value_counts(normalize=True).max()
majority = v3.realized.value_counts().idxmax()

# episode-clustered bootstrap for accuracy - base_rate
rng = np.random.default_rng(0)
episodes = v3.episode.unique()
groups = {e: g for e, g in v3.groupby("episode")}
deltas = []
for _ in range(5000):
    chosen = rng.choice(episodes, size=len(episodes), replace=True)
    sample = pd.concat([groups[e] for e in chosen])
    deltas.append(sample.correct.mean()
                  - sample.realized.value_counts(normalize=True).max())
lo, hi = np.percentile(deltas, [2.5, 97.5])

conf = pd.crosstab(v3.brief_expected_outcome, v3.realized, normalize="index").round(3)
print("confusion (rows=predicted, cols=realized, row-normalized):")
print(conf)
print(f"\nn={n} fresh-brief predictions across {v3.episode.nunique()} episodes")
print(f"accuracy={acc:.3f} vs majority-class base rate={base_rate:.3f} ({majority})")
print(f"accuracy - base rate = {acc-base_rate:+.3f}, 95% cluster-bootstrap CI [{lo:+.3f}, {hi:+.3f}]")

res = pd.DataFrame([dict(
    test="A6i_brief_expected_outcome_accuracy", n=n,
    episodes=int(v3.episode.nunique()), accuracy=round(acc, 4),
    base_rate=round(base_rate, 4), majority_class=majority,
    delta=round(acc - base_rate, 4), ci95_lo=round(lo, 4), ci95_hi=round(hi, 4),
    verdict="PASS" if lo > 0 else "FAIL",
)])
res.to_csv(OUT / "brief_accuracy.csv", index=False)
print("\n", res.to_string(index=False))
