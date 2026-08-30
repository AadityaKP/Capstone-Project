"""C2 (observational, supporting evidence only): do real companies that
recovered from stress states shift spending in the direction the agent does?

Stress company-quarter: QoQ growth below the panel p25 AND discretionary spend
intensity (S&M%+R&D%) above the panel median - the "declining + heavy spending"
profile. Agent behaviour under stress (A4 exploratory + ActionModifier): cut
marketing, protect/raise R&D.

For each stress quarter t with 4 subsequent quarters: outcome = revenue growth
t -> t+4; allocation change = mean intensity over t+1..t+2 minus intensity at t,
for S&M% and R&D% separately. Companies-quarters are split at the median
outcome into 'improved' vs 'lagged' halves and allocation changes compared.

Two caveats, stated up front:
1. Intensity has revenue in the denominator, so growth mechanically lowers it;
   the absolute-spend growth columns are reported alongside for that reason.
2. This is observational direction-checking across surviving public SaaS
   companies. It supports or undermines *plausibility* of the agent's
   direction; it is not causal validation.

Writes validation/results/c2_allocation_consistency.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "validation/results"

df = pd.read_csv(ROOT / "data/edgar_ratios.csv").dropna(
    subset=["revenue", "qoq_growth", "sm_pct_revenue", "rnd_pct_revenue"])
df["qi"] = df.fiscal_period.str.split("Q").map(lambda x: int(x[0]) * 4 + int(x[1]))
df = df.sort_values(["ticker", "qi"])
df["disc"] = df.sm_pct_revenue + df.rnd_pct_revenue

g_p25 = df.qoq_growth.quantile(0.25)
disc_med = df.disc.median()

idx = {(r.ticker, r.qi): r for r in df.itertuples()}

rows = []
for r in df.itertuples():
    if not (r.qoq_growth < g_p25 and r.disc > disc_med):
        continue
    nxt = [idx.get((r.ticker, r.qi + k)) for k in range(1, 5)]
    if any(v is None for v in nxt):
        continue
    outcome = nxt[3].revenue / r.revenue - 1.0
    sm_next = np.mean([nxt[0].sm_pct_revenue, nxt[1].sm_pct_revenue])
    rnd_next = np.mean([nxt[0].rnd_pct_revenue, nxt[1].rnd_pct_revenue])
    rows.append(dict(
        ticker=r.ticker, quarter=r.fiscal_period,
        growth_t=r.qoq_growth, outcome_4q_growth=outcome,
        d_sm_intensity=sm_next - r.sm_pct_revenue,
        d_rnd_intensity=rnd_next - r.rnd_pct_revenue,
        d_sm_abs=np.mean([nxt[0].revenue * nxt[0].sm_pct_revenue,
                          nxt[1].revenue * nxt[1].sm_pct_revenue])
                 / (r.revenue * r.sm_pct_revenue) - 1.0,
        d_rnd_abs=np.mean([nxt[0].revenue * nxt[0].rnd_pct_revenue,
                           nxt[1].revenue * nxt[1].rnd_pct_revenue])
                  / (r.revenue * r.rnd_pct_revenue) - 1.0))

s = pd.DataFrame(rows)
med = s.outcome_4q_growth.median()
s["group"] = np.where(s.outcome_4q_growth >= med, "improved", "lagged")

summary = s.groupby("group").agg(
    n=("ticker", "size"), companies=("ticker", "nunique"),
    median_d_sm_intensity=("d_sm_intensity", "median"),
    median_d_rnd_intensity=("d_rnd_intensity", "median"),
    median_d_sm_abs=("d_sm_abs", "median"),
    median_d_rnd_abs=("d_rnd_abs", "median"),
    median_outcome=("outcome_4q_growth", "median")).reset_index()

mw_sm = stats.mannwhitneyu(s[s.group == "improved"].d_sm_intensity,
                           s[s.group == "lagged"].d_sm_intensity)
mw_rnd = stats.mannwhitneyu(s[s.group == "improved"].d_rnd_intensity,
                            s[s.group == "lagged"].d_rnd_intensity)

s.to_csv(OUT / "c2_allocation_states.csv", index=False)
summary.to_csv(OUT / "c2_allocation_consistency.csv", index=False)
print(f"stress definition: QoQ growth < {g_p25:.3f} AND S&M%+R&D% > {disc_med:.3f}")
print(f"stress quarters with 4 subsequent quarters: {len(s)} "
      f"({s.ticker.nunique()} companies)\n")
print(summary.round(4).to_string(index=False))
print(f"\nMWU improved-vs-lagged d_sm_intensity p={mw_sm.pvalue:.3f}; "
      f"d_rnd_intensity p={mw_rnd.pvalue:.3f}")
print("(clustering by company not adjusted - observational supporting evidence only)")
