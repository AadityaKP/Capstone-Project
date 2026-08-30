"""E6 (exploratory): revenue drawdown/recovery, same definition on both panels.

Definition (identical code path for EDGAR companies and simulator episodes):
quarterly revenue series -> running peak; a drawdown episode begins when
revenue falls >=5% below the prior peak, its depth is the maximum decline from
that peak, and it recovers when revenue regains the prior peak. Episodes still
open at the end of a series are censored (counted, not given a duration).

Comparative only - the simulator's shocks are injected macro events, EDGAR
drawdowns arise from everything (COVID, competition, churn, macro), so no
pass/fail verdict was pre-declared. Writes validation/results/e6_drawdown_recovery.csv.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background"
OUT = ROOT / "validation/results"

THRESH = 0.05


def episodes_of(series):
    """Drawdown episodes in one quarterly revenue series."""
    out = []
    peak = series[0]
    in_dd, depth, start = False, 0.0, None
    for i, v in enumerate(series):
        if not in_dd:
            if v >= peak:
                peak = v
            elif (peak - v) / peak >= THRESH:
                in_dd, depth, start = True, (peak - v) / peak, i
        else:
            depth = max(depth, (peak - v) / peak)
            if v >= peak:
                out.append(dict(depth=depth, quarters=i - start + 1, recovered=True))
                in_dd, peak = False, v
    if in_dd:
        out.append(dict(depth=depth, quarters=len(series) - start, recovered=False))
    return out


def summarize(unit_series, label):
    all_eps, units_with = [], 0
    total_quarters = 0
    for s in unit_series:
        total_quarters += len(s)
        eps = episodes_of(s)
        if eps:
            units_with += 1
        all_eps.extend(eps)
    depths = [e["depth"] for e in all_eps]
    rec = [e for e in all_eps if e["recovered"]]
    return dict(
        panel=label, units=len(unit_series), unit_quarters=total_quarters,
        drawdown_episodes=len(all_eps),
        episodes_per_100q=100 * len(all_eps) / total_quarters,
        share_units_with_drawdown=units_with / len(unit_series),
        median_depth=float(np.median(depths)) if depths else np.nan,
        p90_depth=float(np.percentile(depths, 90)) if depths else np.nan,
        recovery_rate=len(rec) / len(all_eps) if all_eps else np.nan,
        median_recovery_quarters=float(np.median([e["quarters"] for e in rec])) if rec else np.nan,
        p90_recovery_quarters=float(np.percentile([e["quarters"] for e in rec], 90)) if rec else np.nan)


# EDGAR side
edgar = pd.read_csv(ROOT / "data/edgar_ratios.csv").dropna(subset=["revenue"])
edgar["qi"] = edgar.fiscal_period.str.split("Q").map(lambda x: int(x[0]) * 4 + int(x[1]))
edgar_series = [g.sort_values("qi").revenue.to_numpy()
                for _, g in edgar.groupby("ticker") if len(g) >= 8]

# simulator side (boardroom + oracle_v3 arms, FULL run)
mt = pd.read_csv(FULL / "primary_monthly_trace.csv",
                 usecols=["policy", "episode", "month", "mrr"])
mt["quarter"] = mt.month // 3
rows = [summarize(edgar_series, "EDGAR")]
for pol in ["boardroom", "oracle_v3"]:
    q = (mt[mt.policy == pol].groupby(["episode", "quarter"])
           .agg(qrev=("mrr", "sum"), n=("mrr", "size")).reset_index())
    q = q[q.n == 3]
    sim_series = [g.sort_values("quarter").qrev.to_numpy()
                  for _, g in q.groupby("episode") if len(g) >= 8]
    rows.append(summarize(sim_series, f"sim_{pol}"))

df = pd.DataFrame(rows)
df.to_csv(OUT / "e6_drawdown_recovery.csv", index=False)
print(df.round(3).to_string(index=False))
