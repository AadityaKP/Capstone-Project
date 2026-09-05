"""S11 harvest: read once after a3_mean_revert.py finishes.

- Does the oracle advantage survive recoverable shocks? (paired, 20 seeds)
- E6 recomputation on those episodes with the SAME episodes_of code path as
  the recorded analysis -> e6_drawdown_recovery_mr.csv (compare against the
  recorded sim rows: 1.4-1.6 episodes/100 quarters, median depth ~61-63%,
  recovery ~0-2%; EDGAR: 3.0 / 11% / 85%).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation/analysis"))

from e6_drawdown_recovery import episodes_of  # noqa: E402

RESULTS = ROOT / "validation/results"


def main() -> None:
    ep = pd.read_csv(RESULTS / "a3_oracle_value_mr.csv")
    piv = ep.pivot(index="seed", columns="policy", values="final_mrr")
    diff = (piv.oracle_v3 - piv.boardroom).dropna()
    wins = int((diff > 0).sum())
    wil = stats.wilcoxon(diff.to_numpy())
    print(f"A3 mean-revert: oracle_v3 > boardroom in {wins}/{len(diff)} seeds; "
          f"mean paired diff ${diff.mean():+,.0f}, median ${diff.median():+,.0f}, "
          f"Wilcoxon p={wil.pvalue:.2g}")
    print("survival:", ep.groupby("policy").survived.mean().to_dict())
    print("median final MRR:",
          {p: f"{v:,.0f}" for p, v in
           ep.groupby("policy").final_mrr.median().to_dict().items()})

    monthly = pd.read_csv(RESULTS / "a3_mr_monthly.csv")
    monthly["quarter"] = monthly.month // 3
    rows = []
    for policy, sub in monthly.groupby("policy"):
        series_list = []
        for _, g in sub.groupby("seed"):
            q = (g.groupby("quarter").agg(qrev=("mrr", "sum"), n=("mrr", "size")))
            q = q[q.n == 3]
            if len(q) >= 8:
                series_list.append(q.qrev.to_numpy())
        eps, total_q = [], 0
        for s in series_list:
            total_q += len(s)
            eps.extend(episodes_of(s))
        depths = [e["depth"] for e in eps]
        rec = [e for e in eps if e["recovered"]]
        rows.append(dict(
            panel=f"sim_{policy}_mean_revert", units=len(series_list),
            unit_quarters=total_q, episodes=len(eps),
            episodes_per_100q=100 * len(eps) / max(total_q, 1),
            median_depth=float(np.median(depths)) if depths else np.nan,
            recovery_rate=len(rec) / max(len(eps), 1),
            median_recovery_quarters=(float(np.median([e["quarters"] for e in rec]))
                                      if rec else np.nan)))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "e6_drawdown_recovery_mr.csv", index=False)
    print("\nE6 under mean-revert (recorded sim: ~1.5/100q, depth ~62%, "
          "recovery 0-2%; EDGAR: 3.0/100q, 11%, 85%):")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
