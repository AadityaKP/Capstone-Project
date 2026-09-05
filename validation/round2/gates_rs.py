"""RS gates (plan section 3): read once after a3_random_shock.py finishes.

RS-1: oracle_v3 > boardroom on final MRR in >= 15/20 seeds under random
      shock timing.
RS-2: oracle_v3 - oracle_v3_no_memory paired diff on final MRR, bootstrap CI
      excluding 0.
Also: post-shock Rule-of-40 recovery rate per arm using each episode's OWN
schedule (regain pre-shock R40 within 24 months).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation/results"


def bootstrap_ci(diff, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def recovery_rate(monthly: pd.DataFrame, episodes: pd.DataFrame, policy: str):
    sub = monthly[monthly.policy == policy]
    eps = episodes[episodes.policy == policy]
    events = recovered = 0
    for row in eps.itertuples():
        m = sub[sub.seed == row.seed].sort_values("month")
        r40 = m.set_index("month").rule_of_40
        for shock_month in (row.shock_m1, row.shock_m2, row.shock_m3):
            if shock_month - 1 not in r40.index:
                continue
            pre = r40.loc[shock_month - 1]
            window = r40.loc[shock_month + 1: shock_month + 24]
            if len(window) == 0:
                continue
            events += 1
            recovered += int((window >= pre).any())
    return recovered, events


def main() -> None:
    ep = pd.read_csv(RESULTS / "a3_oracle_value_rs.csv")
    monthly = pd.read_csv(RESULTS / "a3_rs_monthly.csv")

    piv = ep.pivot(index="seed", columns="policy", values="final_mrr")
    d1 = (piv.oracle_v3 - piv.boardroom).dropna()
    wins = int((d1 > 0).sum())
    print(f"RS-1: oracle_v3 > boardroom in {wins}/{len(d1)} seeds "
          f"(need >=15): {'PASS' if wins >= 15 else 'FAIL'}")
    print(f"  paired diff: mean {d1.mean():+,.0f}, median {d1.median():+,.0f}, "
          f"Wilcoxon p={stats.wilcoxon(d1.to_numpy()).pvalue:.2g}")

    d2 = (piv.oracle_v3 - piv.oracle_v3_no_memory).dropna()
    lo, hi = bootstrap_ci(d2.to_numpy())
    rs2 = lo > 0 or hi < 0
    print(f"\nRS-2: v3 - v3_no_memory on final MRR: mean {d2.mean():+,.0f}, "
          f"95% CI [{lo:+,.0f}, {hi:+,.0f}], positive in "
          f"{int((d2 > 0).sum())}/{len(d2)} seeds: "
          f"{'PASS (CI excludes 0)' if rs2 else 'FAIL (CI includes 0)'}")

    print("\npost-shock R40 recovery within 24 months (per-episode schedule):")
    for policy in ("boardroom", "oracle_v3", "oracle_v3_no_memory"):
        rec, ev = recovery_rate(monthly, ep, policy)
        print(f"  {policy}: {rec}/{ev} = {rec / max(ev, 1):.0%}")
    print("\nsurvival:", ep.groupby("policy").survived.mean().to_dict())


if __name__ == "__main__":
    main()
