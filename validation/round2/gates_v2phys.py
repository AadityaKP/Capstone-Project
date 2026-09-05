"""Robustness gates under v2 physics (plan S3, gate table item 'Robustness').

PASS: boardroom > each baseline on final MRR, paired Holm p < 0.05 (from the
A2 v2phys run), AND oracle_v3 > boardroom in >= 15/20 seeds (A3 v2phys).
Run ONCE after both runs finish; output is pasted into LOG.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation/results"


def main() -> None:
    t = pd.read_csv(RESULTS / "statistical_tests_policy_baselines_v2phys.csv")
    fam = t[(t.arm_b == "boardroom") & (t.metric == "final_mrr")]
    a2_ok = bool((fam.holm_p_vs_boardroom < 0.05).all()
                 and (fam.mean_diff_b_minus_a > 0).all())
    print("A2 v2phys: boardroom > each baseline on final MRR (Holm p<0.05):",
          "PASS" if a2_ok else "FAIL")
    print(fam[["arm_a", "mean_diff_b_minus_a", "hedges_g_paired",
               "holm_p_vs_boardroom"]].to_string(index=False))

    a3 = pd.read_csv(RESULTS / "a3_oracle_value_v2phys.csv")
    piv = a3.pivot(index="seed", columns="policy", values="final_mrr")
    diff = (piv.oracle_v3 - piv.boardroom).dropna()
    wins = int((diff > 0).sum())
    wil = stats.wilcoxon(diff.to_numpy())
    a3_ok = wins >= 15
    print(f"\nA3 v2phys: oracle_v3 > boardroom in {wins}/{len(diff)} seeds "
          f"(need >=15): {'PASS' if a3_ok else 'FAIL'}")
    print(f"paired diff final MRR: mean {diff.mean():+,.0f}, "
          f"median {diff.median():+,.0f}, Wilcoxon p={wil.pvalue:.2g}")
    surv = a3.groupby("policy").survived.mean() if "survived" in a3 else None
    if surv is not None:
        print("survival:", surv.to_dict())
    print(f"\nRobustness gate: {'PASS' if (a2_ok and a3_ok) else 'FAIL'}")


if __name__ == "__main__":
    main()
