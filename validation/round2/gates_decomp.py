"""Addendum A gates (PROTOCOL_addendum_A.md): read once, in the morning,
after the overnight queue finishes. Output is pasted into LOG.md; the frozen
interpretation rules in the addendum are then applied literally.

Written and committed BEFORE any addendum result exists (S12).

D-a/D-b/D-c/D-d/L-1: the oracle arm is paired against the arm named in the
addendum table on the same seeds - win count, paired mean/median diff on
final MRR, bootstrap 95% CI, Wilcoxon p, survival, post-shock Rule-of-40 -
PASS if the oracle arm beats its paired boardroom on final MRR in >= 15/20
seeds. RS-2x: v3 - v3_no_memory paired diff on final MRR - the recorded
n=20 CI, the extension-seeds CI, and the pooled n=40 CI.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation/results"

# (id, question, decomp csv, test label, baseline csv, baseline label)
ARMS = [
    ("D-a", "does the brief add value without the spend multipliers?",
     "a3_decomp_da.csv", "oracle_v3_no_modifier",
     "a3_oracle_value_v2phys.csv", "boardroom"),
    ("D-b", "is the loss caused by spend-up past the corridor?",
     "a3_decomp_db.csv", "oracle_v3_tier_bound",
     "a3_oracle_value_v2phys.csv", "boardroom"),
    ("D-c", "does oracle value under fitted acquisition depend on shock "
     "recoverability?",
     "a3_decomp_dc.csv", "oracle_v3_mr",
     "a3_decomp_dc.csv", "boardroom_mr"),
    ("D-d", "is the null LLM-specific? (qwen2.5:7b-instruct, v2 physics)",
     "a3_decomp_dd.csv", "oracle_v3_qwen",
     "a3_oracle_value_v2phys.csv", "boardroom"),
    ("L-1", "does the headline result depend on the LLM? (qwen, legacy)",
     "a3_decomp_l1.csv", "oracle_v3_qwen_legacy",
     "a3/episodes_boardroom.csv", "boardroom"),
]


def bootstrap_ci(diff, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load(results: Path, name: str) -> pd.DataFrame | None:
    path = results / name
    if not path.exists():
        print(f"MISSING (run incomplete): {name}")
        return None
    return pd.read_csv(path)


def survived(df: pd.DataFrame) -> pd.Series:
    if "survived" in df.columns:
        return df.survived.astype(int)
    # run_simulation episode rows carry cause instead of a survived column
    return (df.cause == "Time Limit").astype(int)


def wilcoxon_p(diff: np.ndarray) -> float:
    return 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(diff).pvalue)


def arm_gate(results: Path, arm_id: str, question: str, test_csv: str,
             test_label: str, base_csv: str, base_label: str) -> None:
    print(f"\n{arm_id} ({question})")
    test, base = load(results, test_csv), load(results, base_csv)
    if test is None or base is None:
        return
    test = test[test.policy == test_label]
    base = base[base.policy == base_label]
    if test.empty or base.empty:
        print(f"  MISSING arm rows: {test_label if test.empty else base_label}")
        return
    both = pd.concat([test, base], ignore_index=True)
    piv = both.pivot(index="seed", columns="policy", values="final_mrr")
    diff = (piv[test_label] - piv[base_label]).dropna()
    wins = int((diff > 0).sum())
    lo, hi = bootstrap_ci(diff.to_numpy())
    print(f"  {test_label} > {base_label} on final MRR in {wins}/{len(diff)} "
          f"seeds (need >=15): {'PASS' if wins >= 15 else 'FAIL'}")
    print(f"  paired diff: mean {diff.mean():+,.0f}, median {diff.median():+,.0f}, "
          f"95% CI [{lo:+,.0f}, {hi:+,.0f}], "
          f"Wilcoxon p={wilcoxon_p(diff.to_numpy()):.2g}")
    print(f"  survival: {test_label} {survived(test).mean():.0%}, "
          f"{base_label} {survived(base).mean():.0%}")
    if "post_shock_avg_rule40_25_60" in both.columns:
        r40 = both.pivot(index="seed", columns="policy",
                         values="post_shock_avg_rule40_25_60")
        r40d = (r40[test_label] - r40[base_label]).dropna()
        print(f"  post-shock R40 (mean months 25-60): {test_label} "
              f"{r40[test_label].mean():.1f}, {base_label} "
              f"{r40[base_label].mean():.1f}, paired diff {r40d.mean():+.1f}")
    if "recovered_shock_rate_pct" in both.columns:
        rec = both.groupby("policy").recovered_shock_rate_pct.mean()
        print(f"  recovered-shock rate: {test_label} {rec[test_label]:.0f}%, "
              f"{base_label} {rec[base_label]:.0f}%")


def rs_diff(df: pd.DataFrame) -> pd.Series | None:
    # None when an arm is absent entirely (a partial extension file keeps
    # only whole arms per a3_rs_ext.py's resume contract, but a mid-arm
    # crash still leaves the earlier arm alone in the CSV).
    piv = df.pivot(index="seed", columns="policy", values="final_mrr")
    if not {"oracle_v3", "oracle_v3_no_memory"} <= set(piv.columns):
        return None
    return (piv["oracle_v3"] - piv["oracle_v3_no_memory"]).dropna()


def rs_line(tag: str, d: pd.Series) -> None:
    lo, hi = bootstrap_ci(d.to_numpy())
    print(f"  {tag}: n={len(d)}, mean {d.mean():+,.0f}, median "
          f"{d.median():+,.0f}, 95% CI [{lo:+,.0f}, {hi:+,.0f}], positive in "
          f"{int((d > 0).sum())}/{len(d)}, "
          f"Wilcoxon p={wilcoxon_p(d.to_numpy()):.2g}")


def main(results: Path = RESULTS) -> None:
    for arm in ARMS:
        arm_gate(results, *arm)

    print("\nRS-2x (retrieval increment under random shock timing)")
    rec = load(results, "a3_oracle_value_rs.csv")
    ext = load(results, "a3_oracle_value_rs_ext.csv")
    d_rec = rs_diff(rec) if rec is not None else None
    d_ext = rs_diff(ext) if ext is not None else None
    if rec is not None and d_rec is None:
        print("  MISSING arm rows in a3_oracle_value_rs.csv")
    if ext is not None and d_ext is None:
        print("  MISSING arm rows in a3_oracle_value_rs_ext.csv "
              "(arm incomplete)")
    if d_rec is not None:
        rs_line("recorded seeds 0-19", d_rec)
    if d_ext is not None:
        rs_line("extension seeds 21-40", d_ext)
    if d_rec is not None and d_ext is not None:
        pooled = pd.concat([d_rec, d_ext])
        rs_line("pooled (target n=40)", pooled)
        if len(d_rec) == 20 and len(d_ext) == 20:
            lo, hi = bootstrap_ci(pooled.to_numpy())
            print("  verdict: "
                  + ("small but detectable at n=40 (CI excludes 0)"
                     if lo > 0 or hi < 0
                     else "not detectable at n=40 (CI includes 0)"))
        else:
            print(f"  INCOMPLETE: pooled n={len(pooled)} < 40 - verdict "
                  "deferred until the extension finishes")

    print("\nApply the frozen interpretation rules in "
          "validation/calibration/PROTOCOL_addendum_A.md literally; "
          "scorecard rows are appended in S13 (never edit existing rows).")


if __name__ == "__main__":
    main()
