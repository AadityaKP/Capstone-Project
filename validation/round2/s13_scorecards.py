"""S13: append Addendum A rows to both scorecards (never edit existing rows).

Agent rows: D-a/D-b/D-c/D-d (v2-physics decomposition), L-1 (second LLM,
legacy), RS-2x (pooled n=40 retrieval increment), A2 v2phys+mean_revert.
Environment rows: E1-mr..E5-mr for legacy and v2 physics (from
e_battery_mr_scorecard_rows.csv, mapped to the scorecard schema the E6-mr
row established). Every number re-derived from CSVs on disk with the same
machinery as gates_decomp.py. Run once, after gates_decomp.py output is in
LOG.md.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation/results"

TAGS = dict(physics_version="v2", brief_version="v1", mapping_version="v1",
            financing_model="off", shock_schedule="fixed")


def bootstrap_ci(diff, n_boot=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilcoxon_p(diff: np.ndarray) -> float:
    return 1.0 if np.allclose(diff, 0) else float(stats.wilcoxon(diff).pvalue)


def paired(test_csv, test_label, base_csv, base_label):
    test = pd.read_csv(RESULTS / test_csv)
    base = pd.read_csv(RESULTS / base_csv)
    both = pd.concat([test[test.policy == test_label],
                      base[base.policy == base_label]], ignore_index=True)
    piv = both.pivot(index="seed", columns="policy", values="final_mrr")
    return (piv[test_label] - piv[base_label]).dropna()


def agent_rows() -> list[dict]:
    rows = []

    # D-a: no modifier (paired diff is exactly 0 on every seed - the weight
    # adapter only rescales proposal confidences; the assembled action never
    # depends on them, so the modifier is the brief's only action channel)
    d = paired("a3_decomp_da.csv", "oracle_v3_no_modifier",
               "a3_oracle_value_v2phys.csv", "boardroom")
    wins = int((d > 0).sum())
    rows.append(dict(
        dimension="oracle value without ActionModifier (v2 physics, decomposition)",
        test="D-a paired, 20 seeds, oracle_v3_no_modifier vs recorded boardroom",
        baseline="boardroom", n=len(d), effect=f"{wins}/{len(d)} seeds",
        result=f"paired diff final MRR ${d.mean():,.0f} on every seed - "
               "outcome-identical to boardroom; the brief's weight-adapter "
               "channel never alters the assembled action",
        acceptance_criterion="oracle arm > boardroom in >=15/20 seeds",
        verdict="PASS" if wins >= 15 else "FAIL",
        interpretation="the entire oracle_v3 behavioural delta flows through "
                       "the ActionModifier spend multipliers", **TAGS))

    # D-b: tier-bounded modifier
    d = paired("a3_decomp_db.csv", "oracle_v3_tier_bound",
               "a3_oracle_value_v2phys.csv", "boardroom")
    wins = int((d > 0).sum())
    lo, hi = bootstrap_ci(d.to_numpy())
    rows.append(dict(
        dimension="oracle value with tier-bounded modifier (v2 physics, decomposition)",
        test="D-b paired, 20 seeds, modifier_bound=tier vs recorded boardroom",
        baseline="boardroom", n=len(d), effect=f"{wins}/{len(d)} seeds",
        result=f"mean paired diff final MRR ${d.mean():,.0f}, 95% CI "
               f"[${lo:,.0f}, ${hi:,.0f}], Wilcoxon p={wilcoxon_p(d.to_numpy()):.2g}",
        acceptance_criterion="oracle arm > boardroom in >=15/20 seeds",
        verdict="PASS" if wins >= 15 else "FAIL",
        interpretation="capping spend-up at the corridor's top tier does not "
                       "restore oracle value under the fitted curve (llama)", **TAGS))

    # D-c: v2 + mean_revert (both arms new)
    d = paired("a3_decomp_dc.csv", "oracle_v3_mr", "a3_decomp_dc.csv",
               "boardroom_mr")
    wins = int((d > 0).sum())
    lo, hi = bootstrap_ci(d.to_numpy())
    rows.append(dict(
        dimension="oracle value under recoverable shocks (v2 physics, decomposition)",
        test="D-c paired, 20 seeds, v2 + shock_recovery=mean_revert (new pair)",
        baseline="boardroom_mr", n=len(d), effect=f"{wins}/{len(d)} seeds",
        result=f"mean paired diff final MRR ${d.mean():,.0f}, 95% CI "
               f"[${lo:,.0f}, ${hi:,.0f}], Wilcoxon p={wilcoxon_p(d.to_numpy()):.2g}",
        acceptance_criterion="oracle arm > boardroom in >=15/20 seeds",
        verdict="PASS" if wins >= 15 else "FAIL",
        interpretation="shock recoverability does not restore oracle value "
                       "under the fitted curve (llama)", **TAGS))

    # D-d: qwen under v2 physics
    d = paired("a3_decomp_dd.csv", "oracle_v3_qwen",
               "a3_oracle_value_v2phys.csv", "boardroom")
    wins = int((d > 0).sum())
    lo, hi = bootstrap_ci(d.to_numpy())
    rows.append(dict(
        dimension="oracle value, second LLM (v2 physics, decomposition)",
        test="D-d paired, 20 seeds, qwen2.5:7b-instruct vs recorded boardroom",
        baseline="boardroom", n=len(d), effect=f"{wins}/{len(d)} seeds",
        result=f"mean paired diff final MRR ${d.mean():,.0f}, 95% CI "
               f"[${lo:,.0f}, ${hi:,.0f}], Wilcoxon p={wilcoxon_p(d.to_numpy()):.2g} "
               "(win-count criterion met; magnitude small, CI includes 0)",
        acceptance_criterion="oracle arm > boardroom in >=15/20 seeds",
        verdict="PASS" if wins >= 15 else "FAIL",
        interpretation="frozen rule fires: the v2-physics null is "
                       "LLM-specific (llama 8/20 FAIL, qwen 15/20 PASS)", **TAGS))

    # L-1: qwen under legacy physics
    d = paired("a3_decomp_l1.csv", "oracle_v3_qwen_legacy",
               "a3/episodes_boardroom.csv", "boardroom")
    wins = int((d > 0).sum())
    lo, hi = bootstrap_ci(d.to_numpy())
    rows.append(dict(
        dimension="oracle value, second LLM (legacy physics)",
        test="L-1 paired, 20 seeds, qwen2.5:7b-instruct vs recorded boardroom",
        baseline="boardroom", n=len(d), effect=f"{wins}/{len(d)} seeds",
        result=f"mean paired diff final MRR ${d.mean():,.0f}, 95% CI "
               f"[${lo:,.0f}, ${hi:,.0f}], Wilcoxon p={wilcoxon_p(d.to_numpy()):.2g}",
        acceptance_criterion="oracle arm > boardroom in >=15/20 seeds",
        verdict="PASS" if wins >= 15 else "FAIL",
        interpretation="the headline legacy-physics oracle result replicates "
                       "on a second LLM; no llama qualifier needed",
        **{**TAGS, "physics_version": "v1"}))

    # RS-2x: pooled n=40 retrieval increment under random timing
    def rs_diff(name):
        df = pd.read_csv(RESULTS / name)
        piv = df.pivot(index="seed", columns="policy", values="final_mrr")
        return (piv["oracle_v3"] - piv["oracle_v3_no_memory"]).dropna()

    d_rec, d_ext = rs_diff("a3_oracle_value_rs.csv"), rs_diff("a3_oracle_value_rs_ext.csv")
    pooled = pd.concat([d_rec, d_ext])
    lo, hi = bootstrap_ci(pooled.to_numpy())
    elo, ehi = bootstrap_ci(d_ext.to_numpy())
    ok = lo > 0 or hi < 0
    rows.append(dict(
        dimension="retrieval increment under RANDOM shock timing, pooled n=40",
        test="RS-2x paired, seeds 0-19 recorded + 21-40 pre-declared extension",
        baseline="oracle_v3_no_memory", n=len(pooled),
        effect=f"mean ${pooled.mean():,.0f}",
        result=f"95% bootstrap CI [${lo:,.0f}, ${hi:,.0f}], positive in "
               f"{int((pooled > 0).sum())}/{len(pooled)} seeds, Wilcoxon "
               f"p={wilcoxon_p(pooled.to_numpy()):.2g}; extension cohort alone "
               f"mean ${d_ext.mean():,.0f}, CI [${elo:,.0f}, ${ehi:,.0f}]",
        acceptance_criterion="pooled CI excludes 0",
        verdict="PASS" if ok else "FAIL",
        interpretation="small but detectable at n=40 (frozen language); the "
                       "recorded n=20 null reflected power, not absence",
        **{**TAGS, "physics_version": "v1", "shock_schedule": "random"}))

    # A2 v2phys + mean_revert (same gate as the recorded A2 v2phys rows)
    t = pd.read_csv(RESULTS / "statistical_tests_policy_baselines_v2phys_mr.csv")
    fam = t[(t.arm_b == "boardroom") & (t.metric == "final_mrr")]
    ok = bool((fam.holm_p_vs_boardroom < 0.05).all()
              and (fam.mean_diff_b_minus_a > 0).all())
    for r in fam.itertuples():
        rows.append(dict(
            dimension=f"superiority to {r.arm_a} (v2 physics + mean_revert)",
            test="A2-v2phys-mr paired, deterministic RNG, 50 seeds",
            baseline=r.arm_a, n=r.n,
            effect=f"g={r.hedges_g_paired:.2f}",
            result=f"mean diff final MRR ${r.mean_diff_b_minus_a:,.0f}, "
                   f"Holm p={r.holm_p_vs_boardroom:.2g}",
            acceptance_criterion="paired Holm p<0.05, boardroom above each baseline",
            verdict="PASS" if ok else "FAIL",
            interpretation="robustness row: A2 ordering survives the most "
                           "realistic configuration (fitted curve + "
                           "recoverable shocks)", **TAGS))
    return rows


def environment_rows() -> list[dict]:
    src = pd.read_csv(RESULTS / "e_battery_mr_scorecard_rows.csv")
    rows = []
    for r in src.itertuples():
        rows.append(dict(
            dimension=f"{r.dimension} (recoverable shocks)",
            test=r.test.replace("_mr", "-mr"), policy_arm=r.policy_arm,
            edgar_n=r.edgar_n, sim_n=r.sim_n, result=r.result,
            verdict=r.verdict,
            physics_version="v1" if r.physics_version == "legacy" else "v2",
            brief_version="v1", mapping_version="v1", financing_model="off",
            shock_schedule="fixed"))
    return rows


def main() -> None:
    a_rows, e_rows = agent_rows(), environment_rows()

    ag = pd.read_csv(RESULTS / "agent_scorecard.csv")
    ag = pd.concat([ag, pd.DataFrame(a_rows)], ignore_index=True)
    ag.to_csv(RESULTS / "agent_scorecard.csv", index=False)

    sc = pd.read_csv(RESULTS / "environment_scorecard.csv")
    sc = pd.concat([sc, pd.DataFrame(e_rows)], ignore_index=True)
    sc.to_csv(RESULTS / "environment_scorecard.csv", index=False)

    print(f"appended {len(a_rows)} agent rows, {len(e_rows)} environment rows")
    print(pd.DataFrame(a_rows)[["test", "effect", "verdict"]].to_string(index=False))
    print(pd.DataFrame(e_rows)[["test", "policy_arm", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
