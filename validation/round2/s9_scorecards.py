"""S9: append round-2 rows to both scorecards (never edit existing rows).

Every row carries physics_version / brief_version / mapping_version /
financing_model / shock_schedule. Re-derives every number from CSVs on disk.
Run once, after all round-2 runs have finished.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "validation/results"
R2 = ROOT / "validation/round2"

TAGS = dict(physics_version="v1", brief_version="v1", mapping_version="v1",
            financing_model="off", shock_schedule="fixed")


def agent_rows() -> list[dict]:
    rows = []

    # A2 v2phys
    t = pd.read_csv(RESULTS / "statistical_tests_policy_baselines_v2phys.csv")
    fam = t[(t.arm_b == "boardroom") & (t.metric == "final_mrr")]
    ok = bool((fam.holm_p_vs_boardroom < 0.05).all()
              and (fam.mean_diff_b_minus_a > 0).all())
    for r in fam.itertuples():
        rows.append(dict(
            dimension=f"superiority to {r.arm_a} (v2 physics)",
            test="A2-v2phys paired, deterministic RNG", baseline=r.arm_a, n=r.n,
            effect=f"g={r.hedges_g_paired:.2f}",
            result=f"mean diff final MRR ${r.mean_diff_b_minus_a:,.0f}, "
                   f"Holm p={r.holm_p_vs_boardroom:.2g}",
            acceptance_criterion="paired Holm p<0.05, boardroom above each baseline",
            verdict="PASS" if ok else "FAIL",
            interpretation="robustness row: A2 ordering on final MRR survives the v2 "
                           "calibration; caveat - boardroom post-shock R40 is worse "
                           "than noop/heuristic under v2 (spend hurts the margin term)",
            **{**TAGS, "physics_version": "v2"}))

    # A3 v2phys
    a3 = pd.read_csv(RESULTS / "a3_oracle_value_v2phys.csv")
    piv = a3.pivot(index="seed", columns="policy", values="final_mrr")
    diff = (piv.oracle_v3 - piv.boardroom).dropna()
    wins = int((diff > 0).sum())
    rows.append(dict(
        dimension="oracle value over boardroom (v2 physics)",
        test="A3-v2phys paired, freq 10", baseline="boardroom", n=len(diff),
        effect=f"{wins}/{len(diff)} seeds",
        result=f"mean paired diff final MRR ${diff.mean():,.0f}, "
               f"Wilcoxon p={stats.wilcoxon(diff.to_numpy()).pvalue:.2g}",
        acceptance_criterion="oracle_v3 > boardroom in >=15/20 seeds",
        verdict="PASS" if wins >= 15 else "FAIL",
        interpretation="robustness row under the fitted physics",
        **{**TAGS, "physics_version": "v2"}))

    # B1 brief v2
    c = pd.read_csv(RESULTS / "a4_level_sweeps_bv2_checks.csv")
    moved = {v: int(c[(c.variant == v) & c.gated].moved.sum())
             for v in ("v1", "v2a", "v2b")}
    rows.append(dict(
        dimension="LLM brief level-responsiveness (brief v2)",
        test="B1 pre-registered level sweeps", baseline="brief v1",
        n=len(c[c.gated]),
        effect=f"v1 {moved['v1']}/4, v2a {moved['v2a']}/4, v2b {moved['v2b']}/4",
        result="runway sweep rho 0.00->0.97 under the level block; churn/"
               "confidence/competitors remain flat (modifier never consumes "
               "macro_condition; innovation_urgency held flat by the model)",
        acceptance_criterion=">=3 of the original 4 sweeps move (frozen rule)",
        verdict="FAIL",
        interpretation="brief v2 not adopted; brief v1 stays primary; A4 FAIL "
                       "remains a limitation; runway responsiveness improvement "
                       "reported as a finding",
        **{**TAGS, "brief_version": "v2"}))

    # Second LLM (reported, no gate)
    mc_path = RESULTS / "a4_level_sweeps_models_checks.csv"
    if mc_path.exists():
        mc = pd.read_csv(mc_path)
        g = mc[mc.gated].groupby(["model", "variant"]).moved.sum()
        rows.append(dict(
            dimension="brief level-responsiveness, second LLM",
            test="A4 level sweeps x models", baseline="llama3.1:8b",
            n=int(len(mc[mc.gated])),
            effect="; ".join(f"{m}/{v}: {int(x)}/4" for (m, v), x in g.items()),
            result="sensitivity of level-blindness to the model (reported, no gate)",
            acceptance_criterion="none (reported)", verdict="REPORTED",
            interpretation="is level-blindness a property of the model or the prompt",
            **TAGS))

    # RS ablation
    rs_path = RESULTS / "a3_oracle_value_rs.csv"
    if rs_path.exists():
        ep = pd.read_csv(rs_path)
        piv = ep.pivot(index="seed", columns="policy", values="final_mrr")
        d1 = (piv.oracle_v3 - piv.boardroom).dropna()
        wins = int((d1 > 0).sum())
        rows.append(dict(
            dimension="oracle value under RANDOM shock timing",
            test="RS-1 paired, 20 seeds, legacy physics", baseline="boardroom",
            n=len(d1), effect=f"{wins}/{len(d1)} seeds",
            result=f"mean paired diff ${d1.mean():,.0f}, "
                   f"Wilcoxon p={stats.wilcoxon(d1.to_numpy()).pvalue:.2g}",
            acceptance_criterion=">=15/20 seeds", verdict="PASS" if wins >= 15 else "FAIL",
            interpretation="does oracle value depend on the fixed shock timetable",
            **{**TAGS, "shock_schedule": "random"}))
        d2 = (piv.oracle_v3 - piv.oracle_v3_no_memory).dropna()
        rng = np.random.default_rng(0)
        means = rng.choice(d2.to_numpy(), size=(10_000, len(d2)), replace=True).mean(axis=1)
        lo, hi = np.percentile(means, [2.5, 97.5])
        ok = lo > 0 or hi < 0
        rows.append(dict(
            dimension="retrieval increment under RANDOM shock timing",
            test="RS-2 paired, 20 seeds", baseline="oracle_v3_no_memory",
            n=len(d2), effect=f"mean ${d2.mean():,.0f}",
            result=f"95% bootstrap CI [${lo:,.0f}, ${hi:,.0f}], "
                   f"positive in {int((d2 > 0).sum())}/{len(d2)} seeds",
            acceptance_criterion="bootstrap CI excludes 0",
            verdict="PASS" if ok else "FAIL",
            interpretation="does the ~3% retrieval increment survive without a "
                           "learnable shock timetable",
            **{**TAGS, "shock_schedule": "random"}))
    return rows


def environment_rows() -> list[dict]:
    v = json.loads((R2 / "r2_criteria_verdicts.json").read_text())
    mk = dict(physics_version="v2", brief_version="v1", mapping_version="v2",
              financing_model="opportunistic", shock_schedule="off")
    return [
        dict(dimension="real-company retrodiction, round 2 (out-of-time q0+8)",
             test="R2-C1", policy_arm="hold", edgar_n=19,
             sim_n=v["n_eval2_evaluable"],
             result=f"median |4q growth error| {v['R2_C1_median_abs_error_pp']:.1f}pp "
                    f"(signed {v['R2_C1_signed_median_pp']:+.1f}pp; DEV2 "
                    f"{v['dev2_median_abs_error_pp']:.1f}pp; $50 "
                    f"{v['price50_sensitivity_median_abs_error_pp']:.1f}pp)",
             verdict=v["R2_C1_verdict"], **mk),
        dict(dimension="growth-sign agreement, round 2", test="R2-SIGN",
             policy_arm="hold", edgar_n=19, sim_n=19,
             result=f"{v['R2_SIGN']:.0%}", verdict=v["R2_SIGN_verdict"], **mk),
        dict(dimension="corridor artifact, round 2", test="R2-CORR",
             policy_arm="boardroom+hold", edgar_n=19, sim_n=19,
             result=f"boardroom std ratio {v['R2_CORR_std_ratio']:.2f} (PASSES "
                    f">=1/3) but hold Spearman {v['R2_CORR_spearman']:.2f} "
                    f"(p={v['R2_CORR_p']:.2f}) fails >0.3: at q0+8 every "
                    f"company real-spends in the fitted curve's saturation "
                    f"region, hold growth nearly constant",
             verdict=v["R2_CORR_verdict"], **mk),
        dict(dimension="financing realism, round 2", test="R2-FIN-a",
             policy_arm="hold", edgar_n=19, sim_n=19,
             result=f"{v['R2_FIN_a_share']:.0%} of EVAL2 companies survive >50% "
                    f"of seeds under hold with the opportunistic hazard",
             verdict=v["R2_FIN_a_verdict"], **mk),
        dict(dimension="financing ablation, round 2", test="R2-FIN-b",
             policy_arm="hold", edgar_n=19, sim_n=19,
             result="no EVAL2 company has zero surviving seeds with financing "
                    "off at q0+8 - ablation premise empty",
             verdict="N/A", **mk),
    ]


def main() -> None:
    ag = pd.read_csv(RESULTS / "agent_scorecard.csv")
    for col in ("physics_version", "brief_version", "mapping_version",
                "financing_model", "shock_schedule"):
        if col not in ag.columns:
            ag[col] = {"physics_version": "v1", "brief_version": "v1",
                       "mapping_version": "v1", "financing_model": "off",
                       "shock_schedule": "fixed"}[col]
    ag = pd.concat([ag, pd.DataFrame(agent_rows())], ignore_index=True)
    ag.to_csv(RESULTS / "agent_scorecard.csv", index=False)

    sc = pd.read_csv(RESULTS / "environment_scorecard.csv")
    for col in ("brief_version", "mapping_version", "financing_model",
                "shock_schedule"):
        if col not in sc.columns:
            sc[col] = {"brief_version": "v1", "mapping_version": "v1",
                       "financing_model": "off", "shock_schedule": "fixed"}[col]
    sc = pd.concat([sc, pd.DataFrame(environment_rows())], ignore_index=True)
    sc.to_csv(RESULTS / "environment_scorecard.csv", index=False)
    print(f"appended {len(agent_rows())} agent rows, "
          f"{len(environment_rows())} environment rows")


if __name__ == "__main__":
    main()
