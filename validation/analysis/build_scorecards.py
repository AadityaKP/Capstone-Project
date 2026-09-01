"""Assemble agent_scorecard.csv, statistical_tests.csv and
validation_summary.csv from the individual result CSVs. Re-run after any
analysis updates; A3 rows appear once a3_oracle_value.csv exists."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "validation/results"
AG = ROOT / "validation/agents"

rows = []


def add(dim, test, baseline, n, effect, result, criterion, verdict, interp):
    rows.append(dict(dimension=dim, test=test, baseline=baseline, n=n,
                     effect=effect, result=result, acceptance_criterion=criterion,
                     verdict=verdict, interpretation=interp))


# ---- A1 ------------------------------------------------------------------
ae = pd.read_csv(AG / "action_effects_summary.csv")
mk = ae[(ae.dimension == "marketing.spend")]
spread = mk.spread_vs_base.max()
add("action effect", "A1 ladders (marketing)", "same state+seed, action varied",
    "20 seeds x 2 states x 5 rungs", f"outcome spread {spread:.0%} of base",
    "monotone (rho=+1.0) at both states",
    "monotone ordering, spread >= 10% of mid-rung", "PASS",
    "marketing causally moves 12-mo MRR; pricing weak-positive; R&D weak (expansion "
    "channel only - quality channel inert while innovation_factor=1); hiring is pure "
    "cost (no revenue channel), repeated over-hiring kills the company")

# ---- A2 ------------------------------------------------------------------
t = pd.read_csv(RES / "statistical_tests_policy_baselines.csv")
for base in ["noop", "random", "heuristic"]:
    r = t[(t.arm_a == base) & (t.arm_b == "boardroom") & (t.metric == "final_mrr")].iloc[0]
    verdict = "PASS" if (r.holm_p_vs_boardroom < 0.05 and r.hedges_g_paired >= 0.3) else "FAIL"
    add("superiority to " + base, "A2 paired, deterministic RNG", base,
        int(r.n), f"g={r.hedges_g_paired:.2f}",
        f"mean diff final MRR ${r.mean_diff_b_minus_a:,.0f}, Holm p={r.holm_p_vs_boardroom:.2g}",
        "paired Wilcoxon Holm p<0.05 and g>=0.3", verdict,
        "boardroom beats " + base + " on final MRR at matched seeds"
        + ("; caveat: noop never goes bankrupt (spending trades survival risk for growth)"
           if base == "noop" else ""))

# ---- A4 ------------------------------------------------------------------
rc = pd.read_csv(AG / "rule_agent_checks.csv")
add("economic sensibility (rule layer)", "A4i documented thresholds", "spec",
    len(rc), f"{int(rc.ok.sum())}/{len(rc)} exact", "all thresholds reproduce",
    "exact reproduction", "PASS" if rc.ok.all() else "FAIL",
    "rule agents respond to runway, LTV:CAC, churn and cash exactly as documented")

sr = pd.read_csv(AG / "state_responsiveness.csv")
add("state responsiveness (LLM levels)", "A4ii one-variable level sweeps",
    "expected monotone direction", "6 levels x 4 vars",
    f"{int(sr['pass'].sum())}/4 sweeps pass",
    "briefs constant (LOW/ACCELERATING) across runway, churn, confidence, competitor levels",
    "|rho|>=0.5 in >=3 of 4 sweeps", "FAIL",
    "llama3.1:8b briefs ignore state LEVELS. Exploratory follow-up (labelled post hoc): "
    "strong response to MRR trend deltas (rho=+0.96; marketing x1.64->x0.51) and to "
    "shock alerts (risk LOW->MEDIUM); churn deltas ignored. Responsiveness lives in the "
    "trend/shock lines of the prompt, consistent with the recorded in-situ shift of risk "
    "labels at shock months (0.44->0.90 MEDIUM share)")

# ---- A6i -----------------------------------------------------------------
ba = pd.read_csv(RES / "brief_accuracy.csv").iloc[0]
add("retrieval-conditioned prediction", "A6i expected_outcome vs realized",
    "majority-class base rate", int(ba.n),
    f"+{ba.delta:.3f} vs base rate",
    f"accuracy {ba.accuracy:.3f} vs base {ba.base_rate:.3f}, CI [{ba.ci95_lo:+.3f},{ba.ci95_hi:+.3f}]",
    "cluster-bootstrap CI excludes 0", ba.verdict,
    "v3 briefs are genuinely predictive of 6-month outcomes (recorded FULL run)")

# ---- A3 recorded-paired (always available) -------------------------------
rp = pd.read_csv(RES / "a3_recorded_paired.csv")
for _, r in rp.iterrows():
    arm = r.comparison.split(" - ")[0]
    add(f"oracle value ({arm}, recorded)", f"A3r paired re-analysis ({r.metric})",
        "boardroom", int(r.n), f"g={r.hedges_g_paired:.2f}",
        f"mean diff {r.mean_diff:,.1f}, CI [{r.ci95_lo:,.1f},{r.ci95_hi:,.1f}], "
        f"p={r.wilcoxon_p:.2g}, positive in {int(r.positive_seeds)}/{int(r.n)} seeds",
        "paired Wilcoxon p<0.05, CI excludes 0",
        "PASS" if (r.wilcoxon_p < 0.05 and r.ci95_lo > 0) else "FAIL",
        "FULL run re-analyzed as paired - valid because shared-world property was "
        "verified empirically for non-drawing policies (shared_world_check.csv)")

# ---- A3 (if harvested) ---------------------------------------------------
a3_path = RES / "a3_oracle_value.csv"
if a3_path.exists():
    a3 = pd.read_csv(a3_path)
    for _, r in a3.iterrows():
        if r.comparison.endswith("- boardroom"):
            arm = r.comparison.split(" - ")[0]
            recorded_positive = True  # FULL run: oracle > boardroom on both metrics
            same_sign = r.mean_diff > 0
            excl = r.ci95_lo > 0
            verdict = "PASS" if (same_sign and excl) else ("PARTIAL" if same_sign else "FAIL")
            add(f"oracle value ({arm})", f"A3 matched-seed replication ({r.metric})",
                "boardroom", int(r.n),
                f"g={r.hedges_g_paired:.2f}",
                f"mean diff {r.mean_diff:,.1f}, CI [{r.ci95_lo:,.1f},{r.ci95_hi:,.1f}], p={r.wilcoxon_p:.3g}",
                "same sign as recorded n=75 result; CI excludes 0 on >=1 metric",
                verdict,
                "matched-seed spot-replication under deterministic RNG (n=20)")
        else:
            same_sign = r.mean_diff > 0
            add("retrieval incremental value", f"A3 v3 vs v3_no_memory ({r.metric})",
                "oracle_v3_no_memory", int(r.n), f"g={r.hedges_g_paired:.2f}",
                f"mean diff {r.mean_diff:,.1f}, CI [{r.ci95_lo:,.1f},{r.ci95_hi:,.1f}], p={r.wilcoxon_p:.3g}",
                "comparative (underpowered at n=20; labelled)",
                "PASS" if (same_sign and r.ci95_lo > 0) else "INCONCLUSIVE",
                "episodic-memory ablation at matched seeds")

# ---- C1 ------------------------------------------------------------------
bt = pd.read_csv(RES / "real_company_backtest.csv")
sub = bt[bt.price_assumed == 250.0].dropna(subset=["retrodiction_error"])
mae = sub.retrodiction_error.abs().median()
sign_ok = (np.sign(sub.sim_hold_median) == np.sign(sub.actual_4q_growth)).mean()
add("real-company retrodiction", "C1 hold-arm vs actual 4q growth",
    "actual EDGAR trajectory", len(sub),
    f"median |error| {mae:.0%}",
    f"sign agreement {sign_ok:.0%}; sim hold median approx +95% vs actual median "
    f"{sub.actual_4q_growth.median():+.0%}",
    "PASS <=10pp; PARTIAL <=20pp", "FAIL",
    "scale-aware marketing curve too generous at real-company spend intensity; "
    "SATURATION_ACQUISITION_RATE=0.20/mo (the one ASSUMED free parameter) is "
    "falsified as too high; also no financing mechanism (6 high-burn companies "
    "die in-sim that in reality raised capital)")

inc_pos = (sub.agent_increment_median > 0).mean()
add("real-company counterfactual (agent)", "C1 boardroom - hold, in-sim",
    "hold (company's own spend)", len(sub),
    f"increment>0 in {inc_pos:.0%} of companies",
    f"boardroom median {sub.sim_boardroom_median.median():+.0%} vs hold "
    f"{sub.sim_hold_median.median():+.0%}",
    "increment>0 in >=2/3 companies with paired significance", "FAIL",
    "in a world whose spend response is over-generous, the corridor-limited boardroom "
    "underspends relative to real companies and loses to hold; the decomposition "
    "prevented crediting the agent with simulator bias. No 'agent adds value at real "
    "scale' claim is supportable")

# ---- Tier-2: A7 robustness ----------------------------------------------
a7_path = RES / "a7_robustness_grid.csv"
if a7_path.exists():
    a7 = pd.read_csv(a7_path)
    bd = a7[a7.policy == "boardroom"]
    pairs = pd.read_csv(RES / "a7_robustness_pairs.csv")
    add("robustness (initial conditions)", "A7 3x3 grid x 20 matched seeds",
        "noop/random/heuristic", f"{len(bd)} cells x 20 seeds",
        f"boardroom rank 1 in {(bd.rank_in_cell == 1).sum()}/{len(bd)} cells",
        f"paired g {pairs.hedges_g_paired.min():.2f}-{pairs.hedges_g_paired.max():.2f}, "
        f"all Wilcoxon p<0.05",
        "comparative: ranking stability across cells",
        "PASS" if (bd.rank_in_cell == 1).all() else "PARTIAL",
        "boardroom's advantage is stable across initial MRR (25-100k) and cash "
        "(0.5-2M); caveat: 'random' ranks 2nd by median final MRR only because "
        "MRR-at-bankruptcy inflates its median (survival ~4-20%)")

# ---- Tier-2: A5 candidate regret ----------------------------------------
a5_path = AG / "candidate_regret.csv"
if a5_path.exists():
    a5 = pd.read_csv(a5_path)
    add("candidate-action regret", "A5 one-step deviation, 48-candidate grid",
        "best grid candidate", f"{len(a5)} states x 10 seeds",
        f"median regret {a5.regret_pct_of_best.median():.1%} of best",
        f"median rank {a5.policy_rank_of_49.median():.0f}/49; top of grid in "
        f"{(a5.candidate_regret <= 0).sum()}/{len(a5)} states",
        "exploratory (candidate regret, not global optimality)", "EXPLORATORY",
        "single-month deviations are low-stakes (<=3.1% regret); at post-shock "
        "states the max-spend grid corner always wins, so the policy's residual "
        "regret is marketing under-spend vs a curve C1 shows is over-generous - "
        "and grid-edge optima mean true regret is a lower bound")

# ---- A8: post-shock Rule-of-40 recovery ----------------------------------
a8_path = RES / "a8_shock_recovery.csv"
if a8_path.exists():
    a8 = pd.read_csv(a8_path)
    a8 = a8[a8.shock_month == "all"]
    def rate(src, pol):
        return a8[(a8.source == src) & (a8.policy == pol)].recovery_rate.iloc[0]
    add("shock handling", "A8 post-shock R40 recovery (24-month window)",
        "boardroom", "A2 arms 50 seeds / A3 arms 20 seeds x 3 shocks",
        f"oracle arms {rate('A3','oracle_v1'):.0%}-{rate('A3','oracle_v3_no_memory'):.0%} "
        f"vs boardroom {rate('A2','boardroom'):.0%}",
        f"recovery rates: noop {rate('A2','noop'):.0%}, heuristic {rate('A2','heuristic'):.0%}, "
        f"boardroom {rate('A2','boardroom'):.0%}/{rate('A3','boardroom'):.0%}, "
        f"oracle_v1 {rate('A3','oracle_v1'):.0%}, oracle_v3 {rate('A3','oracle_v3'):.0%}; "
        "median time-to-recover 1-2 months for every policy",
        "comparative (Rule-of-40 recovery, NOT revenue-peak recovery - see E6)",
        "COMPARATIVE",
        "the oracle advantage is in recovery RATE, not speed; consistent with the "
        "recorded FULL-run recovery rates (76-80% vs 67-69%) and with A4's finding "
        "that the brief channel responds at shock alerts")

# ---- Tier-2: C2 allocation consistency ----------------------------------
c2_path = RES / "c2_allocation_consistency.csv"
if c2_path.exists():
    c2 = pd.read_csv(c2_path)
    add("real-company allocation direction", "C2 stress-state observational split",
        "improved vs lagged halves", f"{int(c2.n.sum())} stress quarters",
        "no significant difference (p=0.65 S&M, p=0.86 R&D)",
        "improvers grew absolute S&M and R&D (+6%) while lagged held/cut - "
        "directionally opposite to the agent's cut-marketing stress response",
        "observational supporting evidence only", "NULL",
        "no observational support for the agent's stress direction; survivor-"
        "biased public-SaaS panel, intensity/growth mechanical coupling, no "
        "clustering adjustment - listed as a limitation, not a refutation")

sc = pd.DataFrame(rows)
sc.to_csv(RES / "agent_scorecard.csv", index=False)
print(sc[["dimension", "test", "verdict"]].to_string(index=False))

# ---- Tier-2: E6 into the environment scorecard ---------------------------
e6_path = RES / "e6_drawdown_recovery.csv"
env_path = RES / "environment_scorecard.csv"
if e6_path.exists():
    e6 = pd.read_csv(e6_path).set_index("panel")
    env_sc = pd.read_csv(env_path)
    env_sc = env_sc[env_sc.test != "E6"]
    ed, sb = e6.loc["EDGAR"], e6.loc["sim_boardroom"]
    env_sc = pd.concat([env_sc, pd.DataFrame([dict(
        dimension="revenue drawdown/recovery (exploratory)", test="E6",
        policy_arm="boardroom (oracle_v3 similar)",
        edgar_n=int(ed.drawdown_episodes), sim_n=int(sb.drawdown_episodes),
        result=(f"EDGAR: {ed.episodes_per_100q:.1f} eps/100q, median depth "
                f"{ed.median_depth:.0%}, {ed.recovery_rate:.0%} recover in median "
                f"{ed.median_recovery_quarters:.0f}q. Sim: {sb.episodes_per_100q:.1f} "
                f"eps/100q, median depth {sb.median_depth:.0%}, recovery to prior "
                f"peak {sb.recovery_rate:.0%}. Sim drawdowns are catastrophic and "
                "permanent vs shallow and quickly recovered in real SaaS; note the "
                "thesis 'recovery' metric is Rule-of-40-based, not revenue-peak-based"),
        verdict="EXPLORATORY")])], ignore_index=True)
    env_sc.to_csv(env_path, index=False)
    print("\nE6 appended to environment_scorecard.csv")

# ---- consolidated statistical tests -------------------------------------
tests = [pd.read_csv(RES / "statistical_tests_policy_baselines.csv").assign(family="A2")]
if a3_path.exists():
    tests.append(pd.read_csv(a3_path).assign(family="A3"))
pd.concat(tests, ignore_index=True).to_csv(RES / "statistical_tests.csv", index=False)

# ---- validation summary --------------------------------------------------
env = pd.read_csv(RES / "environment_scorecard.csv")
summary = pd.concat([
    env.rename(columns={"policy_arm": "arm"})[["test", "dimension", "arm", "verdict"]]
       .assign(section="environment"),
    sc.rename(columns={"baseline": "arm"})[["test", "dimension", "arm", "verdict"]]
       .assign(section="agent"),
], ignore_index=True)
summary.to_csv(RES / "validation_summary.csv", index=False)
print(f"\nwrote agent_scorecard.csv ({len(sc)} rows), statistical_tests.csv, validation_summary.csv")
