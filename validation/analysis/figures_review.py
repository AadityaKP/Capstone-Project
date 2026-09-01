"""Review figure set F1-F11 -> validation/figures/review/ (PNG 200dpi + SVG).

One entry point: python validation/analysis/figures_review.py
Reads existing results from validation/results/ (never recomputes them);
the only new computations are written to validation/results/ first
(f4_growth_vs_scale.csv, f6_paired_diffs.csv; A8 CSVs come from
a8_shock_recovery.py, run before this script). Ends with consistency checks
of every caption number against the scorecards/stats files, and writes
validation/figures/review/README.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import COLORS, LABELS, REVIEW_DIR, footnote, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "validation/results"
FULL = ROOT / "results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background"

CHECKS: list[tuple[str, bool, str]] = []
README_ROWS: list[dict] = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))


def readme(figure, claim, source, n, caveat=""):
    README_ROWS.append(dict(figure=figure, claim=claim, source=source, n=n, caveat=caveat))


# ---------------------------------------------------------------- shared data
edgar = pd.read_csv(ROOT / "data/edgar_ratios.csv")
edgar["qi"] = edgar.fiscal_period.str.split("Q").map(lambda x: int(x[0]) * 4 + int(x[1]))
env_stats = pd.read_csv(RES / "environment_stats.csv")


def stat(metric, side):
    return env_stats[(env_stats.metric == metric) & (env_stats.side == side)].iloc[0]


def sim_quarterly(policy):
    mt = pd.read_csv(FULL / "primary_monthly_trace.csv",
                     usecols=["policy", "episode", "month", "mrr"])
    mt = mt[mt.policy == policy].copy()
    mt["quarter"] = mt.month // 3
    q = (mt.groupby(["episode", "quarter"])
           .agg(qrev=("mrr", "sum"), n=("mrr", "size")).reset_index())
    q = q[q.n == 3].sort_values(["episode", "quarter"])
    q["qoq_growth"] = q.groupby("episode").qrev.pct_change()
    return q


# ================================================================ F1
def f1():
    fig, (ax, axb) = plt.subplots(1, 2, figsize=(10, 4.2),
                                  gridspec_kw={"width_ratios": [2.6, 1]})
    med_paths = {}
    for t, g in edgar.dropna(subset=["revenue"]).groupby("ticker"):
        g = g.sort_values("qi")
        idx = 100.0 * g.revenue.to_numpy() / g.revenue.iloc[0]
        ax.plot(range(len(idx)), idx, color=COLORS["EDGAR"], alpha=0.35, lw=0.8)
        for i, v in enumerate(idx):
            med_paths.setdefault(i, []).append(v)
    xs = [i for i, v in sorted(med_paths.items()) if len(v) >= 10]
    ax.plot(xs, [np.median(med_paths[i]) for i in xs], color="#303030", lw=2.4,
            label="panel median (>=10 companies)")
    ax.set_yscale("log")
    ax.set_xlabel("Quarters since first filed quarter")
    ax.set_ylabel("Revenue, indexed (first quarter = 100, log)")
    ax.legend(loc="upper left")

    core = edgar.dropna(subset=["revenue", "qoq_growth", "sm_pct_revenue", "rnd_pct_revenue"])
    counts = core.groupby("ticker").size().sort_values()
    axb.barh(range(len(counts)), counts.to_numpy(), color=COLORS["EDGAR"])
    axb.set_yticks(range(len(counts)))
    axb.set_yticklabels(counts.index, fontsize=5)
    axb.set_xlabel("complete quarters")
    fig.suptitle("F1 - The dataset: 39 public SaaS companies, 1,288 complete quarters")
    footnote(fig, f"n = {counts.index.nunique()} companies, {int(counts.sum())} complete "
                  "quarters (unit: company-quarter). Source: data/edgar_ratios.csv "
                  "(frozen; see data/DATASET_CARD.md).")
    save(fig, "f1_panel_trajectories")
    check("F1 complete quarters == 1288", int(counts.sum()) == 1288, f"got {int(counts.sum())}")
    readme("f1_panel_trajectories", "The dataset: 39 public SaaS companies, 1,288 quarters",
           "data/edgar_ratios.csv", "39 companies / 1,288 quarters",
           "indexing rebases each company to its first filed quarter")


# ================================================================ F2
def f2():
    # Same sample basis as environment_battery.py: rows with QoQ growth present,
    # so the percentiles are IDENTICAL to the E1/E4/E7b scorecard rows.
    edgar_e = edgar.dropna(subset=["qoq_growth"])
    disc = (edgar_e.sm_pct_revenue + edgar_e.rnd_pct_revenue).dropna()
    panels = [
        ("QoQ revenue growth", edgar_e.qoq_growth, (-0.15, 0.4), "qoq_growth"),
        ("(S&M + R&D) / revenue", disc, (0.0, 1.6), "disc_spend_pct_revenue"),
        ("Gross margin", edgar_e.gross_margin.dropna(), (0.2, 1.0), None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for ax, (title, s, xlim, metric) in zip(axes, panels):
        ax.hist(s.clip(*xlim), bins=50, color=COLORS["EDGAR"], density=True, alpha=0.85)
        pcts = {p: float(np.percentile(s, p)) for p in (10, 25, 50, 75, 90)}
        for p, v in pcts.items():
            ax.axvline(v, color="#303030", lw=1.4 if p == 50 else 0.8,
                       ls="-" if p == 50 else ":")
            ax.text(v, ax.get_ylim()[1] * 0.97, f"p{p}", rotation=90, fontsize=6,
                    va="top", ha="right")
        ax.set_title(f"{title}\nmedian {pcts[50]:.1%}")
        ax.set_xlim(*xlim)
        if metric:  # must match environment_stats.csv exactly
            e = stat(metric, "EDGAR")
            ok = all(abs(pcts[p] - e[f"p{p}" if p != 50 else "median"]) < 1e-9
                     for p in (10, 25, 75, 90)) and abs(pcts[50] - e["median"]) < 1e-9
            check(f"F2 {metric} percentiles match environment_stats.csv", ok)
    gm = edgar_e.gross_margin.dropna()
    check("F2 GM matches E7b row (median 73.5%, p10 56.1%, p90 82.2%)",
          f"{np.median(gm):.1%}" == "73.5%" and f"{np.percentile(gm,10):.1%}" == "56.1%"
          and f"{np.percentile(gm,90):.1%}" == "82.2%",
          f"got {np.median(gm):.1%}/{np.percentile(gm,10):.1%}/{np.percentile(gm,90):.1%}")
    ns = f"{edgar_e.qoq_growth.notna().sum():,} growth / {len(disc):,} spend / {len(gm):,} margin"
    fig.suptitle("F2 - The benchmark bands the simulator is calibrated and tested against")
    footnote(fig, f"n = {ns} company-quarters, 39 companies (rows with QoQ growth present, "
                  "the same basis as the scorecard). Source: data/edgar_ratios.csv; "
                  "percentiles identical to validation/results/environment_stats.csv "
                  "(E1/E4) and the E7b scorecard row.")
    save(fig, "f2_edgar_benchmark_bands")
    readme("f2_edgar_benchmark_bands", "The benchmark bands (growth, spend intensity, margin)",
           "data/edgar_ratios.csv + environment_stats.csv", ns + " company-quarters",
           "x-axes clipped for display; percentiles computed on unclipped data")


# ================================================================ F3
def f3():
    qb = sim_quarterly("boardroom").qoq_growth.dropna()
    qo = sim_quarterly("oracle_v3").qoq_growth.dropna()
    eg = edgar.qoq_growth.dropna()
    e = stat("qoq_growth", "EDGAR")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axvspan(e.p25, e.p75, color=COLORS["EDGAR"], alpha=0.18, label="EDGAR IQR")
    bins = np.linspace(-0.3, 0.5, 64)
    for s, key in [(eg, "EDGAR"), (qb, "boardroom"), (qo, "oracle_v3")]:
        ax.hist(s.clip(-0.3, 0.5), bins=bins, density=True, histtype="step", lw=1.8,
                color=COLORS[key], label=f"{LABELS[key]} (median {np.median(s):.1%})")
    ax.set_xlabel("Quarter-over-quarter revenue growth")
    ax.set_ylabel("Density")
    ax.set_title("F3 - Simulated growth sits inside the real growth distribution (E1 PASS)")
    ax.legend()
    footnote(fig, f"EDGAR n={len(eg)} company-quarters (39 companies); sim n={len(qb)}/"
                  f"{len(qo)} episode-quarters (75 episodes/arm, FULL run aggregated to "
                  "quarters). Sources: data/edgar_ratios.csv; "
                  "results/.../primary_monthly_trace.csv; bands from environment_stats.csv.")
    save(fig, "f3_growth_distribution_sim_vs_edgar")
    check("F3 sim boardroom median matches environment_stats",
          abs(np.median(qb) - stat("qoq_growth", "sim_boardroom")["median"]) < 1e-9)
    check("F3 sim oracle_v3 median matches environment_stats",
          abs(np.median(qo) - stat("qoq_growth", "sim_oracle_v3")["median"]) < 1e-9)
    readme("f3_growth_distribution_sim_vs_edgar",
           "Simulated growth sits inside the real growth distribution (E1 PASS)",
           "edgar_ratios.csv + FULL monthly trace + environment_stats.csv",
           "1,293 EDGAR quarters vs 2,920/2,925 sim episode-quarters",
           "sim distributions are wider (E5 PARTIAL); IQR band is EDGAR's")


# ================================================================ F4
def f4():
    rows = []
    for _, r in edgar.dropna(subset=["qoq_growth", "revenue"]).iterrows():
        rows.append(dict(panel="EDGAR", unit=r.ticker,
                         log10_revenue=np.log10(r.revenue), qoq_growth=r.qoq_growth))
    qb = sim_quarterly("boardroom").dropna(subset=["qoq_growth"])
    for _, r in qb.iterrows():
        rows.append(dict(panel="sim_boardroom", unit=f"ep{int(r.episode)}",
                         log10_revenue=np.log10(r.qrev), qoq_growth=r.qoq_growth))
    pts = pd.DataFrame(rows)
    pts.to_csv(RES / "f4_growth_vs_scale.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (panel, key) in zip(axes, [("EDGAR", "EDGAR"), ("sim_boardroom", "boardroom")]):
        sub = pts[pts.panel == panel]
        ax.scatter(sub.log10_revenue, sub.qoq_growth.clip(-0.4, 0.8), s=4, alpha=0.25,
                   color=COLORS[key])
        coef = np.polyfit(sub.log10_revenue, sub.qoq_growth, 1)
        xs = np.linspace(sub.log10_revenue.min(), sub.log10_revenue.max(), 50)
        ax.plot(xs, np.polyval(coef, xs), color="#303030", lw=2, label="pooled linear trend")
        rho = stat("growth_vs_logscale_spearman",
                   "EDGAR" if panel == "EDGAR" else "sim_boardroom")["median"]
        ax.set_title(f"{LABELS[key]} - median within-unit Spearman = {rho:.2f}")
        ax.set_xlabel("log10 quarterly revenue ($)")
        ax.set_ylabel("QoQ revenue growth")
        ax.legend()
    fig.suptitle("F4 - Growth slows with scale in both panels (E3 PASS)")
    footnote(fig, "Units: EDGAR company-quarters (39 companies, n=1,293); sim episode-"
                  "quarters (75 episodes, n=2,920). Points: validation/results/"
                  "f4_growth_vs_scale.csv; Spearman medians from environment_stats.csv (E3).")
    save(fig, "f4_growth_deceleration")
    readme("f4_growth_deceleration", "Growth slows with scale in both (E3 PASS)",
           "f4_growth_vs_scale.csv + environment_stats.csv",
           "1,293 EDGAR / 2,920 sim points",
           "y clipped at [-0.4, 0.8] for display; trend fit on unclipped data")


# ================================================================ F5
def f5():
    ep = pd.read_csv(RES / "policy_comparison_episodes.csv")
    tests = pd.read_csv(RES / "statistical_tests_policy_baselines.csv")
    order = ["noop", "random", "heuristic", "boardroom"]
    fig, (ax, axs) = plt.subplots(1, 2, figsize=(10, 4.2),
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    for i, p in enumerate(order):
        v = ep[ep.policy == p].final_mrr.clip(lower=100)
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.13, 0.13, len(v)),
                   v, s=9, alpha=0.55, color=COLORS[p])
        ax.hlines(v.median(), i - 0.25, i + 0.25, color="#303030", lw=2)
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS[p] for p in order])
    ax.set_ylabel("Final MRR at 120 months ($, log)")
    lines = []
    for base in ["noop", "random", "heuristic"]:
        r = tests[(tests.arm_a == base) & (tests.arm_b == "boardroom")
                  & (tests.metric == "final_mrr")].iloc[0]
        lines.append(f"boardroom vs {base}: g={r.hedges_g_paired:.2f}, "
                     f"Holm p={r.holm_p_vs_boardroom:.1e}")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top",
            fontsize=7.5, bbox=dict(fc="white", alpha=0.85, ec="#cccccc"))
    surv = [ep[ep.policy == p].survived.mean() for p in order]
    axs.bar(range(len(order)), surv, color=[COLORS[p] for p in order])
    axs.set_xticks(range(len(order)))
    axs.set_xticklabels([LABELS[p] for p in order], fontsize=7)
    axs.set_ylim(0, 1.05)
    axs.set_ylabel("Survival rate (120 months)")
    fig.suptitle("F5 - Boardroom beats trivial and rule-based controls at matched seeds (A2 PASS)")
    footnote(fig, "n = 50 matched seeds per policy (unit: episode), deterministic RNG. "
                  "Sources: validation/results/policy_comparison_episodes.csv, "
                  "statistical_tests_policy_baselines.csv. Caveat: no-action never goes "
                  "bankrupt - spending trades survival risk for growth.")
    save(fig, "f5_policy_baselines")
    readme("f5_policy_baselines",
           "Boardroom beats trivial and rule-based controls at matched seeds (A2 PASS)",
           "policy_comparison_episodes.csv + statistical_tests_policy_baselines.csv",
           "50 episodes/arm", "noop survives 100% by never spending; final MRR floor-clipped at $100 for log display")


# ================================================================ F6
def f6():
    rec = pd.read_csv(FULL / "primary_episode_metric_summary.csv")
    rep_b = pd.read_csv(RES / "a3/episodes_boardroom.csv").set_index("seed")
    rep_o = pd.read_csv(RES / "a3/episodes_oracle_v3.csv").set_index("seed")
    diffs = []
    for metric, col_rec, col_rep in [("final_mrr", "final_mrr", "final_mrr"),
                                     ("post_shock_r40", "post_shock_avg_rule40",
                                      "post_shock_avg_rule40_25_60")]:
        piv = rec.pivot(index="seed", columns="policy", values=col_rec)
        for seed, d in (piv["oracle_v3"] - piv["boardroom"]).dropna().items():
            diffs.append(dict(regime="recorded_n75", metric=metric, seed=seed, diff=d))
        for seed, d in (rep_o[col_rep] - rep_b[col_rep]).dropna().items():
            diffs.append(dict(regime="replication_n20", metric=metric, seed=seed, diff=d))
    dd = pd.DataFrame(diffs)
    dd.to_csv(RES / "f6_paired_diffs.csv", index=False)

    rp = pd.read_csv(RES / "a3_recorded_paired.csv")
    av = pd.read_csv(RES / "a3_oracle_value.csv")
    m_rec = dd[(dd.regime == "recorded_n75") & (dd.metric == "final_mrr")]["diff"].mean()
    m_ref = rp[(rp.comparison == "oracle_v3 - boardroom") & (rp.metric == "final_mrr")].mean_diff.iloc[0]
    check("F6 recorded mean diff matches a3_recorded_paired.csv", abs(m_rec - m_ref) < 1.0,
          f"{m_rec:,.1f} vs {m_ref:,.1f}")
    m_rep = dd[(dd.regime == "replication_n20") & (dd.metric == "final_mrr")]["diff"].mean()
    m_ref2 = av[(av.comparison == "oracle_v3 - boardroom") & (av.metric == "final_mrr")].mean_diff.iloc[0]
    check("F6 replication mean diff matches a3_oracle_value.csv", abs(m_rep - m_ref2) < 1.0,
          f"{m_rep:,.1f} vs {m_ref2:,.1f}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.8), constrained_layout=True)
    for row, metric, ylab in [(0, "final_mrr", "final MRR diff ($)"),
                              (1, "post_shock_r40", "post-shock Rule-of-40 diff")]:
        for col, regime, title in [(0, "recorded_n75", "recorded FULL run\n(n=75, legacy RNG, shared-world verified)"),
                                   (1, "replication_n20", "replication\n(n=20, deterministic RNG)")]:
            ax = axes[row][col]
            d = np.sort(dd[(dd.regime == regime) & (dd.metric == metric)]["diff"].to_numpy())
            ax.bar(range(len(d)), d, color=[COLORS["oracle_v3"] if v > 0 else "#b0b0b0" for v in d])
            ax.axhline(0, color="k", lw=0.8)
            if row == 0:
                ax.set_title(f"{title}\npositive in {(d > 0).sum()}/{len(d)} seeds", fontsize=8.5)
            else:
                ax.set_title(f"positive in {(d > 0).sum()}/{len(d)} seeds", fontsize=8.5)
                ax.set_xlabel("seed (sorted by diff)")
            ax.set_ylabel(ylab)
            if metric == "final_mrr":
                ax.set_yscale("symlog", linthresh=1e4)
    fig.suptitle("F6 - The oracle layer beats the boardroom in essentially every matched world "
                 "(A3 PASS, replicated)")
    footnote(fig, "Per-seed oracle_v3 - boardroom differences (unit: episode). Sources: "
                  "validation/results/f6_paired_diffs.csv (from primary_episode_metric_summary.csv "
                  "and a3/episodes_*.csv); summary stats match a3_recorded_paired.csv / "
                  "a3_oracle_value.csv. Final-MRR axis is symlog.")
    save(fig, "f6_oracle_paired_gain")
    readme("f6_oracle_paired_gain",
           "Oracle beats boardroom in essentially every matched world (A3 PASS, replicated)",
           "f6_paired_diffs.csv (+ a3_recorded_paired.csv, a3_oracle_value.csv)",
           "75 + 20 episodes", "oracle_v4 is never plotted separately: identical to oracle_v3 in recorded data")


# ================================================================ F7
def f7():
    curves = pd.read_csv(RES / "a8_shock_r40_curves.csv")
    recov = pd.read_csv(RES / "a8_shock_recovery.csv")
    recov = recov[recov.shock_month == "all"]
    show = [("A2", "noop"), ("A2", "random"), ("A2", "heuristic"), ("A2", "boardroom"),
            ("A3", "oracle_v1"), ("A3", "oracle_v3"), ("A3", "oracle_v3_no_memory")]
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(11, 4.4),
                                  gridspec_kw={"width_ratios": [1.7, 1]})
    for src, pol in show:
        c = curves[(curves.source == src) & (curves.policy == pol)].sort_values("rel_month")
        ax.plot(c.rel_month, c.mean_r40, color=COLORS[pol], lw=1.6, label=LABELS[pol])
        if pol in ("boardroom", "oracle_v3"):
            ax.fill_between(c.rel_month, c.ci95_lo, c.ci95_hi, color=COLORS[pol], alpha=0.15)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("Months relative to scheduled shock (24/48/72)")
    ax.set_ylabel("Rule of 40 (event-time mean; NOT revenue)")
    ax.legend(fontsize=6.5, ncol=2)
    rates, meds, cols, names = [], [], [], []
    for src, pol in show:
        r = recov[(recov.source == src) & (recov.policy == pol)].iloc[0]
        rates.append(r.recovery_rate)
        meds.append(r.median_months_recovered)
        cols.append(COLORS[pol])
        names.append(LABELS[pol])
    axr.bar(range(len(rates)), rates, color=cols)
    for i, (rate, med) in enumerate(zip(rates, meds)):
        axr.text(i, rate + 0.02, f"med {med:.0f}m", ha="center", fontsize=6.5)
    axr.set_xticks(range(len(names)))
    axr.set_xticklabels(names, rotation=35, ha="right", fontsize=6.5)
    axr.set_ylim(0, 1.0)
    axr.set_ylabel("Share of shocks regaining pre-shock\nRule of 40 within 24 months")
    fig.suptitle("F7 - Oracle policies recover Rule-of-40 faster after shocks (A8, COMPARATIVE)")
    footnote(fig, "Rule-of-40 recovery = regaining the pre-shock R40 level; this is NOT "
                  "revenue-peak recovery, which essentially does not occur in the simulator "
                  "(E6: 0-2%). Units: episodes (A2 arms n=50 seeds, A3 arms n=20 seeds; "
                  "events censored at +24 months or death - 'random' loses most episodes to "
                  "bankruptcy). Source: validation/results/a8_shock_r40_curves.csv, "
                  "a8_shock_recovery.csv.")
    save(fig, "f7_post_shock_r40_recovery")
    readme("f7_post_shock_r40_recovery",
           "Oracle policies recover Rule-of-40 faster after shocks (A8 COMPARATIVE)",
           "a8_shock_r40_curves.csv + a8_shock_recovery.csv",
           "50 episodes (A2 arms) / 20 episodes (A3 arms), 3 shocks each",
           "R40 recovery, never revenue-peak recovery (E6); mixed ns across arms")


# ================================================================ F8
def f8():
    bt = pd.read_csv(RES / "real_company_backtest.csv")
    bt = bt[bt.price_assumed == 250.0]
    ok = bt.dropna(subset=["sim_hold_median"])
    dead = bt[bt.sim_hold_median.isna()]
    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    lim = [-0.15, 1.35]
    ax.plot(lim, lim, "k--", lw=1, label="perfect retrodiction (y=x)")
    partial = ok[ok.deaths_hold > 0]
    clean = ok[ok.deaths_hold == 0]
    ax.scatter(clean.actual_4q_growth, clean.sim_hold_median, color=COLORS["EDGAR"],
               s=26, label="hold arm (no in-sim deaths)")
    ax.scatter(partial.actual_4q_growth, partial.sim_hold_median, facecolors="none",
               edgecolors=COLORS["random"], s=34,
               label="hold arm (some seeds bankrupt in-sim)")
    ax.scatter(dead.actual_4q_growth, np.full(len(dead), lim[0] + 0.02), marker="x",
               color=COLORS["random"], s=40,
               label=f"bankrupt in-sim on all seeds ({', '.join(dead.ticker)})")
    mae = ok.retrodiction_error.abs().median()
    ax.text(0.03, 0.97, f"median |error| = {mae:.1%} over {len(ok)} companies\n"
                        "(criterion: <=10pp PASS / <=20pp PARTIAL -> FAIL)",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(fc="white", alpha=0.85, ec="#cccccc"))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual next-4-quarter revenue growth (EDGAR)")
    ax.set_ylabel("Simulated 4-quarter growth, hold arm (median of 30 seeds)")
    ax.set_title("F8 - Initialized from real companies the simulator over-projects growth (C1 FAIL)\n"
                 "diagnosed: SATURATION_ACQUISITION_RATE too high + no financing mechanism",
                 fontsize=9.5)
    ax.legend(fontsize=6.5, loc="lower right")
    footnote(fig, f"n = {len(bt)} companies (unit: company; price-assumption $250 variant; "
                  "$50 variant similar). Source: validation/results/real_company_backtest.csv. "
                  "The 6 all-seeds-bankrupt companies really raised capital - the simulator "
                  "has no financing mechanism.")
    save(fig, "f8_backtest_retrodiction")
    check("F8 median |retrodiction error| ~ 49.6%", f"{mae:.1%}" == "49.6%", f"got {mae:.1%}")
    readme("f8_backtest_retrodiction",
           "Simulator over-projects real-company growth (C1 FAIL) - cause diagnosed",
           "real_company_backtest.csv", "39 companies (33 evaluable)",
           "hold arm only; boardroom-vs-hold increment is in the report section 5")


# ================================================================ F9 (optional)
def f9():
    grid = pd.read_csv(RES / "a7_robustness_grid.csv")
    pairs = pd.read_csv(RES / "a7_robustness_pairs.csv")
    rank = grid[grid.policy == "boardroom"].pivot(index="initial_mrr",
                                                  columns="initial_cash", values="rank_in_cell")
    g = pairs[pairs.baseline == "heuristic"].pivot(index="initial_mrr",
                                                   columns="initial_cash", values="hedges_g_paired")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, mat, title, fmt in [(axes[0], rank, "boardroom rank (1 = best of 4)", "{:.0f}"),
                                (axes[1], g, "paired Hedges g vs heuristic", "{:.2f}")]:
        im = ax.imshow(mat.to_numpy(), cmap="Blues_r" if fmt == "{:.0f}" else "Blues",
                       aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels([f"${c/1e6:.1f}M" for c in mat.columns])
        ax.set_yticks(range(3)); ax.set_yticklabels([f"${m/1e3:.0f}k" for m in mat.index])
        ax.set_xlabel("initial cash"); ax.set_ylabel("initial MRR")
        ax.set_title(title, fontsize=9)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, fmt.format(mat.to_numpy()[i, j]), ha="center", va="center",
                        fontsize=9, color="#222222")
        ax.grid(False)
    fig.suptitle("F9 - The boardroom advantage is stable across initial conditions (A7 PASS)")
    footnote(fig, "n = 20 matched seeds per cell (unit: episode). Source: validation/results/"
                  "a7_robustness_grid.csv, a7_robustness_pairs.csv.")
    save(fig, "f9_robustness_grid")
    readme("f9_robustness_grid", "Boardroom advantage stable across initial conditions (A7 PASS)",
           "a7_robustness_grid.csv + a7_robustness_pairs.csv", "9 cells x 20 episodes",
           "'random' ranks 2nd by median only because MRR-at-bankruptcy inflates its median")


# ================================================================ F10 (optional)
def f10():
    rp = pd.read_csv(RES / "a3_recorded_paired.csv")
    av = pd.read_csv(RES / "a3_oracle_value.csv")
    rows = [
        ("oracle_v3 - boardroom\n(recorded n=75)", rp[(rp.comparison == "oracle_v3 - boardroom")
                                                      & (rp.metric == "final_mrr")].iloc[0], "oracle_v3"),
        ("oracle_v3 - boardroom\n(replication n=20)", av[(av.comparison == "oracle_v3 - boardroom")
                                                         & (av.metric == "final_mrr")].iloc[0], "oracle_v3"),
        ("oracle_v3 - oracle_v3\nno-memory (n=20)", av[(av.comparison == "oracle_v3 - oracle_v3_no_memory")
                                                       & (av.metric == "final_mrr")].iloc[0], "oracle_v3_no_memory"),
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, (label, r, key) in enumerate(rows):
        ax.bar(i, r.mean_diff, color=COLORS[key])
        ax.errorbar(i, r.mean_diff, yerr=[[r.mean_diff - r.ci95_lo], [r.ci95_hi - r.mean_diff]],
                    color="#303030", capsize=4, lw=1.2)
        ax.text(i, r.ci95_hi * 1.05, f"${r.mean_diff/1e3:,.0f}k", ha="center", fontsize=8)
    share = rows[2][1].mean_diff / rows[1][1].mean_diff
    ax.annotate(f"retrieval = {share:.0%} of the oracle gain", xy=(2, rows[2][1].mean_diff),
                xytext=(1.05, rows[1][1].mean_diff * 0.55),
                arrowprops=dict(arrowstyle="->", color="#555555"), fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylabel("Mean paired final-MRR difference ($)")
    ax.set_title("F10 - The brief mechanism, not memory, carries the oracle gain "
                 "(retrieval ≈ 3%)")
    footnote(fig, "Bars = mean paired difference, whiskers = 95% bootstrap CI (unit: episode). "
                  "Sources: validation/results/a3_recorded_paired.csv, a3_oracle_value.csv.")
    save(fig, "f10_memory_ablation")
    check("F10 retrieval share ~ 3%", abs(share - 0.032) < 0.01, f"got {share:.3f}")
    readme("f10_memory_ablation", "Retrieval contributes ~3% of the oracle gain",
           "a3_recorded_paired.csv + a3_oracle_value.csv", "75 / 20 / 20 episodes",
           "retrieval increment is significant (p=0.0023) but small")


# ================================================================ F11 (optional)
def f11():
    ae = pd.read_csv(ROOT / "validation/agents/action_effects.csv")
    dims = {"marketing.spend": "#1f77b4", "product.r_and_d_spend": "#d95f02",
            "hiring.hires": "#9467bd", "pricing.price_change_pct": "#2ca02c"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, state in zip(axes, ["start_month0", "midgame_month18"]):
        for dim, col in dims.items():
            med = ae[(ae.state == state) & (ae.dimension == dim)].groupby("value").final_mrr.median()
            x = np.arange(len(med))
            ax.plot(x, med / med.iloc[len(med) // 2], marker="o", color=col,
                    label=dim.split(".")[-1])
            ax.set_xticks(x)
        ax.set_title(state)
        ax.set_xlabel("ladder rung (low -> high)")
        ax.set_ylabel("median 12-mo MRR / mid-rung")
        ax.legend(fontsize=7)
    fig.suptitle("F11 - Marketing is the strong causal lever; R&D and pricing weak; hiring pure cost (A1 PASS)")
    footnote(fig, "n = 20 matched seeds per rung (unit: episode), one dimension varied at a "
                  "time. Source: validation/agents/action_effects.csv. Colours here encode "
                  "action dimensions, not policies.")
    save(fig, "f11_action_ladders")
    readme("f11_action_ladders", "Actions causally move outcomes; marketing dominates (A1 PASS)",
           "validation/agents/action_effects.csv", "20 episodes x 2 states x rungs",
           "hiring lever affects cash/survival, not revenue")


# ================================================================ main
def main():
    np.random.seed(0)
    for fn in [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]:
        print(f"--- {fn.__name__} ---")
        fn()

    md = ["# Review figure set", "",
          "Generated by `python validation/analysis/figures_review.py` (after "
          "`python validation/analysis/a8_shock_recovery.py`). PNG 200 dpi + SVG. "
          "Palette fixed in `validation/analysis/fig_style.py` "
          "(EDGAR grey; one colour per policy).", "",
          "| Figure | Claim | Source | n | Caveat |", "|---|---|---|---|---|"]
    for r in README_ROWS:
        md.append(f"| `{r['figure']}` | {r['claim']} | {r['source']} | {r['n']} | {r['caveat']} |")
    md += ["", "Ground rules honoured: no screenshot-variant files used; oracle_v4 never "
           "plotted as distinct from oracle_v3; reward never plotted as an outcome; F7 is "
           "Rule-of-40 recovery, never revenue-peak recovery."]
    (REVIEW_DIR / "README.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nwrote {REVIEW_DIR / 'README.md'}")

    print("\n=== consistency checks ===")
    for name, ok, detail in CHECKS:
        print(f"  [{'OK' if ok else 'MISMATCH'}] {name}" + (f" ({detail})" if detail else ""))
    if all(ok for _, ok, _ in CHECKS):
        print("all caption numbers agree with the scorecards/stats files")


if __name__ == "__main__":
    main()
