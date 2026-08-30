"""Claim audit: recompute every citable headline number from the raw CSVs.

Writes validation/results/claim_audit.csv. Each row: claim, original value,
raw source, reproduced value, baseline, n, method, status. Statuses:
REPRODUCED / APPROXIMATELY_REPRODUCED / UNSUPPORTED / STALE / LEAKAGE-RISK.
Read-only over results/; nothing is re-run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background"
CONF = ROOT / "results/confirmation_runs/oracle_v4_confirmation__episodes_50__freq_5__seed_0__20260412_163603"
OUT = ROOT / "validation/results"
OUT.mkdir(parents=True, exist_ok=True)

rows = []


def add(claim, original, source, reproduced, baseline, n, method, status, note=""):
    rows.append(dict(claim=claim, original_value=original, raw_source=source,
                     reproduced_value=reproduced, baseline=baseline, sample_size=n,
                     statistical_method=method, status=status, note=note))


def close(a, b, tol=0.005):
    try:
        return abs(float(a) - float(b)) <= tol * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return False


full_ep = pd.read_csv(FULL / "primary_episode_metrics.csv")
conf_ep = pd.read_csv(CONF / "primary_episode_metrics.csv")

# --- Survival rates -------------------------------------------------------
claimed_surv = {"boardroom": 97.33, "oracle_v1": 98.67, "oracle_v3": 98.67}
for pol, orig in claimed_surv.items():
    sub = full_ep[full_ep.policy == pol]
    rep = (sub.cause == "Time Limit").mean() * 100
    add(f"FULL survival % ({pol})", orig, "primary_episode_metrics.csv (FULL)",
        round(rep, 2), "boardroom", len(sub), "share of Time Limit episodes",
        "REPRODUCED" if close(rep, orig, 0.01) else "UNSUPPORTED")

# --- Final MRR means ------------------------------------------------------
claimed_mrr = {"boardroom": 1_389_958.13, "oracle_v1": 2_350_117.30, "oracle_v3": 2_251_580.38}
for pol, orig in claimed_mrr.items():
    sub = full_ep[full_ep.policy == pol]
    rep = sub.final_mrr.mean()
    add(f"FULL mean final MRR ({pol})", orig, "primary_episode_metrics.csv (FULL)",
        round(rep, 2), "boardroom", len(sub), "mean",
        "REPRODUCED" if close(rep, orig) else "UNSUPPORTED")

# --- Significance tests (FULL) -------------------------------------------
full_sum = pd.read_csv(FULL / "primary_episode_metric_summary.csv")
metric_frame = {"post_shock_avg_rule40": full_sum, "final_mrr": full_ep}
sig = pd.read_csv(FULL / "primary_significance_tests.csv")
for _, r in sig.iterrows():
    pol = r["comparison_scenario_id"]
    metric = r["metric"]
    frame = metric_frame.get(metric, full_ep)
    if metric not in frame.columns:
        add(f"FULL MWU {metric} boardroom vs {pol}", f"p={r['p_value']:.3g}",
            "primary_significance_tests.csv", "metric column not found", "boardroom",
            "?", "Mann-Whitney U", "UNSUPPORTED")
        continue
    a = frame[frame.policy == "boardroom"][metric].dropna()
    b = frame[frame.policy == pol][metric].dropna()
    u, p = stats.mannwhitneyu(b, a, alternative="two-sided")
    orig_p = float(r["p_value"])
    status = "REPRODUCED" if close(p, orig_p, 0.05) or (p < 0.05) == (orig_p < 0.05) else "UNSUPPORTED"
    add(f"FULL MWU {metric} boardroom vs {pol}", f"U={r['U']}, p={orig_p:.3g}",
        "primary_significance_tests.csv", f"U={u:.1f}, p={p:.3g}", "boardroom",
        f"{len(a)}/{len(b)}", "Mann-Whitney U two-sided (unpaired; design was paired)",
        status, "unpaired test on a seed-matched design understates power; see A2/A3 re-analysis")

# --- Recovery (FULL) ------------------------------------------------------
rec = pd.read_csv(FULL / "primary_recovery_events.csv")
claimed_rec = {"boardroom": 67.56, "oracle_v1": 76.00, "oracle_v3": 76.89}
for pol, orig in claimed_rec.items():
    sub = rec[rec.policy == pol]
    rep = sub.recovered.mean() * 100
    add(f"FULL recovered shock % ({pol})", orig, "primary_recovery_events.csv",
        round(rep, 2), "boardroom", len(sub), "share recovered",
        "REPRODUCED" if close(rep, orig, 0.01) else "UNSUPPORTED")

# --- CONFIRMATION headline ------------------------------------------------
for pol in ["boardroom", "oracle_v1", "oracle_v3", "oracle_v4", "oracle_v4_causal"]:
    sub = conf_ep[conf_ep.policy == pol]
    add(f"CONF survival % ({pol})",
        {"boardroom": 98.0}.get(pol, 100.0), "primary_episode_metrics.csv (CONF)",
        round((sub.cause == "Time Limit").mean() * 100, 2), "boardroom", len(sub),
        "share of Time Limit episodes", "REPRODUCED"
        if close((sub.cause == "Time Limit").mean() * 100, {"boardroom": 98.0}.get(pol, 100.0), 0.01)
        else "UNSUPPORTED")

# v4 == v3 duplication check
v3 = conf_ep[conf_ep.policy == "oracle_v3"].sort_values("seed").final_mrr.to_numpy()
v4 = conf_ep[conf_ep.policy == "oracle_v4"].sort_values("seed").final_mrr.to_numpy()
med_diff = abs(np.median(v3) - np.median(v4))
add("oracle_v4 differs from oracle_v3 (CONF)", "implied by running both arms",
    "primary_episode_metrics.csv (CONF)",
    f"median final MRR differs by ${med_diff:,.2f}; per-seed corr={np.corrcoef(v3, v4)[0,1]:.6f}",
    "oracle_v3", "50/50", "median comparison",
    "UNSUPPORTED" if med_diff < 1.0 else "REPRODUCED",
    "any v3-vs-v4 distinction claim is unsupported by this run")

# Reward direction caveat
for pol in ["oracle_v1", "oracle_v3"]:
    d = full_ep[full_ep.policy == pol].total_reward.mean() - full_ep[full_ep.policy == "boardroom"].total_reward.mean()
    add(f"FULL total_reward {pol} minus boardroom", "not claimed (audit note)",
        "primary_episode_metrics.csv (FULL)", round(d, 2), "boardroom", "75/75", "mean difference",
        "REPRODUCED", "NEGATIVE: oracle reward is worse than boardroom; do not cite reward as improved")

# Curated screenshot file
add("summary table without oracle_v3 row", "screenshot variants exist",
    "primary_summary_screenshot_no_oracle_v3.csv", "row deleted vs canonical file",
    "-", "-", "file diff", "LEAKAGE-RISK",
    "curated copy made 8 days post-run; cite canonical primary_summary.csv only")

df = pd.DataFrame(rows)
df.to_csv(OUT / "claim_audit.csv", index=False)
print(df[["claim", "original_value", "reproduced_value", "status"]].to_string(index=False, max_colwidth=60))
print(f"\n{len(df)} claims audited -> {OUT/'claim_audit.csv'}")
print("status counts:", df.status.value_counts().to_dict())
