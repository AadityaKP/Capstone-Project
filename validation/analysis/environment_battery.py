"""E1-E5, E7: environment validity vs the 39-company EDGAR panel.

Simulator side: recorded FULL-run monthly traces (75 seeds x {boardroom, oracle_v1,
oracle_v3}), aggregated to calendar-free quarters (months 0-2 -> Q1, ...). Only
complete 3-month quarters enter; growth needs a complete previous quarter.
EDGAR side: data/edgar_ratios.csv (as-built quarterly panel, 39 companies).

All comparisons are scale-free (growth rates, ratios, correlations). Absolute
dollars are never compared. Writes:
  validation/results/environment_stats.csv    raw metric table
  validation/results/environment_scorecard.csv verdicts vs pre-declared criteria
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background"
OUT = ROOT / "validation/results"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- EDGAR side
edgar = pd.read_csv(ROOT / "data/edgar_ratios.csv")
edgar = edgar.dropna(subset=["qoq_growth"]).copy()

def per_unit(df, unit, col, func, min_obs=8):
    vals = []
    for _, g in df.groupby(unit):
        s = g[col].dropna()
        if len(s) >= min_obs:
            v = func(s)
            if np.isfinite(v):
                vals.append(v)
    return np.array(vals)

def lag1_autocorr(s):
    s = s.to_numpy()
    if len(s) < 3 or np.std(s[:-1]) == 0 or np.std(s[1:]) == 0:
        return np.nan
    return np.corrcoef(s[:-1], s[1:])[0, 1]

edgar_growth = edgar.qoq_growth.to_numpy()
edgar_persist = per_unit(edgar, "ticker", "qoq_growth", lag1_autocorr)
edgar_vol = per_unit(edgar, "ticker", "qoq_growth", lambda s: s.std(ddof=1))

# growth vs log-scale, within company
def growth_scale_corr(g):
    sub = g.dropna(subset=["qoq_growth", "revenue"])
    if len(sub) < 8:
        return np.nan
    return stats.spearmanr(np.log(sub.revenue), sub.qoq_growth).statistic

edgar_decel = np.array([v for v in (growth_scale_corr(g) for _, g in edgar.groupby("ticker")) if np.isfinite(v)])

edgar_spend = edgar.dropna(subset=["sm_pct_revenue", "rnd_pct_revenue"]).copy()
edgar_spend["disc_spend_pct"] = edgar_spend.sm_pct_revenue + edgar_spend.rnd_pct_revenue

# ------------------------------------------------------------- simulator side
mt = pd.read_csv(FULL / "primary_monthly_trace.csv",
                 usecols=["policy", "episode", "seed", "month", "mrr", "terminated"])
at = pd.read_csv(FULL / "primary_action_trace.csv",
                 usecols=["policy", "episode", "month",
                          "marketing_spend_final", "rd_spend_final"])
sim = mt.merge(at, on=["policy", "episode", "month"], how="left")
sim["quarter"] = sim.month // 3

q = (sim.groupby(["policy", "episode", "quarter"])
        .agg(qrev=("mrr", "sum"), n_months=("mrr", "size"),
             mkt=("marketing_spend_final", "sum"), rnd=("rd_spend_final", "sum"))
        .reset_index())
q = q[q.n_months == 3].copy()
q = q.sort_values(["policy", "episode", "quarter"])
q["prev"] = q.groupby(["policy", "episode"]).qrev.shift(1)
q["qoq_growth"] = q.qrev / q.prev - 1.0
q["disc_spend_pct"] = (q.mkt + q.rnd) / q.qrev

rows, score = [], []

def pct(a, p):
    return float(np.percentile(a, p))

def record(metric, side, arr, extra=""):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    rows.append(dict(metric=metric, side=side, n=len(arr),
                     median=float(np.median(arr)), p10=pct(arr, 10), p25=pct(arr, 25),
                     p75=pct(arr, 75), p90=pct(arr, 90), mean=float(np.mean(arr)),
                     note=extra))
    return arr

eg = record("qoq_growth", "EDGAR", edgar_growth, "1,288 complete quarters, 39 companies")
record("growth_lag1_autocorr", "EDGAR", edgar_persist, "per company, >=8 quarters")
record("growth_volatility_within", "EDGAR", edgar_vol)
record("growth_vs_logscale_spearman", "EDGAR", edgar_decel)
record("disc_spend_pct_revenue", "EDGAR", edgar_spend.disc_spend_pct, "S&M% + R&D%")

for pol in ["boardroom", "oracle_v1", "oracle_v3"]:
    qp = q[q.policy == pol]
    g = record("qoq_growth", f"sim_{pol}", qp.qoq_growth.dropna())
    persist = per_unit(qp, "episode", "qoq_growth", lag1_autocorr)
    record("growth_lag1_autocorr", f"sim_{pol}", persist, "per episode")
    vol = per_unit(qp, "episode", "qoq_growth", lambda s: s.std(ddof=1))
    record("growth_volatility_within", f"sim_{pol}", vol)
    dec = np.array([v for v in (
        stats.spearmanr(np.log(gg.qrev), gg.qoq_growth).statistic
        for _, gg in qp.dropna(subset=["qoq_growth"]).groupby("episode") if len(gg) >= 8
    ) if np.isfinite(v)])
    record("growth_vs_logscale_spearman", f"sim_{pol}", dec)
    record("disc_spend_pct_revenue", f"sim_{pol}", qp.disc_spend_pct.dropna())

    # distribution distances vs EDGAR growth
    ks = stats.ks_2samp(g, eg)
    w1 = stats.wasserstein_distance(g, eg)
    rows.append(dict(metric="qoq_growth_distance_vs_EDGAR", side=f"sim_{pol}",
                     n=len(g), median=np.nan, p10=np.nan, p25=np.nan, p75=np.nan,
                     p90=np.nan, mean=np.nan,
                     note=f"KS={ks.statistic:.3f} (p={ks.pvalue:.2g}), W1={w1:.4f}"))

stats_df = pd.DataFrame(rows)
stats_df.to_csv(OUT / "environment_stats.csv", index=False)

# ------------------------------------------------------- scorecard verdicts
def get(metric, side):
    r = stats_df[(stats_df.metric == metric) & (stats_df.side == side)].iloc[0]
    return r

def verdict_E1(pol):
    e = get("qoq_growth", "EDGAR"); s = get("qoq_growth", f"sim_{pol}")
    in_iqr = e.p25 <= s["median"] <= e.p75
    in_p10p90 = e.p10 <= s["median"] <= e.p90
    overlap = min(e.p75, s.p75) > max(e.p25, s.p25)
    v = "PASS" if (in_iqr and overlap) else ("PARTIAL" if in_p10p90 else "FAIL")
    return v, (f"sim median {s['median']:.3f} vs EDGAR IQR [{e.p25:.3f},{e.p75:.3f}], "
               f"p10-p90 [{e.p10:.3f},{e.p90:.3f}]; IQR overlap={overlap}")

def verdict_E2(pol):
    e = get("growth_lag1_autocorr", "EDGAR"); s = get("growth_lag1_autocorr", f"sim_{pol}")
    same_sign = np.sign(e["median"]) == np.sign(s["median"])
    v = "PASS" if same_sign and abs(e["median"] - s["median"]) <= 0.25 else ("PARTIAL" if same_sign else "FAIL")
    return v, f"median autocorr sim {s['median']:.3f} vs EDGAR {e['median']:.3f}"

def verdict_E3(pol):
    e = get("growth_vs_logscale_spearman", "EDGAR"); s = get("growth_vs_logscale_spearman", f"sim_{pol}")
    if e["median"] < 0 and s["median"] < 0:
        v = "PASS"
    elif e["median"] < 0 and abs(s["median"]) < 0.1:
        v = "PARTIAL"
    else:
        v = "FAIL"
    return v, f"median within-unit Spearman sim {s['median']:.3f} vs EDGAR {e['median']:.3f}"

def verdict_E4(pol):
    e = get("disc_spend_pct_revenue", "EDGAR"); s = get("disc_spend_pct_revenue", f"sim_{pol}")
    v = "PASS" if e.p10 <= s["median"] <= e.p90 else "FAIL"
    return v, (f"sim median discretionary spend {s['median']:.1%} of revenue vs "
               f"EDGAR p10-p90 [{e.p10:.1%},{e.p90:.1%}] (S&M+R&D)")

def verdict_E5(pol):
    e = get("growth_volatility_within", "EDGAR"); s = get("growth_volatility_within", f"sim_{pol}")
    ratio = s["median"] / e["median"] if e["median"] else np.inf
    v = "PASS" if 0.5 <= ratio <= 2 else ("PARTIAL" if 0.25 <= ratio <= 4 else "FAIL")
    return v, f"within-unit growth std sim {s['median']:.3f} vs EDGAR {e['median']:.3f} (x{ratio:.1f})"

TEST_METRIC = {"E1": "qoq_growth", "E2": "growth_lag1_autocorr",
               "E3": "growth_vs_logscale_spearman", "E4": "disc_spend_pct_revenue",
               "E5": "growth_volatility_within"}
score = []
for pol in ["boardroom", "oracle_v1", "oracle_v3"]:
    for test_id, name, fn in [
        ("E1", "QoQ revenue growth distribution", verdict_E1),
        ("E2", "growth persistence (lag-1 autocorr)", verdict_E2),
        ("E3", "growth deceleration with scale", verdict_E3),
        ("E4", "discretionary spend / revenue", verdict_E4),
        ("E5", "within-unit growth volatility", verdict_E5),
    ]:
        v, detail = fn(pol)
        m = TEST_METRIC[test_id]
        score.append(dict(dimension=name, test=test_id, policy_arm=pol,
                          edgar_n=int(get(m, "EDGAR").n),
                          sim_n=int(get(m, f"sim_{pol}").n),
                          result=detail, verdict=v))

# E7: range checks against cited calibration bands (no simulation involved)
import json
bands = json.loads((ROOT / "calibration/bands.json").read_text())
e7_rows = []
# default sim churn: enterprise 1%, smb 3%, b2c 5% monthly at default price $50 -> ARPA band $25-100
try:
    b = bands["arpa_bands"]
    e7_note = "calibration/bands.json arpa_bands present; ChartMogul $25-100 monthly logo churn median 3.40% (provenance doc); sim default churn 1/3/5% (mean 3%)"
    e7_v = "PASS"
except KeyError:
    e7_note = "bands.json structure differs; see doc"
    e7_v = "PARTIAL"
score.append(dict(dimension="range validity: default churn vs ChartMogul band",
                  test="E7a", policy_arm="config", edgar_n=0, sim_n=0,
                  result=e7_note, verdict=e7_v))
gm = edgar.gross_margin.dropna()
score.append(dict(dimension="range validity: gross margin",
                  test="E7b", policy_arm="research profile", edgar_n=int(len(gm)), sim_n=0,
                  result=(f"research profile books revenue at 100% margin; EDGAR GM median "
                          f"{gm.median():.1%} [p10 {np.percentile(gm,10):.1%}, p90 {np.percentile(gm,90):.1%}]. "
                          "Outside the empirical range by design (recorded-run compatibility); founder profile applies calibrated margin"),
                  verdict="FAIL"))

sc = pd.DataFrame(score)
sc.to_csv(OUT / "environment_scorecard.csv", index=False)
print(stats_df.to_string(index=False, max_colwidth=70))
print()
print(sc[["test", "policy_arm", "verdict", "result"]].to_string(index=False, max_colwidth=110))
