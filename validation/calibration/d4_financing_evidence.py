"""D4: financing evidence from the EDGAR panel. Parameters from data, not invention.

A company-quarter is flagged as an external raise when cash+investments
INCREASED while operating cash flow was NEGATIVE (the company burned cash yet
ended with more). Both series are as-ingested: OCF is true quarterly
(de-cumulated at ingest), cash_and_investments is a balance-sheet level.

Outputs (validation/calibration/d4_financing_evidence.csv + _summary.json):
  - raise frequency: raises per company-quarter, overall and per company
  - raise size: (delta cash + |OCF|) expressed as a multiple of monthly burn
    (monthly burn = |OCF|/3 for the quarter of the raise)
  - runway at raise: cash at start of the raise quarter / monthly burn
  - conditional raise probability per month given runway < R (for the F2 rule)

These medians become the F2 financing-rule parameters (R, K, p).
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import CAL_DIR, ROOT, qidx  # noqa: E402


def main() -> None:
    ratios = pd.read_csv(ROOT / "data/edgar_ratios.csv")
    con = sqlite3.connect(ROOT / "data/edgar.db")
    ocf = pd.read_sql_query(
        "SELECT cik, fiscal_period, value AS ocf FROM facts "
        "WHERE concept='operating_cash_flow'", con)
    con.close()
    df = ratios.merge(ocf, on=["cik", "fiscal_period"], how="left")
    df["qi"] = df.fiscal_period.map(qidx)
    df = df.sort_values(["ticker", "qi"])
    df["prev_cash"] = df.groupby("ticker").cash_and_investments.shift(1)
    df["prev_qi"] = df.groupby("ticker").qi.shift(1)
    df["dcash"] = df.cash_and_investments - df.prev_cash

    obs = df.dropna(subset=["dcash", "ocf"]).copy()
    obs = obs[obs.prev_qi == obs.qi - 1]  # consecutive quarters only
    obs["monthly_burn"] = np.where(obs.ocf < 0, -obs.ocf / 3.0, np.nan)
    obs["runway_start_months"] = obs.prev_cash / obs.monthly_burn
    obs["is_raise"] = (obs.dcash > 0) & (obs.ocf < 0)
    # external inflow >= dcash + |OCF| (cash rose despite burning |OCF|)
    obs["raise_size"] = np.where(obs.is_raise, obs.dcash - obs.ocf, np.nan)
    obs["raise_multiple_of_monthly_burn"] = obs.raise_size / obs.monthly_burn

    raises = obs[obs.is_raise]
    burn_q = obs[obs.ocf < 0]  # quarters where a raise was even possible

    out_cols = ["ticker", "fiscal_period", "prev_cash", "cash_and_investments",
                "dcash", "ocf", "monthly_burn", "runway_start_months",
                "is_raise", "raise_size", "raise_multiple_of_monthly_burn"]
    obs[out_cols].to_csv(CAL_DIR / "d4_financing_evidence.csv", index=False)

    # Conditional per-quarter raise probability as a function of runway
    bins = [0, 6, 12, 18, 24, 36, np.inf]
    labels = ["<6mo", "6-12", "12-18", "18-24", "24-36", ">36"]
    burn_q = burn_q.assign(runway_bin=pd.cut(burn_q.runway_start_months,
                                             bins, labels=labels))
    cond = burn_q.groupby("runway_bin", observed=True).is_raise.agg(["mean", "size"])
    print("P(raise in quarter | runway bin, burning):")
    print(cond.to_string())

    med_runway = float(raises.runway_start_months.median())
    med_mult = float(raises.raise_multiple_of_monthly_burn.median())

    # Raises happen at every runway level (opportunistic mega-raises at 100+
    # months dominate the unconditional median). The F2 rule models RESCUE
    # financing, so R is read off the conditional table: the runway level below
    # which P(raise | burning) per quarter is >= 0.5 — the empirical break sits
    # at 18 months (<6: 0.86, 6-12: 0.56, 12-18: 0.50, 18-24: 0.26). K and p
    # are then medians WITHIN that rescue regime, not over all raises.
    r_threshold = 18.0
    eligible = burn_q[burn_q.runway_start_months < r_threshold]
    rescue = raises[raises.runway_start_months < r_threshold]
    rescue_mult = float(rescue.raise_multiple_of_monthly_burn.median())
    p_q = float(eligible.is_raise.mean())
    p_m = 1 - (1 - p_q) ** (1 / 3)

    summary = dict(
        n_company_quarters=int(len(obs)),
        n_burning_quarters=int(len(burn_q)),
        n_raises=int(len(raises)),
        raise_freq_per_burning_quarter=float(burn_q.is_raise.mean()),
        median_raise_multiple_of_monthly_burn=med_mult,
        p25_raise_multiple=float(raises.raise_multiple_of_monthly_burn.quantile(.25)),
        p75_raise_multiple=float(raises.raise_multiple_of_monthly_burn.quantile(.75)),
        median_runway_at_raise_months=med_runway,
        rescue_regime_R_months=r_threshold,
        n_rescue_raises=int(len(rescue)),
        median_rescue_raise_multiple_of_monthly_burn=rescue_mult,
        conditional_p_raise_per_quarter_below_R=p_q,
        conditional_p_raise_per_month_below_R=p_m,
        F2_params=dict(
            financing_runway_threshold_months=round(r_threshold, 1),
            financing_raise_multiple=round(rescue_mult, 1),
            financing_monthly_prob=round(p_m, 3)),
        note=("R = 18mo, the break in P(raise|burning) by runway bin (>=0.5 "
              "below, 0.26 above); K = median raise multiple among rescue "
              "raises (runway < R); p = per-month raise prob conditional on "
              "burning and runway < R. Unconditional medians also recorded."))
    (CAL_DIR / "d4_financing_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
