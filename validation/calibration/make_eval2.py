"""R2-1: build the round-2 state tables BEFORE any simulation (PROTOCOL_round2.md).

- Initialization offset chosen by the frozen rule: q0+8 if >= 12 of the 19
  HOLDOUT companies have >= 13 complete (revenue-bearing) quarters from their
  round-1 q0; otherwise q0+4. Printed and committed before any result.
- eval2_states.csv: one row per (company, price), split in {DEV2 (CAL),
  EVAL2 (HOLDOUT)}, v1-mapping fields exactly as mapped_states.csv plus the
  round-2 company-specific CAC (cac_v2, clamp flag) and the raw panel fields
  the runner needs. No quantity uses any row with qi > init quarter.
- cac_mapping_r2.csv: per-company CAC derivation + clamp bounds (computed on
  CAL companies' quarters only, per price band).
- financing_hazard_r2.json: runway-binned raise hazard from CAL companies'
  burning quarters only (D4 raise definition unchanged), with inheritance for
  bins under 10 observations, printed next to the round-1 D4 numbers.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import CAL_DIR, CHURN_BY_PRICE, ROOT, load_panel, load_split, pick_init  # noqa: E402

EPS = 1.0
BINS = [0.0, 12.0, 24.0, 48.0]  # lower edges; last bin open-ended
MIN_BIN_N = 10


def company_cac(df_co: pd.DataFrame, qi: int, price: float) -> float | None:
    """Protocol CAC formula; uses ONLY rows with qi_ <= qi (asserted)."""
    window = df_co[df_co.qi <= qi]
    assert (window.qi <= qi).all(), "look-ahead into the mapping"
    by_qi = {r.qi: r for r in window.itertuples()}
    need = [qi - k for k in range(0, 4)]
    if any(q not in by_qi or pd.isna(by_qi[q].revenue) or pd.isna(by_qi[q].sm_pct_revenue)
           for q in need):
        return None
    if qi - 4 not in by_qi or pd.isna(by_qi[qi - 4].revenue):
        return None
    c_m = CHURN_BY_PRICE[price]
    trailing_sm = sum(by_qi[q].revenue * by_qi[q].sm_pct_revenue for q in need)
    rev_mean = float(np.mean([by_qi[q].revenue for q in need]))
    net_new = by_qi[qi].revenue - by_qi[qi - 4].revenue
    churned_est = 12.0 * c_m * rev_mean
    gross_new = max(net_new + churned_est, EPS)
    new_customers = gross_new / (3.0 * price)
    return float(trailing_sm / max(new_customers, 1e-9))


def clamp_bounds(panel: pd.DataFrame, cal_tickers: set[str], price: float):
    vals = []
    for ticker in sorted(cal_tickers):
        df_co = panel[panel.ticker == ticker]
        for qi in df_co.qi:
            v = company_cac(df_co, int(qi), price)
            if v is not None and np.isfinite(v):
                vals.append(v)
    lo, hi = np.percentile(vals, [5, 95])
    return float(lo), float(hi), len(vals)


def hazard_table(panel: pd.DataFrame, cal_tickers: set[str]) -> dict:
    con = sqlite3.connect(ROOT / "data/edgar.db")
    ocf = pd.read_sql_query(
        "SELECT cik, fiscal_period, value AS ocf FROM facts "
        "WHERE concept='operating_cash_flow'", con)
    con.close()
    df = panel.merge(ocf, on=["cik", "fiscal_period"], how="left")
    df = df[df.ticker.isin(cal_tickers)].sort_values(["ticker", "qi"])
    df["prev_cash"] = df.groupby("ticker").cash_and_investments.shift(1)
    df["prev_qi"] = df.groupby("ticker").qi.shift(1)
    df["dcash"] = df.cash_and_investments - df.prev_cash
    obs = df.dropna(subset=["dcash", "ocf"])
    obs = obs[obs.prev_qi == obs.qi - 1]
    burn = obs[obs.ocf < 0].copy()
    burn["monthly_burn"] = -burn.ocf / 3.0
    burn["runway_start"] = burn.prev_cash / burn.monthly_burn
    burn["is_raise"] = burn.dcash > 0
    burn["raise_mult"] = np.where(burn.is_raise,
                                  (burn.dcash - burn.ocf) / burn.monthly_burn, np.nan)

    rows = []
    for i, lo in enumerate(BINS):
        hi = BINS[i + 1] if i + 1 < len(BINS) else np.inf
        sub = burn[(burn.runway_start >= lo) & (burn.runway_start < hi)]
        n = int(len(sub))
        q_b = float(sub.is_raise.mean()) if n else np.nan
        k_b = float(sub[sub.is_raise].raise_mult.median()) if sub.is_raise.any() else np.nan
        rows.append(dict(bin_lo=lo, bin_hi=None if np.isinf(hi) else hi, n=n,
                         q_b=q_b, K_b=k_b, inherited=False))

    # bins with < MIN_BIN_N CAL burning quarters inherit the neighbouring
    # bin's values (nearest bin with enough data; lower-runway side preferred)
    for i, r in enumerate(rows):
        if r["n"] >= MIN_BIN_N and np.isfinite(r["q_b"]) and np.isfinite(r["K_b"]):
            continue
        donor = None
        for j in list(range(i - 1, -1, -1)) + list(range(i + 1, len(rows))):
            if rows[j]["n"] >= MIN_BIN_N and np.isfinite(rows[j]["q_b"]) \
                    and np.isfinite(rows[j]["K_b"]) and not rows[j]["inherited"]:
                donor = j
                break
        if donor is not None:
            r["q_b"], r["K_b"] = rows[donor]["q_b"], rows[donor]["K_b"]
            r["inherited"] = True
    for r in rows:
        r["h_b"] = float(1 - (1 - r["q_b"]) ** (1 / 3))
    return dict(bins=[r["bin_lo"] for r in rows], rows=rows,
                h=[r["h_b"] for r in rows], K=[r["K_b"] for r in rows],
                n_burning_quarters=int(len(burn)),
                n_raises=int(burn.is_raise.sum()),
                source="CAL companies only; D4 raise definition unchanged")


def main() -> None:
    panel = load_panel()
    split = load_split().set_index("ticker")
    q0 = {}
    for ticker in split.index:
        row, _ = pick_init(panel[panel.ticker == ticker])
        if row is not None:
            q0[ticker] = int(row.qi)

    # frozen offset rule
    n_deep = 0
    for ticker in split[split.split == "HOLDOUT"].index:
        if ticker not in q0:
            continue
        df_co = panel[panel.ticker == ticker]
        n_complete = int(df_co[(df_co.qi >= q0[ticker])
                               & df_co.revenue.notna()].qi.nunique())
        n_deep += int(n_complete >= 13)
    offset = 8 if n_deep >= 12 else 4
    print(f"HOLDOUT companies with >=13 complete quarters from q0: {n_deep}/19 "
          f"-> offset q0+{offset}")

    cal_set = set(split[split.split == "CAL"].index)
    clamps = {p: clamp_bounds(panel, cal_set, p) for p in (250.0, 50.0)}
    for p, (lo, hi, n) in clamps.items():
        print(f"CAC clamp price={p:.0f}: [p5, p95] = [{lo:,.0f}, {hi:,.0f}] "
              f"over {n} CAL company-quarters")

    state_rows, cac_rows, excluded = [], [], []
    for ticker in sorted(split.index):
        if ticker not in q0:
            excluded.append((ticker, "no round-1 q0"))
            continue
        qi = q0[ticker] + offset
        df_co = panel[panel.ticker == ticker]
        by_qi = {r.qi: r for r in df_co.itertuples()}
        r = by_qi.get(qi)
        core_ok = r is not None and not any(
            pd.isna(v) for v in (r.revenue, r.sm_pct_revenue, r.rnd_pct_revenue,
                                 r.gross_margin, r.cash_and_investments, r.ga))
        future = [by_qi.get(qi + k) for k in range(1, 5)]
        future_ok = all(f is not None and pd.notna(f.revenue) for f in future)
        if not (core_ok and future_ok):
            excluded.append((ticker, f"incomplete at q0+{offset}"))
            continue
        actual = future[3].revenue / r.revenue - 1.0
        sp = "DEV2" if split.loc[ticker, "split"] == "CAL" else "EVAL2"
        for price in (250.0, 50.0):
            cac_raw = company_cac(df_co, qi, price)
            lo, hi, _ = clamps[price]
            clamped = cac_raw is not None and (cac_raw < lo or cac_raw > hi)
            cac_v2 = None if cac_raw is None else float(np.clip(cac_raw, lo, hi))
            c_m = CHURN_BY_PRICE[price]
            state_rows.append(dict(
                ticker=ticker, split=sp, init_quarter=r.fiscal_period,
                price_assumed=price, mrr=r.revenue / 3.0,
                cash=float(r.cash_and_investments), monthly_burn=float(r.ga) / 3.0,
                gross_margin=r.gross_margin,
                sm_monthly=r.revenue * r.sm_pct_revenue / 3.0,
                rnd_monthly=r.revenue * r.rnd_pct_revenue / 3.0,
                churn_assumed=c_m, cac_assumed=price / c_m / 3.0,
                cac_v2=cac_v2, cac_clamped=bool(clamped),
                revenue=r.revenue, sm_pct_revenue=r.sm_pct_revenue,
                rnd_pct_revenue=r.rnd_pct_revenue,
                cash_and_investments=float(r.cash_and_investments),
                ga=float(r.ga), actual_4q_growth=actual))
            cac_rows.append(dict(ticker=ticker, split=sp, price_assumed=price,
                                 cac_raw=cac_raw, cac_v2=cac_v2,
                                 clamp_lo=lo, clamp_hi=hi, clamped=bool(clamped),
                                 ltv=price / c_m,
                                 ltv_cac_v2=(price / c_m / cac_v2) if cac_v2 else None))

    ev = pd.DataFrame(state_rows)
    ev.to_csv(CAL_DIR / "eval2_states.csv", index=False)
    pd.DataFrame(cac_rows).to_csv(CAL_DIR / "cac_mapping_r2.csv", index=False)
    print(f"\neval2 states: {ev.ticker.nunique()} companies "
          f"({(ev[ev.price_assumed == 250].split == 'EVAL2').sum()} EVAL2, "
          f"{(ev[ev.price_assumed == 250].split == 'DEV2').sum()} DEV2); "
          f"excluded: {excluded}")
    n_cl = int(ev[ev.price_assumed == 250].cac_clamped.sum())
    print(f"clamped companies (price 250): {n_cl}")

    hz = hazard_table(panel, cal_set)
    (CAL_DIR / "financing_hazard_r2.json").write_text(json.dumps(hz, indent=2))
    print("\nround-2 hazard table (CAL only) vs round-1 D4 (full panel: "
          "R=18mo, K=24.4, p=0.261/mo):")
    for r in hz["rows"]:
        print(f"  runway [{r['bin_lo']:.0f}, "
              f"{r['bin_hi'] if r['bin_hi'] is not None else 'inf'}): n={r['n']} "
              f"q={r['q_b']:.3f}/qtr h={r['h_b']:.3f}/mo K={r['K_b']:.1f}x"
              f"{' (inherited)' if r['inherited'] else ''}")


if __name__ == "__main__":
    main()
