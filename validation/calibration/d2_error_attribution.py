"""D2: decompose the C1 hold-arm 4q growth error into mechanisms, on CAL only.

6 representative CAL companies (2 per size tercile, chosen deterministically:
per tercile the lowest- and highest-init-revenue CAL company, excluding the
all-seed hold-arm deaths ASAN/DOMO/TENB whose growth is undefined).

Mechanism switches, one at a time (RNG-aligned; see backtest_lib.physics_patches):
  (a)  marketing acquisition contribution zeroed (spend still leaves cash)
  (b1) churn pinned flat at the assumed band median (2.7%/mo at $250 ARPA),
       bypassing the quality/macro/tenure multiplier stack
  (b2) company-implied flat churn: bisection for the flat churn that makes sim
       hold growth match actual (identifiability of the churn explanation)
  (c)  expansion MRR zeroed

Contribution of a mechanism = baseline median growth - switched median growth
(pp of the over-projection the mechanism accounts for). 10 seeds, hold arm,
v1 physics (this is a diagnostic; nothing is fixed here).

Writes validation/calibration/d2_error_attribution.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import (CAL_DIR, company_table, load_panel, load_split,  # noqa: E402
                          pick_init, run_company_arm)

SEEDS = range(10)
BAND_MEDIAN_CHURN = 0.027
EXCLUDE = {"ASAN", "DOMO", "TENB"}  # all-seed hold deaths (CAL side)


def pick_companies() -> list[str]:
    split = load_split()
    cal = split[(split.split == "CAL") & (~split.ticker.isin(EXCLUDE))]
    chosen = []
    for terc in ("small", "mid", "large"):
        sub = cal[cal.revenue_tercile == terc].sort_values("init_revenue")
        chosen += [sub.iloc[0].ticker, sub.iloc[-1].ticker]
    return chosen


def implied_churn(row, actual: float, lo=0.001, hi=0.20, iters=14) -> tuple[float, float]:
    """Flat churn that equates sim hold median growth with actual growth."""
    g_lo = run_company_arm(row, "hold", SEEDS, patches={"churn_override": lo})["median_growth"]
    g_hi = run_company_arm(row, "hold", SEEDS, patches={"churn_override": hi})["median_growth"]
    if not (g_hi <= actual <= g_lo):  # growth decreasing in churn
        # unbracketable: report the closer endpoint
        return (lo, g_lo) if abs(g_lo - actual) < abs(g_hi - actual) else (hi, g_hi)
    for _ in range(iters):
        mid = (lo + hi) / 2
        g_mid = run_company_arm(row, "hold", SEEDS,
                                patches={"churn_override": mid})["median_growth"]
        if g_mid > actual:
            lo = mid
        else:
            hi = mid
    mid = (lo + hi) / 2
    g_mid = run_company_arm(row, "hold", SEEDS,
                            patches={"churn_override": mid})["median_growth"]
    return mid, g_mid


def main() -> None:
    panel = load_panel()
    ct = company_table(panel).set_index("ticker")
    rows = []
    for ticker in pick_companies():
        row, _ = pick_init(panel[panel.ticker == ticker])
        actual = ct.loc[ticker, "actual_4q_growth"]

        base = run_company_arm(row, "hold", SEEDS)
        a = run_company_arm(row, "hold", SEEDS, patches={"zero_acquisition": True})
        b1 = run_company_arm(row, "hold", SEEDS,
                             patches={"churn_override": BAND_MEDIAN_CHURN})
        c = run_company_arm(row, "hold", SEEDS, patches={"zero_expansion": True})
        ic, ic_growth = implied_churn(row, actual)

        rows.append(dict(
            ticker=ticker,
            init_revenue=ct.loc[ticker, "init_revenue"],
            actual_4q_growth=actual,
            baseline_growth=base["median_growth"],
            baseline_error_pp=(base["median_growth"] - actual) * 100,
            a_zero_acquisition_growth=a["median_growth"],
            a_contribution_pp=(base["median_growth"] - a["median_growth"]) * 100,
            b1_band_median_churn_growth=b1["median_growth"],
            b1_contribution_pp=(base["median_growth"] - b1["median_growth"]) * 100,
            c_zero_expansion_growth=c["median_growth"],
            c_contribution_pp=(base["median_growth"] - c["median_growth"]) * 100,
            b2_implied_flat_churn=ic,
            b2_growth_at_implied=ic_growth,
            deaths_baseline=base["deaths"]))
        r = rows[-1]
        print(f"{ticker}: err {r['baseline_error_pp']:+.1f}pp | acq {r['a_contribution_pp']:.1f} "
              f"| churn-mult {r['b1_contribution_pp']:.1f} | expansion {r['c_contribution_pp']:.1f} "
              f"| implied churn {ic:.3%}")

    df = pd.DataFrame(rows)
    df.to_csv(CAL_DIR / "d2_error_attribution.csv", index=False)
    print("\nmedian contributions (pp): acquisition "
          f"{df.a_contribution_pp.median():.1f}, churn-multiplier-stack "
          f"{df.b1_contribution_pp.median():.1f}, expansion {df.c_contribution_pp.median():.1f}; "
          f"median baseline error {df.baseline_error_pp.median():.1f}")


if __name__ == "__main__":
    main()
