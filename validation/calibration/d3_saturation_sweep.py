"""D3: identifiability sweep of SATURATION_ACQUISITION_RATE, on CAL only.

Same 6 companies as D2. Sweep the one assumed free parameter of the
scale-aware marketing curve over 4 orders of magnitude (log grid, 15 points,
centered on the current 0.20 being 'several times too high'), hold arm, 10
seeds. If |growth error| vs the constant has no clear interior minimum across
companies, fitting would be fake and the protocol says STOP.

Writes validation/calibration/d3_saturation_sweep.csv and .png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import CAL_DIR, load_panel, pick_init, run_company_arm  # noqa: E402
from d2_error_attribution import pick_companies  # noqa: E402

SEEDS = range(10)
GRID = np.logspace(np.log10(2e-4), np.log10(2.0), 15)


def main() -> None:
    panel = load_panel()
    from backtest_lib import company_table
    ct = company_table(panel).set_index("ticker")
    rows = []
    for ticker in pick_companies():
        row, _ = pick_init(panel[panel.ticker == ticker])
        actual = ct.loc[ticker, "actual_4q_growth"]
        for s in GRID:
            out = run_company_arm(row, "hold", SEEDS,
                                  patches={"saturation_rate": float(s)})
            rows.append(dict(ticker=ticker, saturation_rate=float(s),
                             median_growth=out["median_growth"],
                             deaths=out["deaths"],
                             actual_4q_growth=actual,
                             abs_error_pp=abs(out["median_growth"] - actual) * 100
                             if np.isfinite(out["median_growth"]) else np.nan))
        print(f"{ticker} done")

    df = pd.DataFrame(rows)
    df.to_csv(CAL_DIR / "d3_saturation_sweep.csv", index=False)

    med = df.groupby("saturation_rate").abs_error_pp.median()
    fig, ax = plt.subplots(figsize=(8, 5))
    for ticker, sub in df.groupby("ticker"):
        ax.plot(sub.saturation_rate, sub.abs_error_pp, alpha=0.4, marker=".",
                label=ticker)
    ax.plot(med.index, med.values, color="black", lw=2.5, marker="o",
            label="median (6 companies)")
    ax.axvline(0.20, color="red", ls="--", lw=1, label="current 0.20")
    ax.set_xscale("log")
    ax.set_xlabel("SATURATION_ACQUISITION_RATE")
    ax.set_ylabel("|4q growth error| (pp), hold arm, 10 seeds")
    ax.set_title("D3: identifiability of the marketing saturation constant (CAL)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(CAL_DIR / "d3_saturation_sweep.png", dpi=150)

    print("\nmedian |error| by rate:")
    print(med.to_string(float_format="%.1f"))
    print(f"\nargmin: {med.idxmin():.4g} -> {med.min():.1f}pp "
          f"(vs {med.loc[GRID[np.argmin(np.abs(GRID - 0.2))]]:.1f}pp near 0.20)")


if __name__ == "__main__":
    main()
