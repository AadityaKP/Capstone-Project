"""F1: fit the single free parameter of the v2 marketing curve, on CAL only.

Replays each of the 20 CAL companies' real S&M spend (hold arm, 10 matched
seeds) across a grid of SATURATION_ACQUISITION_RATE values refined around the
D3 basin, with the FULL v2 physics package (financing enabled - so the three
CAL companies that die of un-modelled financing contribute rather than being
silently dropped). Objective: median |4q cumulative growth error| across CAL.

Uncertainty: bootstrap over CAL companies (2,000 resamples of the per-company
|error| matrix; argmin per resample -> percentile CI). Sensitivity (reported,
NOT fit, per D2's mandate): CAL loss at the fitted rate under the churn-band
medians 2.0%/2.7%/3.4%.

Writes: f1_loss_curve.csv, f1_per_company_errors.csv, marketing_fit.md.
The fitted value is then transcribed into
business_logic.SATURATION_ACQUISITION_RATE_V2 (provenance comment points here).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import (CAL_DIR, cal_tickers, company_table, load_panel,  # noqa: E402
                          pick_init, run_company_arm)

SEEDS = range(10)
GRID = np.unique(np.round(np.logspace(np.log10(0.01), np.log10(0.30), 25), 5))
CHURN_SENSITIVITY = [0.020, 0.027, 0.034]
N_BOOT = 2000
BOOT_SEED = 20260904


def cal_errors(saturation_rate: float, churn_override: float | None = None) -> pd.Series:
    """Per-CAL-company signed error (sim hold median growth - actual)."""
    panel = load_panel()
    ct = company_table(panel).set_index("ticker")
    errs = {}
    for ticker in cal_tickers():
        row, _ = pick_init(panel[panel.ticker == ticker])
        patches = {"churn_override": churn_override} if churn_override else None
        out = run_company_arm(
            row, "hold", SEEDS,
            extra_env_config={"saturation_acquisition_rate": float(saturation_rate),
                              "financing_enabled": True,
                              "competitive_entry": "scale_neutral"},
            patches=patches)
        errs[ticker] = out["median_growth"] - ct.loc[ticker, "actual_4q_growth"]
    return pd.Series(errs)


def main() -> None:
    per_company = {}
    rows = []
    for s in GRID:
        errs = cal_errors(s)
        per_company[s] = errs
        rows.append(dict(saturation_rate=s,
                         median_abs_error_pp=float(errs.abs().median() * 100),
                         mean_abs_error_pp=float(errs.abs().mean() * 100),
                         median_signed_error_pp=float(errs.median() * 100),
                         n_evaluable=int(errs.notna().sum())))
        print(f"s={s:.4f}: median|err|={rows[-1]['median_abs_error_pp']:.1f}pp "
              f"(n={rows[-1]['n_evaluable']})")

    loss = pd.DataFrame(rows)
    loss.to_csv(CAL_DIR / "f1_loss_curve.csv", index=False)
    err_mat = pd.DataFrame(per_company)  # companies x grid
    err_mat.to_csv(CAL_DIR / "f1_per_company_errors.csv")

    fitted = float(loss.loc[loss.median_abs_error_pp.idxmin(), "saturation_rate"])
    fit_loss = float(loss.median_abs_error_pp.min())

    # bootstrap CI over companies
    rng = np.random.default_rng(BOOT_SEED)
    abs_mat = err_mat.abs().to_numpy()  # (n_companies, n_grid)
    n_co = abs_mat.shape[0]
    argmins = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_co, n_co)
        med = np.nanmedian(abs_mat[idx], axis=0)
        argmins.append(err_mat.columns[int(np.nanargmin(med))])
    argmins = np.array(argmins, dtype=float)
    ci_lo, ci_hi = np.percentile(argmins, [2.5, 97.5])

    # churn sensitivity at the fitted rate (reported, not fit)
    churn_rows = []
    for c in CHURN_SENSITIVITY:
        errs = cal_errors(fitted, churn_override=c)
        churn_rows.append(dict(flat_churn=c,
                               median_abs_error_pp=float(errs.abs().median() * 100),
                               median_signed_error_pp=float(errs.median() * 100)))
    churn_df = pd.DataFrame(churn_rows)

    md = f"""# F1 marketing-curve fit (CAL only)

- **Fitted `SATURATION_ACQUISITION_RATE_V2` = {fitted:.4f}** (was 0.20, ASSUMED)
- CAL median |4q growth error| at fit: **{fit_loss:.1f} pp** (v1 assumed value's
  loss on the same grid: {float(loss[loss.saturation_rate >= 0.19].median_abs_error_pp.iloc[0]) if (loss.saturation_rate >= 0.19).any() else float('nan'):.1f} pp at the nearest grid point)
- 95% bootstrap CI over CAL companies (2,000 resamples, argmin per resample):
  **[{ci_lo:.4f}, {ci_hi:.4f}]**
- Protocol: hold arm, 10 matched seeds, financing_enabled=True (so the three
  CAL companies with un-modelled financing deaths contribute), split per
  `panel_split.csv`, HOLDOUT untouched.
- Loss curve: `f1_loss_curve.csv`; per-company signed errors:
  `f1_per_company_errors.csv`.

## Churn-band sensitivity at the fitted rate (reported, NOT fit - D2 showed
churn is a minor term; the band median is an assumption of the mapping)

{churn_df.to_string(index=False)}

## Loss curve

{loss.to_string(index=False)}
"""
    (CAL_DIR / "marketing_fit.md").write_text(md)
    print(f"\nFITTED: {fitted:.4f}, CI [{ci_lo:.4f}, {ci_hi:.4f}], "
          f"loss {fit_loss:.1f}pp")


if __name__ == "__main__":
    main()
