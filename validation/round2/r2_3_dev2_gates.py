"""R2-3 gates (PROTOCOL_round2.md), DEV2 only. EVAL2 is NOT touched here.

1. R2-REG (part): re-run the round-1 HOLDOUT backtest configuration with
   round-1 flags on this branch and diff against the committed
   real_company_backtest_v2.csv HOLDOUT rows - must be identical. Written to
   validation/round2/ only; the round-1 file is never modified.
2. R2-VAR precondition: LTV:CAC spread at initialization on DEV2 under
   mapping_version="v2" (need IQR >= 0.5).
3. DEV2 hold-arm sanity: 10 seeds, mapping v2 + opportunistic financing -
   raises fire, nothing is NaN.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation/calibration"))

from backtest_lib import (V1_ALL_SEED_HOLD_DEATHS, holdout_tickers,  # noqa: E402
                          load_panel, load_split, pick_init, run_company_arm)
from eval2_lib import load_eval2_states, load_hazard, run_eval2_company_arm  # noqa: E402

OUT = ROOT / "validation/round2"
SEEDS = range(30)
ARMS = ("hold", "noop", "heuristic", "boardroom")
ROUND1_V2_ENV = {"marketing_curve": "v2", "financing_enabled": True,
                 "competitive_entry": "scale_neutral"}


def r2_reg_holdout_diff() -> bool:
    panel = load_panel()
    rec = pd.read_csv(ROOT / "validation/results/real_company_backtest_v2.csv")
    rec = rec[rec.split == "HOLDOUT"]
    rows = []
    for ticker in sorted(holdout_tickers()):
        row, future = pick_init(panel[panel.ticker == ticker])
        for price in (250.0, 50.0):
            arm_out = {}
            for arm in ARMS:
                corridor = "scale_aware" if arm in ("heuristic", "boardroom") else "legacy"
                arm_out[arm] = run_company_arm(
                    row, arm, SEEDS, price=price,
                    extra_env_config=dict(ROUND1_V2_ENV), corridor=corridor)
            rows.append(dict(ticker=ticker, price_assumed=price,
                             sim_hold_median=arm_out["hold"]["median_growth"],
                             sim_noop_median=arm_out["noop"]["median_growth"],
                             sim_heuristic_median=arm_out["heuristic"]["median_growth"],
                             sim_boardroom_median=arm_out["boardroom"]["median_growth"],
                             deaths_hold=arm_out["hold"]["deaths"],
                             deaths_boardroom=arm_out["boardroom"]["deaths"]))
        print(f"  rep {ticker} done", flush=True)
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "r2_reg_holdout_reproduction.csv", index=False)
    m = rec.merge(rep, on=["ticker", "price_assumed"], suffixes=("_rec", "_new"))
    ok = True
    for col in ("sim_hold_median", "sim_noop_median", "sim_heuristic_median",
                "sim_boardroom_median", "deaths_hold", "deaths_boardroom"):
        a = m[f"{col}_rec"].to_numpy(dtype=float)
        b = m[f"{col}_new"].to_numpy(dtype=float)
        col_ok = bool(np.allclose(a, b, rtol=0, atol=1e-9, equal_nan=True))
        print(f"  R2-REG {col}: {'OK' if col_ok else 'MISMATCH'}")
        ok &= col_ok
    return ok


def r2_var_precondition() -> tuple[bool, float]:
    cac = pd.read_csv(ROOT / "validation/calibration/cac_mapping_r2.csv")
    dev = cac[(cac.split == "DEV2") & (cac.price_assumed == 250.0)].dropna(
        subset=["ltv_cac_v2"])
    iqr = float(dev.ltv_cac_v2.quantile(0.75) - dev.ltv_cac_v2.quantile(0.25))
    print(f"  R2-VAR: DEV2 LTV:CAC at init - n={len(dev)}, "
          f"median {dev.ltv_cac_v2.median():.2f}, IQR {iqr:.2f} (need >= 0.5), "
          f"range [{dev.ltv_cac_v2.min():.2f}, {dev.ltv_cac_v2.max():.2f}]")
    return iqr >= 0.5, iqr


def dev2_hold_sanity() -> bool:
    ev = load_eval2_states()
    hz = load_hazard()
    dev = ev[(ev.split == "DEV2") & (ev.price_assumed == 250.0)]
    ok = True
    total_raises = 0
    for row in dev.itertuples():
        out = run_eval2_company_arm(row, "hold", range(10),
                                    mapping_version="v2", financing=True,
                                    financing_model="opportunistic", hazard=hz)
        total_raises += out["n_financing_raises"]
        if out["growths"] and not all(np.isfinite(out["growths"])):
            print(f"  NaN growth for {row.ticker}")
            ok = False
    print(f"  DEV2 hold sanity: 20 companies x 10 seeds, "
          f"{total_raises} raises fired, all growths finite: {ok}")
    return ok and total_raises > 0


def main() -> None:
    print("R2-VAR precondition (DEV2):")
    var_ok, _ = r2_var_precondition()
    print("\nDEV2 hold-arm sanity (mapping v2 + opportunistic financing):")
    sane = dev2_hold_sanity()
    print("\nR2-REG round-1 HOLDOUT reproduction (round-1 flags, this branch):")
    reg_ok = r2_reg_holdout_diff()
    print(f"\nR2-3 gates: R2-VAR {'PASS' if var_ok else 'FAIL'} | "
          f"DEV2 sanity {'PASS' if sane else 'FAIL'} | "
          f"R2-REG {'EXACT' if reg_ok else 'MISMATCH'}")
    if not (var_ok and sane and reg_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
