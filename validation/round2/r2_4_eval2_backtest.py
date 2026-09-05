"""R2-4: EVAL2 - run ONCE (PROTOCOL_round2.md). Verdicts computed by script.

4 arms x 30 matched seeds x 12 months x {financing on/off} x {$250, $50},
mapping_version="v2", financing_model="opportunistic" when on. DEV2 rows are
run alongside and labelled (any number of DEV2 runs is allowed; criteria use
EVAL2 at $250 only). Outputs:
  validation/results/real_company_backtest_r2.csv
  validation/round2/r2_criteria_verdicts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "validation/calibration"))

from eval2_lib import load_eval2_states, load_hazard, run_eval2_company_arm  # noqa: E402

RESULTS = ROOT / "validation/results"
OUT = ROOT / "validation/round2"
SEEDS = range(30)
ARMS = ("hold", "noop", "heuristic", "boardroom")


def main() -> None:
    ev = load_eval2_states()
    hz = load_hazard()
    rows = []
    for row in ev.itertuples():
        for financing in (True, False):
            arm_out = {}
            for arm in ARMS:
                arm_out[arm] = run_eval2_company_arm(
                    row, arm, SEEDS, mapping_version="v2",
                    financing=financing, financing_model="opportunistic",
                    hazard=hz)
            hold_v = np.array(arm_out["hold"]["growths"])
            board_v = np.array(arm_out["boardroom"]["growths"])
            n_pair = min(len(hold_v), len(board_v))
            rows.append(dict(
                physics_version="v2", mapping_version="v2",
                financing_model="opportunistic" if financing else "off",
                ticker=row.ticker, split=row.split,
                init_quarter=row.init_quarter, price_assumed=row.price_assumed,
                actual_4q_growth=row.actual_4q_growth,
                cac_v2=row.cac_v2, cac_clamped=row.cac_clamped,
                sim_hold_median=arm_out["hold"]["median_growth"],
                sim_noop_median=arm_out["noop"]["median_growth"],
                sim_heuristic_median=arm_out["heuristic"]["median_growth"],
                sim_boardroom_median=arm_out["boardroom"]["median_growth"],
                deaths_hold=arm_out["hold"]["deaths"],
                deaths_boardroom=arm_out["boardroom"]["deaths"],
                raises_hold=arm_out["hold"]["n_financing_raises"],
                retrodiction_error=(arm_out["hold"]["median_growth"]
                                    - row.actual_4q_growth),
                agent_increment_median=(float(np.median(board_v[:n_pair]
                                                        - hold_v[:n_pair]))
                                        if n_pair else np.nan)))
            r = rows[-1]
            print(f"{row.ticker} [{row.split}] p={row.price_assumed:.0f} "
                  f"fin={'on' if financing else 'off'}: "
                  f"actual={row.actual_4q_growth:+.1%} "
                  f"hold={r['sim_hold_median']:+.1%} "
                  f"deaths_hold={r['deaths_hold']}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "real_company_backtest_r2.csv", index=False)

    # ---------------- frozen criteria: EVAL2, $250, financing on ------------
    on = res[(res.split == "EVAL2") & (res.price_assumed == 250.0)
             & (res.financing_model == "opportunistic")]
    off = res[(res.split == "EVAL2") & (res.price_assumed == 250.0)
              & (res.financing_model == "off")]
    n_seeds = len(list(SEEDS))
    evl = on.dropna(subset=["retrodiction_error"])

    med_abs = float(evl.retrodiction_error.abs().median() * 100)
    c1 = "PASS" if med_abs <= 10 else "PARTIAL" if med_abs <= 20 else "FAIL"
    sign = float((np.sign(evl.sim_hold_median)
                  == np.sign(evl.actual_4q_growth)).mean())
    bo = on.dropna(subset=["sim_boardroom_median"])
    std_b = float(bo.sim_boardroom_median.std(ddof=1))
    std_a = float(bo.actual_4q_growth.std(ddof=1))
    rho = stats.spearmanr(evl.sim_hold_median, evl.actual_4q_growth)
    corr_ok = (std_b >= std_a / 3) and (rho.statistic > 0.3)

    fin_a = float((on.deaths_hold < n_seeds / 2).mean())
    fin_a_v = "PASS" if fin_a >= 0.90 else "PARTIAL" if fin_a >= 0.75 else "FAIL"
    zero_surv_off = off[off.deaths_hold == n_seeds].ticker
    on_z = on[on.ticker.isin(zero_surv_off)]
    fin_b = float((on_z.deaths_hold < n_seeds / 2).mean()) if len(on_z) else np.nan
    fin_b_v = ("PASS" if (len(on_z) and fin_b >= 0.80) else
               "FAIL" if len(on_z) else "N/A (no zero-survival companies)")

    verdicts = dict(
        offset="q0+8",
        R2_C1_median_abs_error_pp=med_abs, R2_C1_verdict=c1,
        R2_C1_signed_median_pp=float(evl.retrodiction_error.median() * 100),
        R2_SIGN=sign, R2_SIGN_verdict="PASS" if sign >= 0.70 else "FAIL",
        R2_CORR_std_boardroom=std_b, R2_CORR_std_actual=std_a,
        R2_CORR_std_ratio=std_b / std_a if std_a else np.nan,
        R2_CORR_spearman=float(rho.statistic), R2_CORR_p=float(rho.pvalue),
        R2_CORR_verdict="PASS" if corr_ok else "FAIL",
        R2_FIN_a_share=fin_a, R2_FIN_a_verdict=fin_a_v,
        R2_FIN_b_zero_survival_companies=list(zero_surv_off),
        R2_FIN_b_share=None if np.isnan(fin_b) else fin_b,
        R2_FIN_b_verdict=fin_b_v,
        n_eval2_evaluable=int(len(evl)),
        dev2_median_abs_error_pp=float(
            res[(res.split == "DEV2") & (res.price_assumed == 250.0)
                & (res.financing_model == "opportunistic")]
            .dropna(subset=["retrodiction_error"]).retrodiction_error.abs()
            .median() * 100),
        price50_sensitivity_median_abs_error_pp=float(
            res[(res.split == "EVAL2") & (res.price_assumed == 50.0)
                & (res.financing_model == "opportunistic")]
            .dropna(subset=["retrodiction_error"]).retrodiction_error.abs()
            .median() * 100),
        boardroom_minus_hold_median_pp=float(
            on.agent_increment_median.median() * 100),
    )
    (OUT / "r2_criteria_verdicts.json").write_text(
        json.dumps(verdicts, indent=2, default=str))
    print("\n" + json.dumps(verdicts, indent=2, default=str))


if __name__ == "__main__":
    main()
