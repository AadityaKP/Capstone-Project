"""Phase 4: the ONLY HOLDOUT touch. v2 backtest + frozen-criteria evaluation.

All 39 companies are re-run (CAL rows labelled for transparency), but every
criterion is evaluated on HOLDOUT at the primary $250 mapping exactly as
frozen in PROTOCOL.md. v2 flags: marketing_curve="v2" (CAL-fitted),
financing_enabled=True, corridor="scale_aware" for heuristic and boardroom.
Arms and decomposition identical to the v1 backtest: hold / noop / heuristic /
boardroom, 30 matched seeds, 12 months.

Writes:
  validation/results/real_company_backtest_v2.csv   (never touches the v1 file)
  validation/calibration/p4_criteria_verdicts.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import (CAL_DIR, ROOT, V1_ALL_SEED_HOLD_DEATHS,  # noqa: E402
                          company_table, load_panel, load_split, pick_init,
                          run_company_arm)

RESULTS = ROOT / "validation/results"
SEEDS = range(30)
ARMS = ("hold", "noop", "heuristic", "boardroom")
V2_ENV = {"marketing_curve": "v2", "financing_enabled": True,
          "competitive_entry": "scale_neutral"}


def main() -> None:
    panel = load_panel()
    split = load_split().set_index("ticker")
    rows = []
    for ticker in sorted(split.index):
        row, future = pick_init(panel[panel.ticker == ticker])
        if row is None:
            continue
        actual = split.loc[ticker, "actual_4q_growth"]
        for price in (250.0, 50.0):
            arm_out = {}
            for arm in ARMS:
                corridor = "scale_aware" if arm in ("heuristic", "boardroom") else "legacy"
                arm_out[arm] = run_company_arm(
                    row, arm, SEEDS, price=price,
                    extra_env_config=dict(V2_ENV), corridor=corridor)
            hold_v = np.array(arm_out["hold"]["growths"])
            board_v = np.array(arm_out["boardroom"]["growths"])
            n_pair = min(len(hold_v), len(board_v))
            rows.append(dict(
                physics_version="v2",
                ticker=ticker, split=split.loc[ticker, "split"],
                init_quarter=row.fiscal_period, price_assumed=price,
                actual_4q_growth=actual,
                sim_hold_median=arm_out["hold"]["median_growth"],
                sim_noop_median=arm_out["noop"]["median_growth"],
                sim_heuristic_median=arm_out["heuristic"]["median_growth"],
                sim_boardroom_median=arm_out["boardroom"]["median_growth"],
                deaths_hold=arm_out["hold"]["deaths"],
                deaths_boardroom=arm_out["boardroom"]["deaths"],
                raises_hold=arm_out["hold"]["n_financing_raises"],
                retrodiction_error=(arm_out["hold"]["median_growth"] - actual),
                agent_increment_median=(float(np.median(board_v[:n_pair] - hold_v[:n_pair]))
                                        if n_pair else np.nan)))
            r = rows[-1]
            print(f"{ticker} [{r['split']}] p={price:.0f}: actual={actual:+.1%} "
                  f"hold={r['sim_hold_median']:+.1%} board={r['sim_boardroom_median']:+.1%} "
                  f"err={r['retrodiction_error']:+.1%} deaths_hold={r['deaths_hold']}")

    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "real_company_backtest_v2.csv", index=False)

    # ---------------- frozen criteria, HOLDOUT, price=250 ----------------
    ho = res[(res.split == "HOLDOUT") & (res.price_assumed == 250.0)]
    ev = ho.dropna(subset=["retrodiction_error"])

    med_abs_err = float(ev.retrodiction_error.abs().median() * 100)
    c1_verdict = ("PASS" if med_abs_err <= 10 else
                  "PARTIAL" if med_abs_err <= 20 else "FAIL")
    sign_agree = float((np.sign(ev.sim_hold_median)
                        == np.sign(ev.actual_4q_growth)).mean())
    sign_verdict = "PASS" if sign_agree >= 0.70 else "FAIL"

    bo = ho.dropna(subset=["sim_boardroom_median"])
    std_board = float(bo.sim_boardroom_median.std(ddof=1))
    std_actual = float(bo.actual_4q_growth.std(ddof=1))
    rho = stats.spearmanr(ev.sim_hold_median, ev.actual_4q_growth)
    corridor_ok = (std_board >= std_actual / 3) and (rho.statistic > 0.3)

    # financing check: all 6 v1 all-seed hold deaths, both splits, price 250
    fin = res[(res.price_assumed == 250.0)
              & (res.ticker.isin(V1_ALL_SEED_HOLD_DEATHS))]
    n_seeds = len(list(SEEDS))
    fin_survive = (fin.deaths_hold < n_seeds / 2)
    fin_frac = float(fin_survive.mean())
    fin_verdict = "PASS" if fin_frac >= 0.8 else "FAIL"

    verdicts = dict(
        C1_v2_median_abs_error_pp=med_abs_err,
        C1_v2_verdict=c1_verdict,
        growth_sign_agreement=sign_agree,
        growth_sign_verdict=sign_verdict,
        corridor_std_boardroom=std_board,
        corridor_std_actual=std_actual,
        corridor_std_ratio=std_board / std_actual if std_actual else np.nan,
        corridor_spearman_hold_vs_actual=float(rho.statistic),
        corridor_spearman_p=float(rho.pvalue),
        corridor_verdict="PASS" if corridor_ok else "FAIL",
        financing_survivor_fraction=fin_frac,
        financing_detail={t: f"{int(d)}/{n_seeds} deaths" for t, d in
                          zip(fin.ticker, fin.deaths_hold)},
        financing_verdict=fin_verdict,
        n_holdout_evaluable=int(len(ev)),
        holdout_median_error_pp_signed=float(ev.retrodiction_error.median() * 100),
        cal_median_abs_error_pp=float(
            res[(res.split == "CAL") & (res.price_assumed == 250.0)]
            .dropna(subset=["retrodiction_error"]).retrodiction_error.abs().median() * 100),
        price50_sensitivity_median_abs_error_pp=float(
            res[(res.split == "HOLDOUT") & (res.price_assumed == 50.0)]
            .dropna(subset=["retrodiction_error"]).retrodiction_error.abs().median() * 100),
    )
    (CAL_DIR / "p4_criteria_verdicts.json").write_text(json.dumps(verdicts, indent=2))
    print("\n" + json.dumps(verdicts, indent=2, default=str))


if __name__ == "__main__":
    main()
