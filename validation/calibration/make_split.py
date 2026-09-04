"""Write the CAL/HOLDOUT panel split BEFORE any fitting (protocol rule 2).

Stratified by revenue-scale tercile (initialization-quarter revenue) and by
actual 4q growth sign. Fixed seed. CAL ~20 / HOLDOUT ~19. Deterministic:
within each stratum companies are shuffled with the seeded RNG and assigned
alternately, the starting side alternating per stratum; totals are then
balanced to exactly 20/19 by flipping RNG-chosen members of the over-full side.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import CAL_DIR, company_table  # noqa: E402

SPLIT_SEED = 20260904
TARGET_CAL = 20


def main() -> None:
    ct = company_table().sort_values("ticker").reset_index(drop=True)
    terc = pd.qcut(ct.init_revenue, 3, labels=["small", "mid", "large"])
    ct["revenue_tercile"] = terc.astype(str)
    ct["growth_sign"] = np.where(ct.actual_4q_growth >= 0, "pos", "neg")
    ct["stratum"] = ct.revenue_tercile + "_" + ct.growth_sign

    rng = np.random.default_rng(SPLIT_SEED)
    ct["split"] = ""
    for i, (_, idx) in enumerate(sorted(ct.groupby("stratum").groups.items())):
        order = rng.permutation(np.array(sorted(idx)))
        sides = ["CAL", "HOLDOUT"] if i % 2 == 0 else ["HOLDOUT", "CAL"]
        for j, row_i in enumerate(order):
            ct.loc[row_i, "split"] = sides[j % 2]

    n_cal = (ct.split == "CAL").sum()
    while n_cal != TARGET_CAL:
        over = "CAL" if n_cal > TARGET_CAL else "HOLDOUT"
        pool = ct.index[ct.split == over].to_numpy()
        flip = rng.choice(pool)
        ct.loc[flip, "split"] = "HOLDOUT" if over == "CAL" else "CAL"
        n_cal = (ct.split == "CAL").sum()

    out = ct[["ticker", "init_quarter", "init_revenue", "actual_4q_growth",
              "revenue_tercile", "growth_sign", "stratum", "split"]]
    out.to_csv(CAL_DIR / "panel_split.csv", index=False)
    print(out.groupby(["split", "revenue_tercile"]).size().unstack(fill_value=0))
    print(out.groupby(["split", "growth_sign"]).size().unstack(fill_value=0))
    print(f"\nCAL={n_cal} HOLDOUT={(out.split == 'HOLDOUT').sum()}")
    print("CAL:", " ".join(sorted(out[out.split == 'CAL'].ticker)))
    print("HOLDOUT:", " ".join(sorted(out[out.split == 'HOLDOUT'].ticker)))


if __name__ == "__main__":
    main()
