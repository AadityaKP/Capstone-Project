"""S8: case-study candidate selection from the RECORDED A3 live-replication
traces (oracle_v3 vs oracle_v3_no_memory, 20 matched seeds, freq 10).

Frozen ranking rule (stated in the write-up; no cherry-picking beyond it):
a (seed, month) qualifies when
  1. the brief label differed between the arms at that month
     (risk_level, growth_outlook, innovation_urgency or expected_outcome),
  2. marketing or R&D spend differed by > 20% (relative to the no-memory arm), and
  3. the oracle_v3 arm's MRR is higher than the no-memory arm's 6 months later.
Candidates are ranked by that 6-months-later MRR advantage (%).

Prints the top 5 and writes case_study_candidates.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A3 = ROOT / "validation/results/a3"
OUT = ROOT / "validation/round2"

BRIEF_COLS = ["risk_level", "growth_outlook", "innovation_urgency", "expected_outcome"]


def main() -> None:
    a_v3 = pd.read_csv(A3 / "actions_oracle_v3.csv")
    a_nm = pd.read_csv(A3 / "actions_oracle_v3_no_memory.csv")
    m_v3 = pd.read_csv(A3 / "monthly_oracle_v3.csv")
    m_nm = pd.read_csv(A3 / "monthly_oracle_v3_no_memory.csv")

    act = a_v3.merge(a_nm, on=["seed", "month"], suffixes=("_v3", "_nm"))
    mrr = m_v3[["seed", "month", "mrr"]].merge(
        m_nm[["seed", "month", "mrr"]], on=["seed", "month"],
        suffixes=("_v3", "_nm"))

    rows = []
    for r in act.itertuples():
        brief_diff = [c for c in BRIEF_COLS
                      if getattr(r, f"{c}_v3") != getattr(r, f"{c}_nm")]
        if not brief_diff:
            continue
        mkt_rel = abs(r.mkt_spend_v3 - r.mkt_spend_nm) / max(r.mkt_spend_nm, 1.0)
        rd_rel = abs(r.rd_spend_v3 - r.rd_spend_nm) / max(r.rd_spend_nm, 1.0)
        if max(mkt_rel, rd_rel) <= 0.20:
            continue
        fut = mrr[(mrr.seed == r.seed) & (mrr.month == r.month + 6)]
        if not len(fut):
            continue
        adv = float(fut.mrr_v3.iloc[0] / max(fut.mrr_nm.iloc[0], 1.0) - 1.0)
        if adv <= 0:
            continue
        rows.append(dict(seed=r.seed, month=r.month,
                         brief_fields_differing=";".join(brief_diff),
                         risk_v3=r.risk_level_v3, risk_nm=r.risk_level_nm,
                         growth_v3=r.growth_outlook_v3, growth_nm=r.growth_outlook_nm,
                         mkt_v3=r.mkt_spend_v3, mkt_nm=r.mkt_spend_nm,
                         rd_v3=r.rd_spend_v3, rd_nm=r.rd_spend_nm,
                         spend_divergence=max(mkt_rel, rd_rel),
                         memory_count_v3=r.memory_count_v3,
                         mrr_adv_6mo_pct=adv * 100))
    df = pd.DataFrame(rows).sort_values("mrr_adv_6mo_pct", ascending=False)
    df.to_csv(OUT / "case_study_candidates.csv", index=False)
    print(f"{len(df)} qualifying (seed, month) candidates")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
