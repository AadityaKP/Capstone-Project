"""S8 (part 2): replay episodes 0..15 for oracle_v3 and oracle_v3_no_memory to
recover the decision-level evidence at the top-ranked case-study point
(seed 15, month 60 - selected by the frozen rule in case_study_select.py).

Episodes 0..14 must be replayed because the memory store and the brief cache
accrue across episodes within a run; a lone seed-15 rerun would not reproduce
the recorded decision. Fidelity check: the replayed episode-15 final MRR must
match the recorded episodes_*.csv rows.

Writes validation/round2/case_study_traces.json (retrieved memories, brief,
pre/post-modifier and final actions for months 54-72 of seed 15, both arms).
LLM job - queued. The figure and write-up are produced by case_study_report.py
from RECORDED monthly traces + these quotes.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulation_runner import run_simulation  # noqa: E402

OUT = ROOT / "validation/round2"
A3 = ROOT / "validation/results/a3"
SEED = 15
WINDOW = range(54, 73)
POLICIES = ["oracle_v3", "oracle_v3_no_memory"]


def main() -> None:
    payload = {}
    for policy in POLICIES:
        df, traces = run_simulation(
            policy=policy, num_episodes=SEED + 1, seed_start=0,
            oracle_frequency=10, environment_config={"deterministic_rng": True},
            return_action_trace=True, return_retrieval_trace=True,
        )
        rec = pd.read_csv(A3 / f"episodes_{policy}.csv")
        rec_final = float(rec[rec.seed == SEED].final_mrr.iloc[0])
        new_final = float(df[df.seed == SEED].final_mrr.iloc[0])
        fidelity = abs(new_final - rec_final) / max(rec_final, 1.0)
        print(f"{policy}: replay episode-{SEED} final MRR {new_final:,.0f} vs "
              f"recorded {rec_final:,.0f} (rel diff {fidelity:.2e})", flush=True)

        rows = []
        for row in traces["action_trace"]:
            if row["seed"] != SEED or row["month"] not in WINDOW:
                continue
            t = row.get("decision_trace") or {}
            rows.append(dict(
                month=row["month"],
                brief=t.get("brief"),
                brief_source=t.get("brief_source"),
                refresh_reason=t.get("refresh_reason"),
                retrieved_memories=t.get("retrieved_memories"),
                pre_modifier_action=t.get("pre_modifier_action"),
                post_modifier_action=t.get("post_modifier_action"),
                final_action=t.get("final_action"),
            ))
        payload[policy] = dict(fidelity_rel_diff=fidelity,
                               recorded_final_mrr=rec_final,
                               replay_final_mrr=new_final,
                               months=rows)
    (OUT / "case_study_traces.json").write_text(
        json.dumps(payload, indent=2, default=str))
    print("wrote case_study_traces.json")


if __name__ == "__main__":
    main()
