"""A3 runner: matched-seed oracle-value experiment (LLM arms).

boardroom / oracle_v1 / oracle_v3 / oracle_v3_no_memory, 20 seeds x 120 months,
oracle_frequency=10 (the FULL run's cadence), environment_config
{"deterministic_rng": True} so all arms share the world at equal seed.
llama3.1:8b via Ollama at temperature 0.

Writes per-arm episode metrics and action traces under validation/results/a3/.
Analysis lives in a3_oracle_value_analyze.py (run after this completes).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulation_runner import run_simulation

OUT = ROOT / "validation/results/a3"
OUT.mkdir(parents=True, exist_ok=True)

N_EPISODES = 20
FREQ = 10
ENV = {"deterministic_rng": True}
POLICIES = ["boardroom", "oracle_v1", "oracle_v3", "oracle_v3_no_memory"]


def main():
    for policy in POLICIES:
        t0 = time.time()
        df, traces = run_simulation(
            policy=policy, num_episodes=N_EPISODES, seed_start=0,
            oracle_frequency=FREQ, environment_config=ENV,
            return_action_trace=True, return_monthly_trace=True,
        )
        df.to_csv(OUT / f"episodes_{policy}.csv", index=False)
        monthly = pd.DataFrame([
            {k: v for k, v in row.items() if k not in ("brief", "decision_trace")}
            for row in traces["monthly_trace"]
        ])
        monthly.to_csv(OUT / f"monthly_{policy}.csv", index=False)
        actions = []
        for row in traces["action_trace"]:
            trace = row.get("decision_trace") or {}
            brief = trace.get("brief") or row.get("brief") or {}
            actions.append({
                "episode": row["episode"], "seed": row["seed"], "month": row["month"],
                "mkt_spend": row["action"]["marketing"]["spend"],
                "rd_spend": row["action"]["product"]["r_and_d_spend"],
                "hires": row["action"]["hiring"]["hires"],
                "brief_source": trace.get("brief_source"),
                "refresh_reason": trace.get("refresh_reason"),
                "risk_level": (brief or {}).get("risk_level"),
                "growth_outlook": (brief or {}).get("growth_outlook"),
                "innovation_urgency": (brief or {}).get("innovation_urgency"),
                "expected_outcome": (brief or {}).get("expected_outcome"),
                "confidence": (brief or {}).get("confidence"),
                "memory_count": trace.get("memory_count"),
            })
        pd.DataFrame(actions).to_csv(OUT / f"actions_{policy}.csv", index=False)
        meta = dict(policy=policy, episodes=N_EPISODES, freq=FREQ, env=ENV,
                    wall_seconds=round(time.time() - t0, 1),
                    llm_calls_total=int(df.llm_calls.sum()))
        (OUT / f"meta_{policy}.json").write_text(json.dumps(meta))
        print(f"DONE {policy}: {meta}")


if __name__ == "__main__":
    main()
