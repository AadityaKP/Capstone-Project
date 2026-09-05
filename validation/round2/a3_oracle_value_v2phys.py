"""A3 robustness under v2 physics (plan S3; decision D1). LLM job - queued.

boardroom and oracle_v3, 20 seeds (0-19, matching a3_oracle_value.csv),
oracle_frequency=10, v2 physics flags + scale-aware corridor on both arms,
llama3.1:8b. Writes validation/results/a3_v2phys/episodes_{policy}.csv and
the combined a3_oracle_value_v2phys.csv. Gate analysis lives in
gates_v2phys.py (run once, after this finishes).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulation_runner import run_simulation  # noqa: E402

OUT = ROOT / "validation/results/a3_v2phys"
OUT.mkdir(parents=True, exist_ok=True)
RESULTS = ROOT / "validation/results"

N_EPISODES = 20
FREQ = 10
V2_ENV = {"deterministic_rng": True, "marketing_curve": "v2",
          "competitive_entry": "scale_neutral"}
V2_AGENT = {"corridor": "scale_aware"}


def main() -> None:
    frames = []
    for policy in ("boardroom", "oracle_v3"):
        t0 = time.time()
        df, traces = run_simulation(
            policy=policy, num_episodes=N_EPISODES, seed_start=0,
            oracle_frequency=FREQ, environment_config=dict(V2_ENV),
            oracle_overrides=dict(V2_AGENT),
            return_action_trace=True, return_monthly_trace=False,
        )
        df["physics_version"] = "v2"
        df.to_csv(OUT / f"episodes_{policy}.csv", index=False)
        meta = dict(policy=policy, episodes=N_EPISODES, freq=FREQ,
                    env=V2_ENV, agent=V2_AGENT,
                    wall_seconds=round(time.time() - t0, 1),
                    llm_calls_total=int(df.llm_calls.sum()) if "llm_calls" in df else 0)
        (OUT / f"meta_{policy}.json").write_text(json.dumps(meta))
        frames.append(df)
        print(f"DONE {policy}: {meta}", flush=True)
    pd.concat(frames, ignore_index=True).to_csv(
        RESULTS / "a3_oracle_value_v2phys.csv", index=False)


if __name__ == "__main__":
    main()
