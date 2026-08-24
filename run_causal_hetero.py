from __future__ import annotations

import os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from simulation_runner import run_simulation
import pandas as pd

df, monthly_trace = run_simulation(
    policy="oracle_v4_causal_hetero",
    num_episodes=20,
    seed_start=0,
    oracle_frequency=5,
    oracle_overrides={},
    return_monthly_trace=True,
)

out_dir = Path("outputs/oracle_v4_compare_final_v2")
out_dir.mkdir(parents=True, exist_ok=True)

df.to_csv(out_dir / "episode_metrics.csv", index=False)
pd.DataFrame(monthly_trace).to_csv(out_dir / "monthly_trace.csv", index=False)

print("=== EPISODE SUMMARY ===")
cols = ["episode", "seed", "cause", "steps", "final_mrr", "final_cash", "total_reward", "llm_calls", "proposal_llm_calls"]
print(df[[c for c in cols if c in df.columns]].to_string(index=False))
print()
print("Survival Rate:", f"{(df['cause'] == 'Time Limit').mean():.2%}")
print("Avg Final MRR:", f"${df['final_mrr'].mean():,.0f}")
print("Avg Total Reward:", f"{df['total_reward'].mean():.2f}")
print("Avg LLM calls:", f"{df['llm_calls'].mean():.1f}")
if "proposal_llm_calls" in df.columns:
    print("Avg Proposal LLM calls:", f"{df['proposal_llm_calls'].mean():.1f}")
print(f"Outputs saved to: {out_dir.resolve()}")
