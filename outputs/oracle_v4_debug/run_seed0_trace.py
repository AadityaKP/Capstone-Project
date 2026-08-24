import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from config import sim_config
from simulation_runner import run_simulation


OUT_DIR = Path("outputs/oracle_v4_debug")
OUT_DIR.mkdir(parents=True, exist_ok=True)

df, traces = run_simulation(
    policy="oracle_v4_causal_hetero",
    num_episodes=1,
    seed_start=0,
    oracle_frequency=5,
    return_action_trace=True,
    return_monthly_trace=True,
)

action_trace = traces["action_trace"]
monthly_trace = traces["monthly_trace"]
monthly_by_month = {int(row["month"]): row for row in monthly_trace}

rows = []
cash_before = float(sim_config.INITIAL_CASH)

print("TRACE_DEBUG_START", flush=True)
for row in action_trace:
    month = int(row["month"])
    if month > 15:
        continue

    monthly = monthly_by_month.get(month, {})
    decision_trace = row.get("decision_trace") or {}
    cash_after = monthly.get("cash")

    proposals = []
    for proposal in decision_trace.get("proposals") or []:
        proposals.append(
            {
                "agent": proposal.get("agent"),
                "actions": proposal.get("actions"),
                "causal_confidence": proposal.get("causal_confidence"),
                "base_score": proposal.get("base_score"),
                "final_confidence": proposal.get("final_confidence"),
            }
        )

    debug_row = {
        "month": month,
        "cash_before": cash_before,
        "cash_after": cash_after,
        "proposal_source": decision_trace.get("proposal_source"),
        "causal_stress_node": decision_trace.get("causal_stress_node"),
        "pre_modifier_action": decision_trace.get("pre_modifier_action"),
        "post_modifier_action": decision_trace.get("post_modifier_action"),
        "final_action": decision_trace.get("final_action"),
        "proposals": proposals,
    }
    rows.append(debug_row)
    print(json.dumps(debug_row, sort_keys=True), flush=True)

    if cash_after is not None:
        cash_before = float(cash_after)

payload = {
    "results": df.to_dict(orient="records"),
    "trace": rows,
}
(OUT_DIR / "seed0_trace.json").write_text(
    json.dumps(payload, indent=2),
    encoding="utf-8",
)
print("TRACE_DEBUG_END", flush=True)
