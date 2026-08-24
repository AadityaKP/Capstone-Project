import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from simulation_runner import run_simulation

OUT = Path(r"outputs/oracle_v4_compare/comparison.json")

configs = [
    ("oracle_v3", {}),
    ("oracle_v4_causal_hetero", {}),
]
results = {}
for policy, overrides in configs:
    print(f"RUN_START policy={policy}", flush=True)
    df, monthly = run_simulation(
        policy=policy,
        num_episodes=2,
        seed_start=0,
        oracle_frequency=5,
        oracle_overrides=overrides,
        return_monthly_trace=True,
    )
    result = {
        "episodes": int(len(df)),
        "survival_rate": float((df["cause"] == "Time Limit").mean()),
        "avg_duration": float(df["steps"].mean()),
        "avg_rule_40": float(df["avg_rule_40"].mean()),
        "avg_final_mrr": float(df["final_mrr"].mean()),
        "avg_llm_calls": float(df["llm_calls"].mean()),
        "avg_proposal_llm_calls": float(df.get("proposal_llm_calls", 0).mean()) if "proposal_llm_calls" in df else 0.0,
        "avg_proposal_cache_hits": float(df.get("proposal_cache_hits", 0).mean()) if "proposal_cache_hits" in df else 0.0,
        "avg_proposal_fallbacks": float(df.get("proposal_fallbacks", 0).mean()) if "proposal_fallbacks" in df else 0.0,
        "causes": df["cause"].tolist(),
        "per_episode": df[["seed", "cause", "steps", "final_mrr", "avg_rule_40", "llm_calls", "proposal_llm_calls", "proposal_cache_hits", "proposal_fallbacks"]].to_dict(orient="records"),
    }
    causal_values = []
    proposal_sources = {}
    stress_nodes = {}
    for row in monthly:
        trace = row.get("decision_trace") or {}
        src = trace.get("proposal_source")
        if src:
            proposal_sources[src] = proposal_sources.get(src, 0) + 1
        stress = trace.get("causal_stress_node")
        if stress:
            stress_nodes[stress] = stress_nodes.get(stress, 0) + 1
        for proposal in trace.get("proposals") or []:
            value = proposal.get("causal_confidence")
            if value is not None:
                causal_values.append(float(value))
    result["avg_causal_confidence"] = (sum(causal_values) / len(causal_values)) if causal_values else None
    result["causal_confidence_count"] = len(causal_values)
    result["proposal_sources"] = proposal_sources
    result["stress_nodes"] = stress_nodes
    results[policy] = result
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"RUN_DONE policy={policy}", flush=True)

print("COMPARISON_JSON_START", flush=True)
print(json.dumps(results, indent=2), flush=True)
print("COMPARISON_JSON_END", flush=True)
