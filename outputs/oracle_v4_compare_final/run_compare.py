import gc
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from simulation_runner import run_simulation


OUT_DIR = Path("outputs/oracle_v4_compare_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "comparison.json"
EXTRACT = OUT_DIR / "monthly_extract.json"

configs = [
    ("oracle_v3", {}),
    ("oracle_v4_causal_hetero", {}),
]

# Load any previously completed results so the script is safely resumable.
results = {}
if OUT.exists():
    try:
        results = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"RESUME loaded existing results for: {list(results.keys())}", flush=True)
    except Exception:
        results = {}

for policy, overrides in configs:
    if policy in results:
        print(f"SKIP policy={policy} (already in comparison.json)", flush=True)
        continue

    print(f"RUN_START policy={policy}", flush=True)
    df, monthly = run_simulation(
        policy=policy,
        num_episodes=20,
        seed_start=0,
        oracle_frequency=5,
        oracle_overrides=overrides,
        return_monthly_trace=True,
    )
    per_episode_cols = [
        c for c in [
            "seed", "cause", "steps", "final_mrr", "final_cash",
            "avg_rule_40", "llm_calls",
            "proposal_llm_calls", "proposal_cache_hits", "proposal_fallbacks",
        ]
        if c in df.columns
    ]
    result = {
        "episodes": int(len(df)),
        "survival_rate": float((df["cause"] == "Time Limit").mean()),
        "avg_duration": float(df["steps"].mean()),
        "avg_rule_40": float(df["avg_rule_40"].mean()),
        "avg_final_mrr": float(df["final_mrr"].mean()),
        "avg_final_cash": float(df["final_cash"].mean()),
        "avg_llm_calls": float(df["llm_calls"].mean()),
        "avg_proposal_llm_calls": float(df["proposal_llm_calls"].mean()) if "proposal_llm_calls" in df else 0.0,
        "avg_proposal_cache_hits": float(df["proposal_cache_hits"].mean()) if "proposal_cache_hits" in df else 0.0,
        "avg_proposal_fallbacks": float(df["proposal_fallbacks"].mean()) if "proposal_fallbacks" in df else 0.0,
        "causes": df["cause"].tolist(),
        "per_episode": df[per_episode_cols].to_dict(orient="records"),
    }

    causal_values = []
    causal_by_agent = {}
    proposal_sources = {}
    stress_nodes = {}
    extract_rows = []
    for row in monthly:
        trace = row.get("decision_trace") or {}
        source = trace.get("proposal_source")
        if source:
            proposal_sources[source] = proposal_sources.get(source, 0) + 1
        stress = trace.get("causal_stress_node")
        if stress:
            stress_nodes[stress] = stress_nodes.get(stress, 0) + 1
        for proposal in trace.get("proposals") or []:
            value = proposal.get("causal_confidence")
            agent_name = proposal.get("agent", "UNKNOWN")
            if value is not None:
                numeric = float(value)
                causal_values.append(numeric)
                causal_by_agent.setdefault(agent_name, []).append(numeric)

        if policy == "oracle_v4_causal_hetero":
            proposals_by_agent = {p.get("agent"): p for p in (trace.get("proposals") or [])}
            cfo = proposals_by_agent.get("CFO", {})
            cmo = proposals_by_agent.get("CMO", {})
            cpo = proposals_by_agent.get("CPO", {})
            extract_rows.append({
                "episode": row.get("episode"),
                "month": row.get("month"),
                "causal_stress_node": trace.get("causal_stress_node"),
                "stress_persistence_months": trace.get("stress_persistence_months"),
                "cfo_price_change_pct": (cfo.get("actions") or {}).get("pricing", {}).get("price_change_pct"),
                "cmo_marketing_spend": (cmo.get("actions") or {}).get("marketing", {}).get("spend"),
                "cpo_rd_spend": (cpo.get("actions") or {}).get("product", {}).get("r_and_d_spend"),
            })

    result["avg_causal_confidence"] = (sum(causal_values) / len(causal_values)) if causal_values else None
    result["causal_confidence_count"] = len(causal_values)
    result["avg_causal_confidence_by_agent"] = {
        agent_name: sum(values) / len(values)
        for agent_name, values in causal_by_agent.items()
        if values
    }
    result["proposal_sources"] = proposal_sources
    result["stress_nodes"] = stress_nodes
    results[policy] = result
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if policy == "oracle_v4_causal_hetero":
        EXTRACT.write_text(json.dumps(extract_rows, indent=2), encoding="utf-8")
    print(f"RUN_DONE policy={policy}", flush=True)

    # Free large per-episode data before the next policy runs.
    del df, monthly, extract_rows, causal_values, causal_by_agent
    gc.collect()

print("COMPARISON_JSON_START", flush=True)
print(json.dumps(results, indent=2), flush=True)
print("COMPARISON_JSON_END", flush=True)
