import builtins
import gc
import json
import os
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

from simulation_runner import run_simulation

OUT_DIR = Path("outputs/oracle_v4_compare_final_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "comparison.json"
EXTRACT = OUT_DIR / "monthly_extract.json"

V3_SOURCE = Path("outputs/oracle_v4_compare_final/comparison.json")

# --------------------------------------------------------------------------- #
# Stall watcher                                                                #
# --------------------------------------------------------------------------- #
_original_print = builtins.print
_last_episode_end = {"t": time.time(), "count": 0}
_stop_watcher = threading.Event()

_HEARTBEAT_S = 30
_STALL_WARN_S = 90   # warn if no EPISODE_END in this many seconds


def _intercepting_print(*args, **kwargs):
    msg = " ".join(str(a) for a in args)
    if "EPISODE_END" in msg:
        _last_episode_end["t"] = time.time()
        _last_episode_end["count"] += 1
    _original_print(*args, **kwargs)


builtins.print = _intercepting_print


def _watcher_loop():
    while not _stop_watcher.wait(_HEARTBEAT_S):
        elapsed = time.time() - _last_episode_end["t"]
        done = _last_episode_end["count"]
        ts = time.strftime("%H:%M:%S")
        if elapsed >= _STALL_WARN_S:
            _original_print(
                f"[STALL]     {ts} | {elapsed:.0f}s since last EPISODE_END "
                f"(episodes done so far: {done}) — possible hang",
                flush=True,
            )
        else:
            _original_print(
                f"[HEARTBEAT] {ts} | {elapsed:.0f}s since last EPISODE_END "
                f"| episodes done: {done}",
                flush=True,
            )


_watcher = threading.Thread(target=_watcher_loop, daemon=True)
_watcher.start()

# --------------------------------------------------------------------------- #
# Results bootstrap                                                            #
# --------------------------------------------------------------------------- #
results = {}
if OUT.exists():
    try:
        results = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"RESUME loaded existing results for: {list(results.keys())}", flush=True)
    except Exception:
        results = {}

# Seed oracle_v3 from the previous final run so we skip re-running it
if "oracle_v3" not in results and V3_SOURCE.exists():
    try:
        v3_data = json.loads(V3_SOURCE.read_text(encoding="utf-8"))
        if "oracle_v3" in v3_data:
            results["oracle_v3"] = v3_data["oracle_v3"]
            OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(
                f"SEEDED oracle_v3 from {V3_SOURCE} "
                f"(episodes={results['oracle_v3'].get('episodes')}, "
                f"survival={results['oracle_v3'].get('survival_rate'):.0%})",
                flush=True,
            )
    except Exception as e:
        print(f"WARN could not seed v3: {e}", flush=True)

# --------------------------------------------------------------------------- #
# Run loop                                                                     #
# --------------------------------------------------------------------------- #
configs = [
    ("oracle_v4_causal_hetero", {}),
]

for policy, overrides in configs:
    if policy in results:
        print(f"SKIP policy={policy} (already in comparison.json)", flush=True)
        continue

    print(f"\nRUN_START policy={policy} at {time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()

    df, monthly = run_simulation(
        policy=policy,
        num_episodes=20,
        seed_start=0,
        oracle_frequency=5,
        oracle_overrides=overrides,
        return_monthly_trace=True,
    )

    elapsed_run = time.time() - t0
    print(f"RUN_DONE policy={policy} | wall_time={elapsed_run:.0f}s", flush=True)

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
        "avg_proposal_llm_calls": float(df["proposal_llm_calls"].mean()) if "proposal_llm_calls" in df.columns else 0.0,
        "avg_proposal_cache_hits": float(df["proposal_cache_hits"].mean()) if "proposal_cache_hits" in df.columns else 0.0,
        "avg_proposal_fallbacks": float(df["proposal_fallbacks"].mean()) if "proposal_fallbacks" in df.columns else 0.0,
        "causes": df["cause"].tolist(),
        "per_episode": df[per_episode_cols].to_dict(orient="records"),
        "wall_time_s": elapsed_run,
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
        agent: sum(vals) / len(vals)
        for agent, vals in causal_by_agent.items()
        if vals
    }
    result["proposal_sources"] = proposal_sources
    result["stress_nodes"] = stress_nodes

    results[policy] = result
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if policy == "oracle_v4_causal_hetero":
        EXTRACT.write_text(json.dumps(extract_rows, indent=2), encoding="utf-8")

    del df, monthly, extract_rows, causal_values, causal_by_agent
    gc.collect()

_stop_watcher.set()

print("\nCOMPARISON_JSON_START", flush=True)
print(json.dumps(results, indent=2), flush=True)
print("COMPARISON_JSON_END", flush=True)
