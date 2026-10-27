"""Loop-is-live check (docs/oefa_loop_plan.md section 7).

Both of the loop's failure modes are silent: write_causal_outcome returns
without doing anything when no graph store is enabled, and memory retrieval
returns nothing when the scope is a per-request UUID. This script makes each
one loud. Run it before a demo (section 8.3 step 2):

    venv\\Scripts\\python.exe experiments\\loop_live_check.py
    venv\\Scripts\\python.exe experiments\\loop_live_check.py --profile founder --real-store

Checks
  1. capabilities   graph store, memory store, LLM, per the same probes
                    /api/health uses
  2. memory         two analyses for the same throwaway company; the second
                    must retrieve a memory the first matured. Uses an
                    isolated temporary Chroma path unless --real-store.
  3. causal write   under oracle_v4_causal with Neo4j up, a write with
                    source="sim" must report True

Exit status is non-zero when the graph or memory check fails. LLM
reachability is reported, not required: the board falls back to rules and
says so, which is a legitimate state to demo (runbook 4.2).
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def payload(company_id: str, month: int, mrr: float) -> dict:
    # Nine months of history so the oldest entries clear the 6-month
    # maturation horizon within a single analysis and get written.
    history = [{"mrr": mrr * (0.85 + 0.02 * i), "churn": 0.05} for i in range(9)]
    return {
        "company_id": company_id,
        "company_age_months": month,
        "month_index": 0,
        "config": {
            "company_name": "Loop check", "initial_mrr": mrr, "initial_cash": 200_000,
            "average_price": 80, "cac": 90, "churn_enterprise": 0.05, "churn_smb": 0.05,
            "churn_b2c": 0.05, "competitors": 5, "product_quality": 0.5,
            "monthly_costs": 40_000, "initial_headcount": 3,
        },
        "history": history,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", choices=["founder", "review2"], default=os.getenv("SIM_PROFILE", "founder"))
    parser.add_argument("--real-store", action="store_true",
                        help="use the configured Chroma path instead of a temporary one (writes a throwaway company scope into it)")
    parser.add_argument("--database", default=None, help="SQLite path for the check (default: temporary)")
    args = parser.parse_args()

    os.environ["SIM_PROFILE"] = args.profile
    tmp = tempfile.mkdtemp(prefix="loop_check_")
    if not args.real_store:
        os.environ["FOUNDER_CHROMA_PATH"] = os.path.join(tmp, "chroma_founder")
        os.environ["CHROMA_PATH"] = os.path.join(tmp, "chroma")
    os.environ["DATABASE_PATH"] = args.database or os.path.join(tmp, "check.db")

    from backend import loop_status, sim_profile
    from backend.database import initialize_database
    from backend.advise_service import run_analysis

    initialize_database()
    failures: list[str] = []

    print("== 1. capabilities ==")
    status = loop_status.loop_status(force=True)
    for key in ("advisor_mode", "sim_profile", "graph_store_enabled", "graph_store_reason",
                "memory_store_enabled", "memory_scope", "llm_reachable", "llm_reason", "loop_live"):
        print(f"  {key:22} {status[key]}")
    if not status["graph_store_enabled"]:
        failures.append(f"graph store: {status['graph_store_reason']}")
    if not status["memory_store_enabled"]:
        failures.append(f"memory store: {status['memory_reason']}")

    print("== 2. memory: a second analysis reads what the first matured ==")
    company_id = f"loopcheck-{uuid.uuid4().hex[:8]}"
    first = run_analysis(payload(company_id, month=20, mrr=30_000))
    second = run_analysis(payload(company_id, month=21, mrr=31_000))
    scope = second["trace"].get("memory_scope")
    count = second["trace"].get("memory_count", 0)
    print(f"  scope                  {scope}")
    print(f"  first  memory_count    {first['trace'].get('memory_count', 0)}  pending={first['trace']['oracle_state']['pending_memories']}")
    print(f"  second memory_count    {count}  pending={second['trace']['oracle_state']['pending_memories']}")
    print(f"  llm_ok                 {first['llm_ok']} / {second['llm_ok']}")
    if not scope or not str(scope).startswith("company:"):
        failures.append(f"memory scope is not a company key: {scope!r}")
    if count <= 0:
        failures.append("second analysis retrieved no memories from the first (memory is write-only or the store cannot embed)")

    print("== 3. causal write lands ==")
    from backend.advise_service import build_env_state, build_oracle
    oracle, _ = build_oracle(build_env_state(payload(company_id, month=21, mrr=31_000)), company_id)
    print(f"  oracle.mode            {oracle.mode}")
    print(f"  graph_store_enabled    {oracle.graph_store_enabled}")
    if oracle.graph_store_enabled:
        landed = oracle.write_causal_outcome(
            {"marketing": {"spend": 1.0, "channel": "ppc"}, "product": {"r_and_d_spend": 1.0},
             "hiring": {"hires": 0, "cost_per_employee": 10000}, "pricing": {"price_change_pct": 0.0}},
            {"mrr_pct": 0.0}, stress_node="Steady_State", source="sim", weight=0.0,
        )
        print(f"  write attempted        {landed}  (source=sim, weight=0 - no confidence change)")
        if not landed:
            failures.append("causal write did not land despite an enabled graph store")
    else:
        print("  skipped: no graph store (see check 1)")

    print()
    if failures:
        print("LOOP NOT LIVE:")
        for item in failures:
            print(f"  - {item}")
        print("Either fix the service or change what you claim on stage (plan section 8.3).")
        return 1
    print("LOOP LIVE: graph writes land, memory is per-company and readable"
          + ("" if status["llm_reachable"] else ", LLM unreachable (rules-only briefs)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
