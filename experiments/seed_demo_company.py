"""Seed the demo company (docs/oefa_loop_plan.md section 8.3 step 4).

The demo must open on a company that already has history: a first cycle has
no memory to retrieve, no prediction error to report and nothing learned. This
script creates a company with six months of real-looking history and TWO
completed cycles with the HITL close already done on the first - including
one action marked "didn't", so the "we're not counting this month as evidence
about it" line has something to show.

It runs the real services (cycle_service.run_cycle_sync, submit_feedback), so
the server side - cycles, cycle_months, cycle_feedback, the company's Oracle
calendar, causal writes when the graph is up - is genuinely populated. The
browser is the founder's record (state-ownership decision (a)), so it also
writes data/demo_bootstrap.json in the browser store's shape; Settings offers
"Load the seeded demo company" (GET /api/demo/bootstrap) to import it.

    venv\\Scripts\\python.exe experiments\\seed_demo_company.py
    venv\\Scripts\\python.exe experiments\\seed_demo_company.py --no-oracle   # rules only, no services needed

Run it with the stack's SIM_PROFILE and database so the browser and the
server agree on which cycles exist.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

OUT = os.path.join(ROOT, "data", "demo_bootstrap.json")

# Six months of history plus the current month: a company that grew through a
# rough patch, so trends, memories and a prediction error all have content.
#
# Cost line and cash balance are chosen so the company survives its own
# board plan under the fitted physics (decision 9). The board's plan for a
# company this size is roughly $17k of monthly spend plus one hire, which
# at an 83.5% gross margin burns about $27k a month on top of the cost
# line; at ~$33k MRR, $30k of costs with $350k in the bank keeps that plan
# alive in every run with ~$100k left after twelve months, where the
# earlier $47k on $200k died in every run under either curve. A demo
# company that is structurally dying teaches nothing about the loop.
HISTORY = [
    {"mrr": 24_600, "cash": 425_000, "costs": 28_500, "price": 85, "churnMonthly": 6.1, "newCustomers": 31, "marketingSpend": 7_000},
    {"mrr": 25_900, "cash": 412_000, "costs": 28_500, "price": 85, "churnMonthly": 5.9, "newCustomers": 34, "marketingSpend": 7_000},
    {"mrr": 27_400, "cash": 399_000, "costs": 29_000, "price": 85, "churnMonthly": 5.7, "newCustomers": 36, "marketingSpend": 6_500},
    {"mrr": 28_800, "cash": 387_000, "costs": 29_000, "price": 85, "churnMonthly": 5.6, "newCustomers": 38, "marketingSpend": 6_000},
    {"mrr": 30_000, "cash": 375_000, "costs": 30_000, "price": 85, "churnMonthly": 5.2, "newCustomers": 41, "marketingSpend": 6_000},
    {"mrr": 31_900, "cash": 362_000, "costs": 29_500, "price": 85, "churnMonthly": 4.6, "newCustomers": 44, "marketingSpend": 4_000},
]
CURRENT = {"mrr": 33_600, "cash": 350_000, "costs": 30_000, "price": 85, "churnMonthly": 4.4, "newCustomers": 46, "marketingSpend": 5_000}


def iso(months_ago: int) -> str:
    d = datetime.now(timezone.utc).replace(day=1, hour=9, minute=30, second=0, microsecond=0)
    year, month = d.year, d.month - months_ago
    while month <= 0:
        month += 12
        year -= 1
    return d.replace(year=year, month=month).isoformat()


def payload(company: dict, month: dict, history: list[dict], horizon: int, use_oracle: bool,
            previous_track_record: dict | None = None) -> dict:
    v = month["values"]
    churn = v["churnMonthly"] / 100.0
    return {
        "company_id": company["id"],
        "company_age_months": company["ageMonths"] + month["index"],
        "month_index": month["index"],
        "config": {
            "company_name": company["name"], "initial_mrr": v["mrr"], "initial_cash": v["cash"],
            "average_price": v["price"], "cac": v["marketingSpend"] / v["newCustomers"],
            "churn_enterprise": churn, "churn_smb": churn, "churn_b2c": churn,
            "competitors": 9, "product_quality": 0.5, "monthly_costs": v["costs"],
            "initial_headcount": company["headcountReal"],
        },
        "history": [{"mrr": h["values"]["mrr"], "churn": h["values"]["churnMonthly"] / 100.0} for h in history],
        "horizon_months": horizon, "use_oracle": use_oracle, "seed": 0,
        "previous_track_record": previous_track_record,
    }


def client_cycle(server: dict, month_id: str, started_from: bool) -> dict:
    return {
        "id": server["id"], "monthId": month_id, "createdAt": server["created_at"], "source": "api",
        "status": server["status"], "horizon": server["horizon_months"],
        "months": server["months"], "summary": server["summary"], "meta": server["meta"],
        "feedback": [
            {"monthIndex": f["month_index"], "submitted": f["submitted"], "result": f["result"],
             "error": None, "closedAt": f["created_at"]}
            for f in server["feedback"]
        ],
        "startedFromTrackRecord": started_from,
    }


def client_analysis(cycle: dict, month_id: str) -> dict:
    m = cycle["months"][0]
    return {
        "id": f"a_{uuid.uuid4().hex[:8]}", "cycleId": cycle["id"], "monthIndex": 1, "monthId": month_id,
        "createdAt": cycle["created_at"], "source": "cycle", "llm_ok": m["execute"]["llm_ok"],
        "reason": m["execute"]["refresh_reason"], "brief": m["execute"]["brief"],
        "trace": m["execute"]["trace"], "display": m["execute"]["display"], "narratives": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-oracle", action="store_true", help="rules-only cycles; no Ollama/Neo4j/Chroma needed")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--name", default="Acme Analytics")
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()

    from backend import cycle_service
    from backend.database import initialize_database

    initialize_database()
    use_oracle = not args.no_oracle

    company = {
        "id": f"demo-{uuid.uuid4().hex[:8]}", "name": args.name,
        "whatYouSell": "Usage analytics for e-commerce teams", "ageMonths": 8,
        "crowdedness": "crowded", "maturity": "solid", "headcountReal": 4, "createdAt": iso(6),
    }
    months = []
    for i, values in enumerate(HISTORY + [CURRENT]):
        months.append({"id": f"m{i + 1}", "index": i, "enteredAt": iso(len(HISTORY) - i), "values": values, "decisions": []})
    previous, current = months[-2], months[-1]

    print(f"seeding {company['name']} ({company['id']}), oracle={'on' if use_oracle else 'off'}")

    # Cycle 1: planned on last month's numbers, then closed with this month's.
    print("cycle 1: planning on last month ...")
    c1 = cycle_service.create_cycle(payload(company, previous, months[:-2], args.horizon, use_oracle))
    c1 = cycle_service.run_cycle_sync(c1["id"])
    if c1["status"] != "completed":
        print(f"cycle 1 failed: {c1.get('error')}")
        return 1
    print(f"  {c1['summary']['months_completed']} months, {c1['summary']['total_latency_s']:.1f}s, "
          f"llm_ok months={c1['summary']['llm_ok_months']}, graph={c1['summary']['graph_store_enabled']}")

    action = c1["months"][0]["execute"]["action"]
    per_action = [{"action_key": "marketing", "done": "did", "note": None}]
    if action["product"]["r_and_d_spend"] > 0:
        per_action.append({"action_key": "product", "done": "partly", "note": "Shipped the onboarding fix, not the reporting work"})
    # One action the founder did NOT do, whatever the board said about it.
    skip = "pricing" if action["pricing"]["price_change_pct"] > 0.001 else ("hiring" if action["hiring"]["hires"] > 0 else "product")
    per_action = [p for p in per_action if p["action_key"] != skip]
    per_action.append({"action_key": skip, "done": "didnt", "note": "Didn't get to it this month"})
    cv = current["values"]
    print("cycle 1: closing with this month's real numbers ...")
    result = cycle_service.submit_feedback(c1["id"], {
        "month_index": 1, "per_action": per_action,
        "actuals": {"mrr": cv["mrr"], "cash": cv["cash"], "churn": cv["churnMonthly"], "costs": cv["costs"]},
    })
    err = result["prediction_error"]["mrr_pct"]
    print(f"  revenue predicted {err['expected']:+.1f}%, actual {err['realized']:+.1f}%; "
          f"evidence written={result['evidence']['written']} ({result['evidence']['reason']})")
    c1 = cycle_service.get_cycle(c1["id"])

    for p in per_action:
        previous["decisions"].append({
            "id": f"d_{uuid.uuid4().hex[:6]}", "domain": p["action_key"], "text": f"{p['action_key']} per the board's plan",
            "state": {"did": "accepted", "partly": "custom", "didnt": "declined"}[p["done"]], "note": p["note"],
        })

    # Cycle 2: planned on this month's numbers, starting from the close.
    print("cycle 2: planning on this month from the close ...")
    c2 = cycle_service.create_cycle(payload(company, current, months[:-1], args.horizon, use_oracle,
                                            previous_track_record=result["track_record"]))
    c2 = cycle_service.run_cycle_sync(c2["id"])
    if c2["status"] != "completed":
        print(f"cycle 2 failed: {c2.get('error')}")
        return 1
    print(f"  {c2['summary']['months_completed']} months, {c2['summary']['total_latency_s']:.1f}s")

    cycles = [client_cycle(c1, previous["id"], False), client_cycle(c2, current["id"], True)]
    bootstrap = {
        "company": company, "months": months,
        "analyses": [client_analysis(c1, previous["id"]), client_analysis(c2, current["id"])],
        "cycles": cycles, "settings": {"narratives": False}, "onboardingDraft": None,
        "seeded_at": datetime.now(timezone.utc).isoformat(),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(bootstrap, handle)
    print(f"written {args.out}; load it from Settings -> 'Load the seeded demo company'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
