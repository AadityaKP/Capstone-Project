"""Capture the real stack's answers for the demo dataset, once.

Run this with Ollama and Neo4j up. It calls the live advise path and writes what
came back into backend/demo_fixtures/, so the same inputs can be replayed later
on a machine running neither.

    venv\\Scripts\\python.exe record_demo_fixtures.py

Re-run it to refresh the recordings; the strategist is sampled, so the wording
will differ between captures even though the inputs do not.
"""

from __future__ import annotations

import sys

from backend import demo_fixtures
from backend.advise_service import run_analysis


def config(mrr, cash, costs, price, churn_pct, marketing, new_customers, team):
    """The shape frontend/src/api.js buildAdvisePayload sends."""
    return {
        "company_name": "Kettle Analytics",
        "initial_mrr": mrr,
        "initial_cash": cash,
        "average_price": price,
        "cac": round(marketing / new_customers, 2),
        "churn_enterprise": churn_pct / 100,
        "churn_smb": churn_pct / 100,
        "churn_b2c": churn_pct / 100,
        "competitors": 5,            # "A few rivals"
        "product_quality": 0.5,      # "Solid"
        "monthly_costs": costs,
        "initial_headcount": team,
    }


# The demo dataset, documented in docs/demo_walkthrough.md.
MONTH_1 = {
    "company_id": "demo-kettle",
    "company_age_months": 15,
    "month_index": 0,
    "config": config(11_000, 95_000, 15_500, 55, 3.4, 2_600, 20, 2),
    "history": [],
}

MONTH_2 = {
    "company_id": "demo-kettle",
    "company_age_months": 16,
    "month_index": 1,
    "config": config(12_800, 92_000, 16_400, 56, 2.9, 3_000, 24, 2),
    "history": [{"mrr": 11_000, "churn": 0.034, "entered_at": "2026-08-01T00:00:00Z"}],
}

RECORDINGS = (("kettle_month_1", MONTH_1), ("kettle_month_2", MONTH_2))


if __name__ == "__main__":
    for name, payload in RECORDINGS:
        print(f"recording {name} ...", flush=True)
        result = run_analysis(payload)
        if not result.get("llm_ok"):
            print(f"  REFUSING to record {name}: llm_ok is False, so this capture is the\n"
                  f"  built-in fallback rather than the strategist. Start Ollama and retry.")
            sys.exit(1)
        path = demo_fixtures.save(name, payload, result)
        action = (result.get("trace") or {}).get("final_action", {})
        print(f"  -> {path}")
        print(f"     risk {result['brief'].get('risk_level')}   "
              f"marketing ${action.get('marketing', {}).get('spend', 0):,.0f}   "
              f"R&D ${action.get('product', {}).get('r_and_d_spend', 0):,.0f}   "
              f"hires {action.get('hiring', {}).get('hires', 0)}")
    print("\ndone. These replay with no Ollama and no Neo4j.")
