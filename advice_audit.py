"""Audit founder advice quality across a matrix of company profiles.

Runs the real advise path (no mocks) and checks each plan against rules the
engine should never violate, whatever the LLM says. Every violation here is a
concrete defect a founder would feel.

    venv\\Scripts\\python.exe advice_audit.py            # full matrix
    venv\\Scripts\\python.exe advice_audit.py --repeat 3 # variance on one profile
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from backend.advise_service import run_analysis

SALARY_SLOT = 8000.0


def profile(name, mrr, cash, costs, price, churn, cac=None, age=12, quality=0.5, competitors=5):
    """A founder profile in the client's payload shape."""
    headcount = max(1, int(round(costs / SALARY_SLOT)))
    return {
        "name": name,
        "payload": {
            "company_id": f"audit-{name}",
            "company_age_months": age,
            "month_index": 0,
            "config": {
                "company_name": name,
                "initial_mrr": mrr,
                "initial_cash": cash,
                "average_price": price,
                "cac": cac if cac is not None else 50,
                "churn_enterprise": churn,
                "churn_smb": churn,
                "churn_b2c": churn,
                "competitors": competitors,
                "product_quality": quality,
                "initial_headcount": headcount,
            },
            "history": [],
        },
        "mrr": mrr,
        "cash": cash,
        "costs": costs,
        "headcount": headcount,
    }


MATRIX = [
    profile("pre_revenue",    1_000,   60_000,  16_000, 25, 0.08, cac=120, age=4,  quality=0.2),
    profile("small_struggling", 12_000, 90_000,  24_000, 40, 0.05, cac=91,  age=12),
    profile("small_healthy",  12_000, 400_000, 20_000, 40, 0.02, cac=60,  age=18, quality=0.7),
    profile("cash_crisis",     30_000,  45_000,  70_000, 80, 0.06, cac=200, age=24),
    profile("calibration",     50_000, 500_000,  60_000, 80, 0.03, cac=90,  age=24),
    profile("scaling",        200_000, 2_000_000, 220_000, 150, 0.02, cac=140, age=36, quality=0.8),
]


def runway_months(cash: float, costs: float, mrr: float) -> float:
    net_burn = costs - mrr
    if net_burn <= 0:
        return float("inf")
    return cash / net_burn


def check(case: dict, result: dict) -> list[str]:
    """Rules the plan must not break. Returns a list of violations."""
    action = result["trace"].get("final_action") or {}
    brief = result.get("brief") or {}
    mrr = case["mrr"]

    marketing = float((action.get("marketing") or {}).get("spend", 0) or 0)
    rd = float((action.get("product") or {}).get("r_and_d_spend", 0) or 0)
    hires = int((action.get("hiring") or {}).get("hires", 0) or 0)
    spend = marketing + rd

    real_runway = runway_months(case["cash"], case["costs"], mrr)
    post_plan_runway = runway_months(case["cash"], case["costs"] + spend, mrr)

    risk = str(brief.get("risk_level", ""))
    violations = []

    if mrr > 0 and spend > mrr:
        violations.append(
            f"discretionary spend ${spend:,.0f} exceeds MRR ${mrr:,.0f} ({spend/mrr:.0%})"
        )
    # Judge hiring on runway that counts revenue. The engine's own
    # _estimate_runway_months ignores MRR and reads 8.9 months for a company
    # with 100, which would flag healthy companies as violations.
    if hires > 0 and real_runway < 24:
        violations.append(
            f"recommends {hires} hire(s) at {real_runway:.1f}mo runway (guard is 24mo)"
        )
    if risk in ("LOW",) and real_runway < 6:
        violations.append(f"risk={risk} but runway is {real_runway:.1f}mo")
    if post_plan_runway < 3 and real_runway >= 3:
        violations.append(
            f"plan cuts runway {real_runway:.1f}mo -> {post_plan_runway:.1f}mo"
        )
    if not result.get("llm_ok"):
        violations.append("llm_ok=false (fallback brief, not real analysis)")

    # The plan contradicting the evidence rendered beside it.
    graph = result["trace"].get("graph_summary") or {}
    effects = set(graph.get("observed") or []) | set(graph.get("expected") or [])
    if hires > 0 and "Hiring_Freeze_Recommended" in effects:
        violations.append("recommends hiring while evidence says hiring was frozen")
    if marketing > mrr * 0.5 and "Marketing_Spend_Cut" in effects:
        violations.append("raises marketing while evidence says marketing was cut")

    return violations


def summarize(case, result, violations) -> dict:
    action = result["trace"].get("final_action") or {}
    mrr = case["mrr"]
    marketing = float((action.get("marketing") or {}).get("spend", 0) or 0)
    rd = float((action.get("product") or {}).get("r_and_d_spend", 0) or 0)
    return {
        "profile": case["name"],
        "mrr": mrr,
        "runway": round(runway_months(case["cash"], case["costs"], mrr), 1),
        "risk": (result.get("brief") or {}).get("risk_level"),
        "confidence": (result.get("brief") or {}).get("confidence"),
        "stress_node": result["trace"].get("causal_stress_node"),
        "marketing": round(marketing),
        "marketing_pct": round(marketing / mrr * 100) if mrr else None,
        "rd": round(rd),
        "rd_pct": round(rd / mrr * 100) if mrr else None,
        "hires": int((action.get("hiring") or {}).get("hires", 0) or 0),
        "scale": round(result["trace"].get("absolute_scale", 1.0), 3),
        "llm_ok": result.get("llm_ok"),
        "violations": violations,
    }


def run_matrix(cases) -> list[dict]:
    rows = []
    for case in cases:
        started = time.time()
        try:
            result = run_analysis(case["payload"])
        except Exception as exc:  # a crash is itself an audit finding
            rows.append({"profile": case["name"], "violations": [f"CRASHED: {exc}"]})
            continue
        violations = check(case, result)
        row = summarize(case, result, violations)
        row["seconds"] = round(time.time() - started, 1)
        rows.append(row)
        print(f"  {case['name']:<18} {row['seconds']:>5.1f}s  "
              f"{len(violations)} violation(s)")
    return rows


def report(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"{'profile':<18}{'MRR':>9}{'runway':>8}{'risk':>10}{'mktg%':>7}{'R&D%':>7}{'hires':>7}")
    print("-" * 78)
    for r in rows:
        if "mrr" not in r:
            print(f"{r['profile']:<18}  CRASHED")
            continue
        print(f"{r['profile']:<18}{r['mrr']:>9,}{r['runway']:>8}{str(r['risk']):>10}"
              f"{str(r['marketing_pct']):>7}{str(r['rd_pct']):>7}{r['hires']:>7}")

    total = sum(len(r.get("violations", [])) for r in rows)
    print("-" * 78)
    print(f"{total} violation(s) across {len(rows)} profiles\n")
    for r in rows:
        if r.get("violations"):
            print(f"  {r['profile']}  (stress: {r.get('stress_node')})")
            for v in r["violations"]:
                print(f"     - {v}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=0,
                        help="run one profile N times to measure variance")
    parser.add_argument("--profile", default="small_struggling")
    parser.add_argument("--out", default="audit_results.json")
    args = parser.parse_args()

    if args.repeat:
        case = next(c for c in MATRIX if c["name"] == args.profile)
        cases = [dict(case, name=f"{case['name']}_{i+1}") for i in range(args.repeat)]
    else:
        cases = MATRIX

    print(f"running {len(cases)} analysis/analyses (about 35s each)\n")
    rows = run_matrix(cases)
    report(rows)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
