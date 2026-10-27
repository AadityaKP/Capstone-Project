"""Prediction-error harness (docs/oefa_loop_plan.md section 7).

Replays a company's recorded months as if each had been a HITL close and
reports how well the board's expected_delta was calibrated against what
actually happened. This number exists only because of Phase 1A: before
expected_delta there was nothing to score.

Input is the founder's own record - the browser store exported as JSON:

    JSON.parse(localStorage.getItem("ssom_founder_v1"))   // in the browser console

saved to a file, or the seeded demo workspace (data/demo_bootstrap.json from
experiments/seed_demo_company.py).

    venv\\Scripts\\python.exe experiments\\prediction_error_harness.py data\\demo_bootstrap.json
    venv\\Scripts\\python.exe experiments\\prediction_error_harness.py my_export.json --use-oracle

For every month t with month t+H on record, the board decides at t (rules
path by default; --use-oracle runs the configured Oracle and needs the
services), the expected_delta for its final action is taken at horizon H,
and the realized delta is the founder's own numbers at t+H. Reported per
KPI: mean error, mean absolute error, direction agreement and the share
within tolerance - the same tolerance the product uses.

Honesty notes printed with the numbers: the board did not actually run on
these months, so the founder never took its advice; this measures the
simulator's calibration against the company's real trajectory under whatever
the founder actually did. That is exactly the question the harness exists to
answer, and exactly why it is not a test of advice quality.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def founder_payload(company: dict, month: dict, history: list[dict]) -> dict:
    """The same shape frontend/src/api.js buildAdvisePayload produces."""
    v = month["values"]
    cac = v.get("cacDirect")
    if not cac and v.get("marketingSpend") and v.get("newCustomers"):
        cac = v["marketingSpend"] / v["newCustomers"]
    churn = (v.get("churnMonthly") or 0) / 100.0
    competitors = {"few": 2, "some": 5, "crowded": 9}.get(company.get("crowdedness"), 5)
    quality = {"early": 0.2, "solid": 0.5, "polished": 0.8}.get(company.get("maturity"), 0.5)
    return {
        "company_id": company.get("id", "harness"),
        "company_age_months": (company.get("ageMonths") or 0) + (month.get("index") or 0),
        "month_index": month.get("index") or 0,
        "config": {
            "company_name": company.get("name", "Harness"),
            "initial_mrr": v["mrr"], "initial_cash": v["cash"], "average_price": v.get("price") or 50,
            "cac": cac or 50, "churn_enterprise": churn, "churn_smb": churn, "churn_b2c": churn,
            "competitors": competitors, "product_quality": quality,
            "monthly_costs": v.get("costs"), "initial_headcount": max(1, int(company.get("headcountReal") or 1)),
        },
        "history": [{"mrr": h["values"]["mrr"], "churn": (h["values"].get("churnMonthly") or 0) / 100.0} for h in history],
        "horizon_months": 1,
        "use_oracle": False,
        "seed": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("export", help="browser store export or data/demo_bootstrap.json")
    parser.add_argument("--horizon", type=int, default=2, help="expected_delta horizon in months (default 2)")
    parser.add_argument("--use-oracle", action="store_true", help="decide with the configured Oracle (needs services)")
    parser.add_argument("--out", default=os.path.join(ROOT, "outputs", "prediction_error_harness.json"))
    args = parser.parse_args()

    os.environ.setdefault("SIM_PROFILE", "founder")
    os.environ.setdefault("DATABASE_PATH", os.path.join(tempfile.mkdtemp(prefix="harness_"), "harness.db"))

    from backend import cycle_service, sim_profile
    from backend.advise_service import build_env_state
    from backend.database import initialize_database
    from boardroom import expectation as ex

    initialize_database()

    with open(args.export, encoding="utf-8") as handle:
        store = json.load(handle)
    company = store["company"]
    months = sorted(store["months"], key=lambda m: m.get("index") or 0)
    horizon = max(1, args.horizon)
    if len(months) <= horizon:
        print(f"need at least {horizon + 1} months on record; found {len(months)}")
        return 2

    env_kwargs = sim_profile.get_env_kwargs(gross_margin=sim_profile.get_applied_gross_margin())
    rows = []
    for t in range(len(months) - horizon):
        month = months[t]
        future = months[t + horizon]
        payload = founder_payload(company, month, months[:t])
        payload["use_oracle"] = args.use_oracle
        cycle_months, _, _ = cycle_service.run_months(payload)
        action = cycle_months[0]["execute"]["action"]
        opening = build_env_state(payload)
        expected = ex.predict_expected_delta(opening, action, horizon_months=horizon, env_kwargs=env_kwargs)
        actual = build_env_state(founder_payload(company, future, months[:t + horizon]))
        realized = ex.kpi_delta_between(opening, actual)
        error = ex.prediction_error(expected, realized)
        rows.append({
            "from_index": month.get("index"), "to_index": future.get("index"),
            "action": action, "expected": {k: expected[k] for k in ex.KPI_KEYS},
            "realized": realized, "error": error,
        })
        print(f"month {month.get('index')} -> {future.get('index')}: "
              + ", ".join(
                  f"{k} exp {expected[k]:+.2f} got {realized[k]:+.2f}" if realized[k] is not None and expected[k] is not None
                  else f"{k} n/a" for k in ex.KPI_KEYS))

    print()
    print(f"calibration over {len(rows)} closes at a {horizon}-month horizon "
          f"({'oracle' if args.use_oracle else 'rules'} path, {sim_profile.get_profile()} profile)")
    report = {}
    for key in ex.KPI_KEYS:
        scored = [r["error"][key] for r in rows if r["error"] and r["error"].get(key)]
        if not scored:
            print(f"  {key:14} not scorable")
            continue
        errors = [s["error"] for s in scored]
        report[key] = {
            "n": len(scored),
            "mean_error": statistics.fmean(errors),
            "mean_abs_error": statistics.fmean(abs(e) for e in errors),
            "sign_agreement": sum(s["sign_agrees"] for s in scored) / len(scored),
            "within_tolerance": sum(s["within_tolerance"] for s in scored) / len(scored),
            "tolerance": ex.TOLERANCE[key],
        }
        r = report[key]
        print(f"  {key:14} n={r['n']:2d}  mean err {r['mean_error']:+7.2f}  MAE {r['mean_abs_error']:6.2f}  "
              f"direction {r['sign_agreement'] * 100:5.1f}%  within +/-{r['tolerance']}: {r['within_tolerance'] * 100:5.1f}%")
    print()
    print("Read this as the simulator's calibration against the company's real trajectory under what the "
          "founder actually did - the board's plan was not followed on these months, so it is not a measure of advice quality.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"company": company.get("name"), "horizon_months": horizon, "use_oracle": args.use_oracle,
                   "profile": sim_profile.get_profile(), "rows": rows, "report": report}, handle, indent=2)
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
