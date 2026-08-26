"""Measurements behind docs/founder_scale_fix_plan.md.

Every number in that document comes from here. Run any section on its own:

    venv/Scripts/python.exe experiments/founder_scale_probes.py rollout
    venv/Scripts/python.exe experiments/founder_scale_probes.py rnd
    venv/Scripts/python.exe experiments/founder_scale_probes.py whatif
    venv/Scripts/python.exe experiments/founder_scale_probes.py configs
    venv/Scripts/python.exe experiments/founder_scale_probes.py marketing
    venv/Scripts/python.exe experiments/founder_scale_probes.py cac
    venv/Scripts/python.exe experiments/founder_scale_probes.py all

`configs` is the before/after for steps 1 and 2. The "before" arm is not a
reconstruction: omitting `monthly_costs` from the payload puts the engine back on
its headcount-slot convention, which is exactly the state the product shipped in.
"""

from __future__ import annotations

import math
import os
import random
import statistics
import sys
from copy import deepcopy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend import whatif_service as W
from backend.advise_service import absolute_scale, build_env_state
from env import business_logic as B
from env.schemas import EnvState, MarketingAction, ProductAction
from env.startup_env import StartupEnv


# A real founder, and the reason this work exists: 20x smaller than the
# $50k-MRR company every constant in the engine was tuned for.
FOUNDER_SMALL = {
    "company_age_months": 8,
    "config": {
        "initial_mrr": 2500, "initial_cash": 5000, "average_price": 25, "cac": 50,
        "churn_enterprise": 0.10, "churn_smb": 0.10, "churn_b2c": 0.10,
        "competitors": 5, "product_quality": 0.5,
        "monthly_costs": 500,          # the number that never used to arrive
        "initial_headcount": 1,
    },
    "recommended_action": {
        "marketing": {"spend": 250, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10000},
        "product": {"r_and_d_spend": 100},
    },
    "current_marketing_spend": 10,
    "n_seeds": 20,
}


def _legacy(payload: dict) -> dict:
    """The same founder with costs withheld: the engine's own salary slot.

    This is the shipped behaviour before step 1, reached through the supported
    code path rather than by patching anything.
    """
    out = deepcopy(payload)
    out["config"].pop("monthly_costs", None)
    return out


def _state(**overrides) -> EnvState:
    base = dict(
        mrr=2500.0, cash=5000.0, cac=50.0, ltv=250.0,
        churn_enterprise=0.10, churn_smb=0.10, churn_b2c=0.10,
        interest_rate=3.0, consumer_confidence=100.0, competitors=5,
        product_quality=0.5, price=25.0, months_elapsed=8, headcount=1,
        valuation_multiple=10.0, unemployment=4.0, innovation_factor=1.0,
        months_in_depression=0,
    )
    base.update(overrides)
    return EnvState(**base)


# ---------------------------------------------------------------- section 0
def rollout() -> None:
    """One instrumented path, on the founder's real costs and on the slot."""
    for tag, payload in (("real costs", FOUNDER_SMALL), ("salary slot", _legacy(FOUNDER_SMALL))):
        state = build_env_state(payload)
        action = W._clean_action(payload["recommended_action"])
        env = StartupEnv(initial_config={
            "max_months": 10_000, "scheduled_shocks": False,
            "scale_aware_marketing": True,
        })
        env.reset(seed=0)
        env.state = state.model_copy(deep=True)

        print(f"\n=== {tag}: burn = ${B.monthly_burn(state):,.0f}/mo")
        print(f"{'mo':>3} {'mrr':>10} {'cash':>11} {'R40':>9} {'burn':>9}  terminated")
        for month in range(12):
            burn = B.monthly_burn(env.state)
            _, _, terminated, _, info = env.step(deepcopy(action))
            s = env.state
            print(f"{month:>3} {s.mrr:>10.1f} {s.cash:>11.1f} "
                  f"{info['rule_of_40']:>9.1f} {burn:>9,.0f}  {terminated}")
            if terminated:
                print(f"    -> dead at month index {month}, cash ${s.cash:,.1f}")
                break


# ---------------------------------------------------------------- section 3
def rnd() -> None:
    """R&D spend against product quality, and the churn drift underneath it.

    Still broken, and still the reason churn barely separates the arms. Step 3
    of the plan; nothing in steps 1 and 2 touches it.
    """
    print("=== R&D effect, innovation_factor = 1.0 (the founder default) ===")
    print("    gain *= (1.0 - innovation_factor)  ->  the product is exactly zero")
    for spend in (0, 200, 1_000, 5_000, 50_000, 500_000):
        s = _state()
        B.apply_innovation_investment(s, ProductAction(r_and_d_spend=float(spend)))
        print(f"  ${spend:>8,}: quality 0.5 -> {s.product_quality!r}   "
              f"innovation 1.0 -> {s.innovation_factor!r}")

    print("\n=== same, starting from innovation_factor = 0.8 ===")
    for spend in (0, 200, 5_000, 50_000):
        s = _state(innovation_factor=0.8)
        B.apply_innovation_investment(s, ProductAction(r_and_d_spend=float(spend)))
        print(f"  ${spend:>8,}: quality -> {s.product_quality:.6f}   "
              f"innovation -> {s.innovation_factor:.6f}")

    print("\n=== churn over a 12-month horizon, no lever touched (tenure decay only) ===")
    for m in range(8, 21):
        decay = max(0.3, math.exp(-0.15 * max(1, m * 0.4)))
        print(f"  months_elapsed={m:>2}: churn={B.compute_churn_rate(_state(months_elapsed=m)) * 100:.3f}%"
              f"   decay={decay:.3f}")

    print("\n=== expansion MRR: a free 2%/month tailwind regardless of spend ===")
    for spend in (0, 200, 5_000):
        s = _state()
        exp = B.compute_expansion_mrr(s, ProductAction(r_and_d_spend=float(spend)))
        print(f"  r&d ${spend:>6,}: expansion = ${exp:.2f}  ({exp / s.mrr * 100:.2f}% of MRR)")


# ---------------------------------------------------------------- section 0
def whatif() -> None:
    """The shipped endpoint, at founder scale."""
    result = W.run_whatif(deepcopy(FOUNDER_SMALL))
    print("starting_state:", result["starting_state"])
    print("shock_tape_shared:", result["shock_tape_shared"])
    for policy, data in result["policies"].items():
        s = data["summary"]
        print(f"\n{policy:<12} survival={s['survival_rate']:>5.0%}  "
              f"12mo MRR=${s['median_terminal_mrr']:>9,.0f}  "
              f"cash=${s['median_terminal_cash']:>10,.0f}")
        print("   mrr  :", [round(v) for v in data["series"]["mrr"]["median"]])
        print("   churn:", [round(v, 2) for v in data["series"]["churn"]["median"]])
    print("\nStill no death month in the payload - that is step 4:")
    print("  summary keys:", sorted(result["policies"]["hold"]["summary"]))


# -------------------------------------------------------------- section 2.1
def configs() -> None:
    """Before and after, both reached through the supported payload."""
    for tag, payload in (
        ("BEFORE - costs withheld, engine charges its $8k salary slot", _legacy(FOUNDER_SMALL)),
        ("AFTER  - costs supplied, scale-aware marketing on", FOUNDER_SMALL),
    ):
        result = W.run_whatif(deepcopy(payload))
        burn = result["starting_state"]["monthly_burn"]
        print(f"\n########## {tag}")
        print(f"           burn charged: ${burn:,.0f}/mo "
              f"(supplied={result['starting_state']['monthly_burn_supplied']})")
        for policy, data in result["policies"].items():
            s = data["summary"]
            print(f"  {policy:<12} survival={s['survival_rate']:>5.0%}  "
                  f"12mo MRR=${s['median_terminal_mrr']:>9,.0f}  "
                  f"cash=${s['median_terminal_cash']:>10,.0f}")
            print(f"     mrr: {[round(v) for v in data['series']['mrr']['median']]}")


# ---------------------------------------------------------------- section 2
def marketing() -> None:
    """Is small-scale marketing too weak or too strong? Both channels, three sizes."""
    def mean_new_mrr(state, spend, channel, scale_aware, n=4000):
        random.seed(7)
        return statistics.mean(
            B.compute_new_mrr(state, MarketingAction(spend=spend, channel=channel),
                              scale_aware=scale_aware)
            for _ in range(n)
        )

    print(f"{'company':>10} {'spend':>7} {'chan':>6} | "
          f"{'ABSOLUTE':>12} {'% of MRR':>9} | {'SCALE-AWARE':>12} {'% of MRR':>9}")
    for mrr, price, cac in ((2_500, 25, 50), (12_000, 50, 300), (50_000, 50, 300)):
        s = _state(mrr=float(mrr), price=float(price), cac=float(cac),
                   cash=50_000.0, ltv=price / 0.05,
                   churn_enterprise=0.05, churn_smb=0.05, churn_b2c=0.05)
        for share in (0.05, 0.10, 0.25):
            dollars = mrr * share
            for channel in ("ppc", "brand"):
                absolute = mean_new_mrr(s, dollars, channel, False)
                aware = mean_new_mrr(s, dollars, channel, True)
                print(f"${mrr:>9,} ${dollars:>6,.0f} {channel:>6} | "
                      f"${absolute:>11,.0f} {absolute / mrr * 100:>8.1f}% | "
                      f"${aware:>11,.0f} {aware / mrr * 100:>8.1f}%")
        print()


# --------------------------------------------------------------- section 2.2
def cac() -> None:
    """The loop the scale-aware curve closes, and the guard that breaks it.

    marketing_curve_params places gamma from state.cac. A month whose response
    is a fraction of a customer writes an enormous CAC, that CAC pushes gamma
    further right, and the next response is smaller still. Nothing damps it.
    """
    starved = {
        "marketing": {"spend": 1.0, "channel": "brand"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 0.0},
        "pricing": {"price_change_pct": 0.0},
    }
    base = build_env_state(FOUNDER_SMALL)

    for guard in (False, True):
        env = StartupEnv(initial_config={
            "max_months": 10_000, "scheduled_shocks": False,
            "scale_aware_marketing": True, "stable_cac": guard,
        })
        env.reset(seed=0)
        env.state = base.model_copy(deep=True)
        print(f"\n=== stable_cac={guard}   ($1/mo of brand spend, far left of gamma)")
        try:
            for month in range(10):
                _, gamma_beta, gamma = B.marketing_curve_params(env.state, "brand")
                print(f"  month {month:>2}: cac={env.state.cac:>12.4g}  gamma={gamma:>12.4g}")
                env.step(deepcopy(starved))
        except OverflowError as exc:
            print(f"  month {month:>2}: OverflowError from hill_response - {exc}")


SECTIONS = {
    "rollout": rollout, "rnd": rnd, "whatif": whatif,
    "configs": configs, "marketing": marketing, "cac": cac,
}

if __name__ == "__main__":
    requested = sys.argv[1] if len(sys.argv) > 1 else "all"
    if requested == "all":
        for name, fn in SECTIONS.items():
            print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
            fn()
    elif requested in SECTIONS:
        SECTIONS[requested]()
    else:
        print(f"unknown section {requested!r}; pick one of {', '.join(SECTIONS)} or 'all'")
        sys.exit(2)
