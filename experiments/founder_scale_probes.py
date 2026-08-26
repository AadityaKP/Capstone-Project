"""Measurements behind docs/founder_scale_fix_plan.md.

Every number in that document comes from here. Run any section on its own:

    venv/Scripts/python.exe experiments/founder_scale_probes.py rollout
    venv/Scripts/python.exe experiments/founder_scale_probes.py rnd
    venv/Scripts/python.exe experiments/founder_scale_probes.py whatif
    venv/Scripts/python.exe experiments/founder_scale_probes.py configs
    venv/Scripts/python.exe experiments/founder_scale_probes.py marketing
    venv/Scripts/python.exe experiments/founder_scale_probes.py all

`configs` simulates the two candidate fixes by patching rather than editing, so
the direction and size of each can be measured before committing to either. It
never writes to the source tree.
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


# A real founder, and the reason this plan exists: 20x smaller than the
# $50k-MRR company every constant in the engine is tuned for.
FOUNDER_SMALL = {
    "company_age_months": 8,
    "config": {
        "initial_mrr": 2500, "initial_cash": 5000, "average_price": 25, "cac": 50,
        "churn_enterprise": 0.10, "churn_smb": 0.10, "churn_b2c": 0.10,
        "competitors": 5, "product_quality": 0.5,
        # what api.js sends today: virtualHeadcount({costs: 500, marketingSpend: 10})
        #   = max(1, round(max(500 - 10, 8000) / 8000)) = 1  ->  charged $8,000/mo
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

TRUE_BURN = 500.0  # the founder's actual monthly costs


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
    """One instrumented path. Shows the death arithmetic month by month."""
    state = build_env_state(FOUNDER_SMALL)
    print(f"absolute_scale({state.mrr:,.0f}) = {absolute_scale(state.mrr)}  "
          f"(0.05 is the floor: every company under $2,500 MRR scales identically)")

    for tag, raw in (
        ("hold (zero spend)", None),
        ("board plan", FOUNDER_SMALL["recommended_action"]),
    ):
        action = W._clean_action(raw)
        env = StartupEnv(initial_config={
            "max_months": 10_000, "scheduled_shocks": False, "gross_margin": None,
        })
        env.reset(seed=0)
        env.state = state.model_copy(deep=True)

        print(f"\n=== {tag}")
        print(f"{'mo':>3} {'mrr':>10} {'cash':>11} {'R40':>9} {'salary':>8}  terminated")
        for month in range(12):
            salary = env.state.headcount * 8000.0
            _, _, terminated, _, info = env.step(deepcopy(action))
            s = env.state
            print(f"{month:>3} {s.mrr:>10.1f} {s.cash:>11.1f} "
                  f"{info['rule_of_40']:>9.1f} {salary:>8.0f}  {terminated}")
            if terminated:
                print(f"    -> dead at month index {month}, cash ${s.cash:,.1f}")
                break


# ---------------------------------------------------------------- section 3
def rnd() -> None:
    """R&D spend against product quality, and the churn drift underneath it."""
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
    """The shipped endpoint, at founder scale. Reproduces the screenshots."""
    result = W.run_whatif(deepcopy(FOUNDER_SMALL))
    print("shock_tape_shared:", result["shock_tape_shared"])
    for policy, data in result["policies"].items():
        s = data["summary"]
        print(f"\n{policy:<12} survival={s['survival_rate']:>5.0%}  "
              f"12mo MRR=${s['median_terminal_mrr']:>9,.0f}  "
              f"cash=${s['median_terminal_cash']:>10,.0f}")
        print("   mrr  :", [round(v) for v in data["series"]["mrr"]["median"]])
        print("   churn:", [round(v, 2) for v in data["series"]["churn"]["median"]])
    print("\nNo death month is reported anywhere in the payload:")
    print("  summary keys:", sorted(result["policies"]["hold"]["summary"]))


# -------------------------------------------------------------- section 2.1
def configs() -> None:
    """The four configurations. Both fixes simulated by patching, not editing."""
    original_step = StartupEnv.step
    original_make_env = W._make_env

    def patched_step(self, action):
        headcount = self.state.headcount
        obs, reward, terminated, truncated, info = original_step(self, action)
        # refund the fake $8k salary slot, charge the founder's real costs
        self.state.cash += headcount * 8000.0 - TRUE_BURN
        terminated = self.state.cash <= 0          # re-evaluate against corrected cash
        info["state"]["cash"] = self.state.cash
        return obs, reward, terminated, truncated, info

    def scale_aware_make_env(_state, gross_margin):
        return StartupEnv(initial_config={
            "max_months": 10_000, "scheduled_shocks": False,
            "gross_margin": gross_margin, "scale_aware_marketing": True,
        })

    def report(tag):
        result = W.run_whatif(deepcopy(FOUNDER_SMALL))
        print(f"\n########## {tag}")
        for policy, data in result["policies"].items():
            s = data["summary"]
            print(f"  {policy:<12} survival={s['survival_rate']:>5.0%}  "
                  f"12mo MRR=${s['median_terminal_mrr']:>9,.0f}  "
                  f"cash=${s['median_terminal_cash']:>10,.0f}")
            print(f"     mrr: {[round(v) for v in data['series']['mrr']['median']]}")

    try:
        report("A. as shipped today")
        StartupEnv.step = patched_step
        report(f"B. real burn (${TRUE_BURN:,.0f}/mo), marketing constants unchanged")
        W._make_env = scale_aware_make_env
        report("C. real burn + scale-aware marketing (both fixes)")
        StartupEnv.step = original_step
        report("D. scale-aware marketing only, fake $8k burn")
    finally:
        StartupEnv.step = original_step
        W._make_env = original_make_env


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


SECTIONS = {
    "rollout": rollout, "rnd": rnd, "whatif": whatif,
    "configs": configs, "marketing": marketing,
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
