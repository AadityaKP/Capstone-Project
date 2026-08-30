"""A4: decision sensibility / state responsiveness.

(i) Rule agents: direct evaluation of the documented thresholds (deterministic).
(ii) LLM brief pathway: one-variable sweeps. The LLM's entire influence on the
headline policies is state -> brief -> fixed ActionModifier arithmetic, so
responsiveness of the brief (and the multipliers it implies) IS the agent's
state responsiveness. llama3.1:8b, temperature 0, oracle_v1 prompt (no memory).

Writes validation/agents/state_responsiveness.csv (+ sweeps detail) and prints
Spearman monotonicity per sweep vs the pre-declared criterion (|rho| >= 0.5,
expected direction, >=3 of 4 sweeps).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent
from env.schemas import EnvState
from oracle.action_modifier import ActionModifier
from oracle.oracle import Oracle

OUT = ROOT / "validation/agents"
OUT.mkdir(parents=True, exist_ok=True)

ORD = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3,
       "ACCELERATING": 0, "STABLE": 1, "DECLINING": 2, "COLLAPSING": 3,
       "EXPANSION": 0, "NEUTRAL": 1, "RECESSION": 2}


def base_state(**over):
    d = dict(mrr=50_000.0, cash=600_000.0, cac=100.0, ltv=1_500.0,
             churn_enterprise=0.01, churn_smb=0.03, churn_b2c=0.05,
             interest_rate=3.0, consumer_confidence=100.0, competitors=5,
             product_quality=0.5, price=50.0, months_elapsed=24, headcount=4,
             valuation_multiple=10.0, unemployment=4.0, innovation_factor=1.0,
             months_in_depression=0)
    d.update(over)
    return EnvState(**d)


# ---------------------------------------------------- (i) rule agents
def rule_checks():
    rows = []
    # CFO: hires 1 iff runway > 24 and LTV/CAC >= 3
    for cash, expect in [(4 * 8000 * 30, 1), (4 * 8000 * 10, 0)]:
        st = base_state(cash=float(cash), cac=100.0, ltv=400.0)   # ratio 4
        got = CFOAgent().act(st)["hiring"]["hires"]
        rows.append(("CFO hires iff runway>24 & LTV/CAC>=3", cash, expect, got, expect == got))
    st = base_state(cash=4 * 8000 * 30.0, cac=100.0, ltv=250.0)   # ratio 2.5 < 3
    got = CFOAgent().act(st)["hiring"]["hires"]
    rows.append(("CFO blocks hire when LTV/CAC<3", "ratio2.5", 0, got, got == 0))
    # CMO spend steps by LTV/CAC
    for ltv, expect in [(500.0, 20000), (250.0, 10000), (150.0, 2000)]:
        got = CMOAgent().act(base_state(cac=100.0, ltv=ltv))["marketing"]["spend"]
        rows.append(("CMO spend steps with LTV/CAC", ltv / 100, expect, got, expect == got))
    # CPO R&D steps with churn; halved when cash < 200k
    for churn, expect in [(0.06, 15000), (0.03, 8000), (0.01, 3000)]:
        st = base_state(churn_enterprise=churn, churn_smb=churn, churn_b2c=churn)
        got = CPOAgent().act(st)["product"]["r_and_d_spend"]
        rows.append(("CPO R&D steps with churn", churn, expect, got, expect == got))
    st = base_state(cash=100_000.0, churn_enterprise=0.06, churn_smb=0.06, churn_b2c=0.06)
    got = CPOAgent().act(st)["product"]["r_and_d_spend"]
    rows.append(("CPO halves R&D when cash<200k", 100_000, 7500, got, got == 7500))
    return pd.DataFrame(rows, columns=["check", "input", "expected", "got", "ok"])


# ---------------------------------------------------- (ii) LLM brief sweeps
SWEEPS = {
    # var -> (state overrides per level, expected direction of risk ordinal)
    "runway_down": dict(levels=[1_200_000, 800_000, 500_000, 300_000, 150_000, 60_000],
                        make=lambda v: base_state(cash=float(v)),
                        expect="risk_up_as_level_drops"),
    "churn_up": dict(levels=[0.01, 0.02, 0.04, 0.06, 0.08, 0.10],
                     make=lambda v: base_state(churn_enterprise=v, churn_smb=v, churn_b2c=v),
                     expect="rd_scale_up"),
    "confidence_down": dict(levels=[130, 110, 90, 70, 55, 40],
                            make=lambda v: base_state(consumer_confidence=float(v)),
                            expect="marketing_scale_down_as_level_drops"),
    "competitors_up": dict(levels=[2, 4, 6, 8, 10, 12],
                           make=lambda v: base_state(competitors=int(v)),
                           expect="risk_up"),
}

ADVERSARIAL = {
    "high_growth_high_burn": base_state(mrr=150_000, cash=120_000, headcount=12),
    "low_growth_low_runway": base_state(mrr=20_000, cash=60_000, headcount=3),
    "strong_cash_weak_product": base_state(cash=2_000_000, product_quality=0.1),
    "severe_shock": base_state(consumer_confidence=45, unemployment=9.5,
                               interest_rate=8.5, months_in_depression=5),
    "strong_product_weak_acquisition": base_state(product_quality=0.95, cac=400.0, ltv=900.0),
    "declining_heavy_spend": base_state(mrr=35_000, cash=250_000, headcount=10,
                                        competitors=9),
}


def brief_row(oracle, mod, state, tag, level):
    brief = oracle.generate_brief(state)
    base_action = {"marketing": {"spend": 10_000.0, "channel": "ppc"},
                   "product": {"r_and_d_spend": 10_000.0},
                   "hiring": {"hires": 2, "cost_per_employee": 10_000.0},
                   "pricing": {"price_change_pct": 0.0}}
    out = mod.modify(base_action, brief)
    return dict(sweep=tag, level=level,
                risk=brief.risk_level.value, growth=brief.growth_outlook.value,
                efficiency=brief.efficiency_pressure.value,
                innovation=brief.innovation_urgency.value,
                macro=brief.macro_condition.value, confidence=brief.confidence,
                parse_ok=getattr(brief, "parse_ok", True),
                mkt_mult=out["marketing"]["spend"] / 10_000.0,
                rd_mult=out["product"]["r_and_d_spend"] / 10_000.0,
                hires_out=out["hiring"]["hires"])


def main():
    rc = rule_checks()
    rc.to_csv(OUT / "rule_agent_checks.csv", index=False)
    print(rc.to_string(index=False))
    print(f"rule checks: {rc.ok.sum()}/{len(rc)} exact\n")

    oracle = Oracle(mode="oracle_v1", memory_store=None, enable_memory_retrieval=False)
    mod = ActionModifier()
    rows = []
    for tag, cfg in SWEEPS.items():
        for v in cfg["levels"]:
            rows.append(brief_row(oracle, mod, cfg["make"](v), tag, v))
            print(f"  {tag} level={v}: {rows[-1]['risk']}/{rows[-1]['growth']}"
                  f" mkt x{rows[-1]['mkt_mult']:.2f} rd x{rows[-1]['rd_mult']:.2f}")
    for name, st in ADVERSARIAL.items():
        rows.append(brief_row(oracle, mod, st, f"adversarial:{name}", np.nan))
        r = rows[-1]
        print(f"  {name}: risk={r['risk']} growth={r['growth']} innov={r['innovation']}"
              f" mkt x{r['mkt_mult']:.2f} rd x{r['rd_mult']:.2f} hires->{r['hires_out']}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "llm_brief_sweeps.csv", index=False)

    # monotonicity vs pre-declared criteria
    checks = []
    def rho_of(tag, col, ordinal=False, flip=False):
        sub = df[df.sweep == tag]
        y = sub[col].map(ORD) if ordinal else sub[col]
        x = sub.level.astype(float)
        if flip:
            x = -x
        r = spearmanr(x, y).statistic
        return float(r) if np.isfinite(r) else 0.0

    checks.append(("runway_down", "risk rises as cash falls",
                   rho_of("runway_down", "risk", ordinal=True, flip=True), 0.5))
    checks.append(("churn_up", "R&D multiplier rises with churn",
                   rho_of("churn_up", "rd_mult"), 0.5))
    checks.append(("confidence_down", "marketing multiplier falls with confidence",
                   rho_of("confidence_down", "mkt_mult"), 0.5))
    checks.append(("competitors_up", "risk rises with competitors",
                   rho_of("competitors_up", "risk", ordinal=True), 0.5))
    cdf = pd.DataFrame(checks, columns=["sweep", "expectation", "spearman_rho", "threshold"])
    cdf["pass"] = cdf.spearman_rho >= cdf.threshold
    cdf.to_csv(OUT / "state_responsiveness.csv", index=False)
    print("\n", cdf.to_string(index=False))
    print(f"\nLLM sweeps passing: {int(cdf['pass'].sum())}/4 (criterion: >=3)")

    exploratory_battery(oracle, mod)


# ------------------------------------------------------------------
# EXPLORATORY (designed AFTER the pre-declared battery failed 0/4):
# the recorded FULL run shows brief distributions shifting at shock
# months (risk MEDIUM share 0.44 -> 0.90 for v1), suggesting the model
# responds to the SHOCK_ALERT line and trend deltas rather than state
# levels. This battery characterizes that channel. Its results are
# labelled exploratory and do not overwrite the pre-declared verdict.
# ------------------------------------------------------------------
from oracle.schemas import TrendContext, TrendDirection  # noqa: E402


def trend(delta_pct, churn_delta=0.0):
    prev = 50_000.0
    cur = prev * (1 + delta_pct / 100.0)
    def dirn(x):
        return (TrendDirection.INCREASING if x > 0.01
                else TrendDirection.DECREASING if x < -0.01
                else TrendDirection.FLAT)
    return TrendContext(
        mrr_trend=dirn(delta_pct), innovation_trend=TrendDirection.FLAT,
        churn_trend=dirn(churn_delta * 100), history_points=5,
        previous_mrr=prev, current_mrr=cur, mrr_delta_pct=delta_pct,
        previous_avg_churn=0.03, current_avg_churn=0.03 + churn_delta,
        churn_delta=churn_delta)


def exploratory_battery(oracle, mod):
    rows = []
    base_action = {"marketing": {"spend": 10_000.0, "channel": "ppc"},
                   "product": {"r_and_d_spend": 10_000.0},
                   "hiring": {"hires": 2, "cost_per_employee": 10_000.0},
                   "pricing": {"price_change_pct": 0.0}}

    def call(tag, level, state, tc=None, shock=None):
        brief = oracle.generate_brief(state, trend_context=tc, memories=[],
                                      shock_label=shock)
        out = mod.modify(base_action, brief)
        rows.append(dict(sweep=tag, level=level, shock=shock,
                         risk=brief.risk_level.value, growth=brief.growth_outlook.value,
                         innovation=brief.innovation_urgency.value,
                         macro=brief.macro_condition.value,
                         mkt_mult=out["marketing"]["spend"] / 10_000.0,
                         rd_mult=out["product"]["r_and_d_spend"] / 10_000.0,
                         hires_out=out["hiring"]["hires"]))
        r = rows[-1]
        print(f"  [explor] {tag} level={level} shock={bool(shock)}: risk={r['risk']} "
              f"growth={r['growth']} mkt x{r['mkt_mult']:.2f} rd x{r['rd_mult']:.2f}")

    print("\n--- exploratory battery (trend/shock channel) ---")
    for d in [10, 5, 0, -5, -15, -30]:
        call("mrr_trend_delta", d, base_state(), tc=trend(d))
    for cd in [0.0, 0.01, 0.02, 0.03]:
        call("churn_delta", cd, base_state(), tc=trend(0, churn_delta=cd))
    for shock in [None,
                  "COMPETITOR_SURGE: 3 new entrants, forced price cut, SMB churn +50%",
                  "RATE_HIKE: +400bps, valuation -40%, confidence crash",
                  "RECESSION: confidence=55, unemployment spike, B2C churn doubled"]:
        call("shock_alert", 0 if shock is None else 1, base_state(), tc=trend(0),
             shock=shock)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "llm_brief_sweeps_exploratory.csv", index=False)

    sub = df[df.sweep == "mrr_trend_delta"]
    rho_g = spearmanr(-sub.level.astype(float), sub.growth.map(ORD)).statistic
    sub2 = df[df.sweep == "churn_delta"]
    rho_c = spearmanr(sub2.level.astype(float), sub2.rd_mult).statistic
    shock_rows = df[df.sweep == "shock_alert"]
    risk_no = shock_rows[shock_rows.level == 0].risk.map(ORD).mean()
    risk_yes = shock_rows[shock_rows.level == 1].risk.map(ORD).mean()
    summary = pd.DataFrame([
        dict(check="growth_outlook worsens as MRR delta falls (exploratory)",
             stat=f"spearman={rho_g:+.2f}"),
        dict(check="R&D multiplier rises with churn delta (exploratory)",
             stat=f"spearman={rho_c:+.2f}" if np.isfinite(rho_c) else "constant"),
        dict(check="risk ordinal, shock vs no shock (exploratory)",
             stat=f"{risk_no:.1f} -> {risk_yes:.1f}"),
    ])
    summary.to_csv(OUT / "state_responsiveness_exploratory.csv", index=False)
    print("\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
