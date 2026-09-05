"""Deterministic level assessment for brief v2 (BRIEF_V2_SPEC.md, round 2).

One place computes the level bands used by BOTH the v2 prompt block and the
v2b guardrail floors, so they can never disagree. Every threshold mirrors a
constant that already exists in the physics or the calibration
(competitors >= 4 / >= 10, confidence < 80, LTV:CAC >= 3 hiring gate,
2.7%/mo ChartMogul churn-band median, runway bands from the boardroom's
refresh trigger scale) - no new tunables (spec rule).
"""
from __future__ import annotations

from env import business_logic
from env.schemas import EnvState

DEFAULT_CHURN_BAND_MEDIAN = 0.027  # ChartMogul $100-250 ARPA monthly logo churn

SEVERITY = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
# unit-economics bands map onto the risk scale for the floor rule:
# HEALTHY is no pressure, PRESSURED sits at MEDIUM, CRITICAL is CRITICAL.
UNIT_ECON_SEVERITY = {"HEALTHY": 0, "PRESSURED": 1, "CRITICAL": 3}


def compute_level_assessment(state: EnvState,
                             churn_band_median: float | None = None) -> dict:
    band_median = (DEFAULT_CHURN_BAND_MEDIAN if churn_band_median is None
                   else float(churn_band_median))
    burn = max(1.0, business_logic.monthly_burn(state))
    runway_months = state.cash / burn
    if runway_months < 6:
        runway_band = "CRITICAL"
    elif runway_months < 12:
        runway_band = "HIGH"
    elif runway_months < 24:
        runway_band = "MEDIUM"
    else:
        runway_band = "LOW"

    avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0
    churn_ratio = avg_churn / max(band_median, 1e-9)
    if churn_ratio > 1.5:
        churn_band = "HIGH"
    elif churn_ratio >= 1.15:
        churn_band = "ELEVATED"
    elif churn_ratio >= 0.85:
        churn_band = "NORMAL"
    else:
        churn_band = "LOW"

    ltv_cac = state.ltv / max(state.cac, 1e-9)
    if ltv_cac < 1.0:
        unit_econ_band = "CRITICAL"
    elif ltv_cac < 3.0:
        unit_econ_band = "PRESSURED"
    else:
        unit_econ_band = "HEALTHY"

    if (state.consumer_confidence < 80 or state.unemployment > 7.0
            or state.months_in_depression >= 3):
        macro_regime = "RECESSION"
    elif state.consumer_confidence > 110 and state.unemployment < 5.0:
        macro_regime = "EXPANSION"
    else:
        macro_regime = "NEUTRAL"

    if state.competitors >= 10:
        comp_band = "SEVERE"
    elif state.competitors >= 4:
        comp_band = "ELEVATED"
    else:
        comp_band = "LOW"

    net_burn = burn - state.mrr
    burn_pct_mrr = net_burn / max(state.mrr, 1.0) * 100.0

    return dict(runway_months=runway_months, runway_band=runway_band,
                avg_churn=avg_churn, band_median=band_median,
                churn_ratio=churn_ratio, churn_band=churn_band,
                ltv_cac=ltv_cac, unit_econ_band=unit_econ_band,
                macro_regime=macro_regime,
                competitors=state.competitors, comp_band=comp_band,
                net_burn=net_burn, burn_pct_mrr=burn_pct_mrr)


def level_block(levels: dict) -> list[str]:
    """The LEVEL ASSESSMENT prompt section, exactly as specified."""
    return [
        "--- LEVEL ASSESSMENT (computed deterministically; use these bands) ---",
        f"Runway: {levels['runway_months']:.1f} months  [band: {levels['runway_band']}]",
        "  bands: <6 CRITICAL | 6-12 HIGH | 12-24 MEDIUM | >24 LOW",
        f"Churn vs benchmark: {levels['avg_churn']:.3f} vs band median "
        f"{levels['band_median']:.3f} → {levels['churn_ratio']:.2f}× [{levels['churn_band']}]",
        "  bands: >1.5× HIGH | 1.15-1.5× ELEVATED | 0.85-1.15× NORMAL | <0.85× LOW",
        f"LTV:CAC: {levels['ltv_cac']:.2f}  [{levels['unit_econ_band']}]",
        "  bands: <1.0 CRITICAL | 1.0-3.0 PRESSURED | ≥3.0 HEALTHY",
        f"Macro regime: {levels['macro_regime']}",
        "  rule: RECESSION if confidence < 80 or unemployment > 7.0 or months_in_depression ≥ 3;",
        "        EXPANSION if confidence > 110 and unemployment < 5.0; else NEUTRAL",
        f"Competitive pressure: {levels['competitors']} competitors [{levels['comp_band']}]",
        "  bands: ≥10 SEVERE | 4-9 ELEVATED | <4 LOW      (these are the thresholds the market uses)",
        f"Cash burn this month: {levels['net_burn']:,.0f}  ({levels['burn_pct_mrr']:.0f}% of MRR)",
    ]


def apply_brief_floors(brief, levels: dict):
    """v2b guardrails: deterministic floors after parsing (spec change 2).

    The LLM may be MORE severe, never less. Returns (brief, floor_log);
    floor_log lists every override applied, for the decision trace.
    """
    from oracle.schemas import MacroCondition, RiskLevel

    floor_log: list[str] = []
    risk = brief.risk_level.value if hasattr(brief.risk_level, "value") else str(brief.risk_level)
    floor_sev = max(SEVERITY.get(levels["runway_band"], 0),
                    UNIT_ECON_SEVERITY.get(levels["unit_econ_band"], 0))
    if SEVERITY.get(risk, 1) < floor_sev:
        floored = [k for k, v in SEVERITY.items() if v == floor_sev][0]
        floor_log.append(f"risk_level {risk}->{floored} "
                         f"(runway {levels['runway_band']}, "
                         f"unit econ {levels['unit_econ_band']})")
        brief = brief.model_copy(update={"risk_level": RiskLevel(floored)})

    macro = (brief.macro_condition.value if hasattr(brief.macro_condition, "value")
             else str(brief.macro_condition))
    if levels["macro_regime"] == "RECESSION" and macro != "RECESSION":
        floor_log.append(f"macro_condition {macro}->RECESSION (computed regime)")
        brief = brief.model_copy(
            update={"macro_condition": MacroCondition("RECESSION")})

    return brief, floor_log
