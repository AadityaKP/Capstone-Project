"""Engine vocabulary -> founder vocabulary. The only place the translation lives.

The simulator carries a research ontology: Rule of 40, innovation_factor,
valuation_multiple, a churn rate that is decayed by tenure and is therefore not
the churn the founder typed. The product is aimed at people who do not have that
ontology, and every one of those terms leaking onto a screen was either noise or
an outright contradiction - "runway infinity" beside "0% survival", "High
confidence" beside "6 estimated inputs", "Healthy" beside a 100x LTV/CAC ratio
derived from ten customers.

The rule this module exists to enforce: **anything with a name a founder would
have to Google gets translated here, at the boundary.** Engine names stay in the
API response under their own keys for debugging; nothing downstream renders one.

Thresholds live in config/founder_view.json rather than in this file, because
the client is local-first - Home computes runway and unit economics from numbers
the browser never sends - so a few of these rules necessarily exist on both
sides. frontend/src/founderView.js reads the same JSON, so the numbers are
shared even though the code is not.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any

_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config",
    "founder_view.json",
)

_RULES: dict[str, Any] | None = None


def rules() -> dict[str, Any]:
    """The shared thresholds. Loaded once; keys starting with _ are commentary."""
    global _RULES
    if _RULES is None:
        with open(_CONFIG_PATH, encoding="utf-8") as handle:
            _RULES = {
                key: value
                for key, value in json.load(handle).items()
                if not key.startswith("_")
            }
    return _RULES


def _money(value: float) -> str:
    """Whole dollars, with a k suffix once the cents stop mattering."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    magnitude = abs(value)
    sign = "-" if value < 0 else ""
    if magnitude >= 1_000_000:
        return f"{sign}${magnitude / 1_000_000:.2f}M"
    if magnitude >= 10_000:
        return f"{sign}${magnitude / 1_000:,.0f}k"
    return f"{sign}${magnitude:,.0f}"


# --------------------------------------------------------------------------
# Money in, money out
# --------------------------------------------------------------------------
def spend_ratio(monthly_outflow: float, mrr: float) -> float | None:
    """Dollars spent per dollar of revenue. None when there is no revenue yet."""
    if mrr is None or mrr <= 0:
        return None
    return monthly_outflow / mrr


def spend_ratio_phrase(monthly_outflow: float, mrr: float) -> str:
    """The founder-facing replacement for Rule of 40.

    "-339" tells a founder at $2,500 MRR nothing. "You spend $3.40 for every $1
    of revenue" is the same fact in a unit they already think in, and it stays
    meaningful at every company size, which Rule of 40 does not.
    """
    ratio = spend_ratio(monthly_outflow, mrr)
    if ratio is None:
        return f"You're spending {_money(monthly_outflow)} a month with no revenue yet"
    return f"You spend ${ratio:,.2f} for every $1 of revenue"


def rule_of_40_is_meaningful(mrr: float) -> bool:
    """Rule of 40 is a public-SaaS benchmark. Below roughly $1M ARR it is noise,
    and printing it invites a founder to optimise a number that does not apply
    to them."""
    return bool(mrr) and mrr >= rules()["rule_of_40_mrr_floor"]


# --------------------------------------------------------------------------
# Runway
# --------------------------------------------------------------------------
def runway_months(cash: float, monthly_burn: float, mrr: float) -> float | None:
    """Months of cash against NET burn. None means not burning cash - never
    infinity, which is a number no screen should ever print."""
    if cash is None or cash <= 0:
        return 0.0
    net_burn = (monthly_burn or 0.0) - (mrr or 0.0)
    if net_burn <= 0:
        return None
    return cash / net_burn


def runway_phrase(cash: float, monthly_burn: float, mrr: float) -> str:
    months = runway_months(cash, monthly_burn, mrr)
    if months is None:
        return "At current costs you're not burning cash"
    if months <= 0:
        return "No cash left"
    if months < 1:
        return "Less than a month of cash at current costs"
    if months >= 60:
        return "Over 5 years of cash at current costs"
    return f"{months:.0f} month{'s' if round(months) != 1 else ''} of cash at current costs"


# --------------------------------------------------------------------------
# Churn
# --------------------------------------------------------------------------
def churn_phrase(monthly_churn_pct: float) -> str:
    """A count, not a percentage.

    Kept deliberately distinct from the founder's own entered rate: the engine's
    churn is decayed by customer tenure and moved by product quality, so it is a
    different quantity that happens to share a name. Screens that show both must
    never use one word for the two.
    """
    if monthly_churn_pct is None or monthly_churn_pct <= 0:
        return "You're not losing customers"
    if monthly_churn_pct >= 50:
        return "You lose more than half your customers every month"
    if monthly_churn_pct < 1:
        return "You lose fewer than 1 in 100 customers a month"
    return f"You lose about 1 in {round(100 / monthly_churn_pct)} customers a month"


# --------------------------------------------------------------------------
# Unit economics
# --------------------------------------------------------------------------
def efficiency(ltv: float | None, cac: float | None,
               new_customers: float | None = None) -> dict[str, Any]:
    """LTV:CAC as a judgement a founder can act on, or an honest refusal.

    The old band said "Healthy" for any ratio at or above 3, with no ceiling. A
    founder who spent $10 and signed 10 customers has a $1 acquisition cost and
    a 100x ratio, and printing "Healthy" for that dressed a measurement artifact
    up as a finding. Above the noise threshold this now declines to answer.
    """
    threshold = rules()
    if not ltv or not cac or cac <= 0:
        return {"band": "unknown", "label": "Not enough data yet",
                "detail": "Add last month's marketing spend and new customers.",
                "ratio": None}

    minimum = threshold["cac_min_customers"]
    if new_customers is not None and 0 < new_customers < minimum:
        return {"band": "unknown", "label": "Not enough data yet",
                "detail": f"{int(new_customers)} new customers is too few to price acquisition from.",
                "ratio": None}

    ratio = ltv / cac
    detail = f"Each customer costs {_money(cac)} to win and pays back {_money(ltv)}."

    # The ceiling exists because an enormous ratio usually means a tiny
    # denominator - $10 of marketing and ten signups is a $1 acquisition cost.
    # When the denominator is visible and large enough to trust, that reason is
    # gone, and refusing to answer on good data is its own kind of dishonesty.
    # Measured on the sample company: 44 customers, $91 acquisition cost, a 20.3x
    # ratio - well-founded, and the flat ceiling was calling it unmeasurable.
    verified_sample = new_customers is not None and new_customers >= minimum
    if ratio > threshold["ltv_cac_unmeasurable"] and not verified_sample:
        return {"band": "unknown", "label": "Can't measure this yet",
                "detail": (f"{detail} A return that large usually means the acquisition "
                           "cost came from too few customers to trust."),
                "ratio": ratio}
    if ratio >= threshold["ltv_cac_healthy"]:
        # Lifetime value is price divided by churn, so a long payback is a
        # statement about churn holding, not an observation. Say so once the
        # number gets large enough for that to matter.
        caveat = (
            " That payback assumes your churn stays where it is."
            if ratio > threshold["ltv_cac_unmeasurable"] else ""
        )
        return {"band": "healthy", "label": "Healthy",
                "detail": detail + caveat, "ratio": ratio}
    if ratio >= threshold["ltv_cac_watch"]:
        return {"band": "watch", "label": "Worth watching", "detail": detail, "ratio": ratio}
    return {"band": "unhealthy", "label": "Costs more than it returns",
            "detail": detail, "ratio": ratio}


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------
def confidence(model_confidence: float | None, assumed_count: int) -> dict[str, Any]:
    """The model's own confidence, capped by how much nobody measured.

    These were two independent numbers rendered side by side, so the strip could
    read "High confidence - 6 estimated inputs". Six unknowns is not high
    confidence in any reading of the words. The count now caps the band, which
    makes that sentence impossible rather than merely unlikely.
    """
    if model_confidence is None:
        band = "Moderate"
    elif model_confidence < 0.4:
        band = "Low"
    elif model_confidence <= 0.7:
        band = "Moderate"
    else:
        band = "High"

    order = ["Low", "Moderate", "High"]
    cap = "Low"
    for rule in rules()["confidence_caps"]:
        if assumed_count <= rule["max_assumed"]:
            cap = rule["cap"]
            break

    capped = order[min(order.index(band), order.index(cap))]
    if assumed_count <= 0:
        sentence = f"{capped} confidence, from the numbers you gave us"
    else:
        sentence = (
            f"{capped} confidence - {assumed_count} of these numbers "
            f"{'is an estimate' if assumed_count == 1 else 'are estimates'}, not yours"
        )
    return {"band": capped, "model_band": band, "assumed_count": assumed_count,
            "sentence": sentence, "capped": capped != band}


# --------------------------------------------------------------------------
# Projection outcomes
# --------------------------------------------------------------------------
def survival_phrase(summary: dict[str, Any]) -> str:
    """"Survives 0%" said nothing about whether the company lasted one month or
    eleven. This says which."""
    runs = summary.get("runs") or 0
    deaths = summary.get("deaths") or 0
    if not runs:
        return "-"
    if not deaths:
        return f"Still going after every one of {runs} runs"

    month = summary.get("median_death_month")
    where = "in every run" if deaths == runs else f"in {deaths} of {runs} runs"
    when = f", typically around month {month}" if month is not None else ""
    return f"Ran out of cash {where}{when}"


def translate_projection(result: dict[str, Any]) -> dict[str, Any]:
    """Add a `display` block to each policy. Engine keys are left untouched."""
    starting = result.get("starting_state") or {}
    mrr = starting.get("mrr") or 0.0
    show_rule_of_40 = rule_of_40_is_meaningful(mrr)

    for policy in (result.get("policies") or {}).values():
        summary = policy.get("summary") or {}
        ratios = (policy.get("series") or {}).get("spend_ratio") or {}
        median = [v for v in (ratios.get("median") or []) if v is not None]
        policy["display"] = {
            "survival": survival_phrase(summary),
            "revenue": _money(summary.get("median_terminal_mrr")),
            "cash": _money(summary.get("median_terminal_cash")),
            "spend_ratio": (
                f"${median[-1]:,.2f} per $1" if median else "-"
            ),
        }

    result["display"] = {
        # The client asks this before deciding which efficiency panel to draw,
        # so the floor is never hardcoded downstream.
        "show_rule_of_40": show_rule_of_40,
        "efficiency_panel_label": (
            "Rule of 40" if show_rule_of_40 else "Spend per $1 of revenue"
        ),
        "efficiency_panel_series": "rule_of_40" if show_rule_of_40 else "spend_ratio",
        "rule_of_40_withheld_because": (
            None if show_rule_of_40 else
            "Rule of 40 is a public-company benchmark and doesn't mean anything below "
            "about $1M a year in revenue, so this shows what you spend per dollar earned "
            "instead."
        ),
    }
    return result
