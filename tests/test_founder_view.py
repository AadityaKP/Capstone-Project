"""The engine-to-founder translation layer (backend/founder_view.py).

Every contradiction this module exists to make impossible gets a test that fails
if it becomes possible again:

  - "High confidence" beside "6 estimated inputs"
  - "Runway ∞" beside "0% survival"
  - "Healthy" beside a 100x return built from ten customers
  - "Rule of 40: -339" shown to a company at $2,500 MRR
"""

from __future__ import annotations

import io
import json
import os
import re

import pytest

from backend import founder_view as fv


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- confidence -----------------------------------------------------------

@pytest.mark.parametrize("assumed,expected", [(0, "High"), (1, "High"), (2, "Moderate"),
                                              (4, "Moderate"), (5, "Low"), (9, "Low")])
def test_assumption_count_caps_the_confidence_band(assumed, expected):
    """The model can be as sure as it likes; six unknowns is not high confidence."""
    assert fv.confidence(0.95, assumed)["band"] == expected


def test_high_confidence_beside_six_estimates_is_unsayable():
    result = fv.confidence(0.95, 6)
    assert result["band"] == "Low"
    assert result["capped"] is True
    assert "6 of these numbers are estimates" in result["sentence"]
    assert "High" not in result["sentence"]


def test_the_cap_only_lowers_never_raises():
    """A model that is unsure stays unsure however complete the inputs are."""
    assert fv.confidence(0.1, 0)["band"] == "Low"


# --- runway ---------------------------------------------------------------

def test_runway_is_never_infinity():
    assert fv.runway_months(cash=5_000, monthly_burn=500, mrr=2_500) is None
    assert "not burning cash" in fv.runway_phrase(5_000, 500, 2_500)
    assert "∞" not in fv.runway_phrase(5_000, 500, 2_500)


def test_runway_counts_months_when_actually_burning():
    assert fv.runway_months(cash=45_000, monthly_burn=70_000, mrr=30_000) == pytest.approx(1.125)
    assert fv.runway_phrase(45_000, 70_000, 30_000) == "1 month of cash at current costs"


def test_no_cash_is_said_plainly():
    assert fv.runway_phrase(0, 500, 2_500) == "No cash left"


# --- unit economics -------------------------------------------------------

def test_a_hundred_times_return_is_refused_when_the_sample_is_unknown():
    """$10 of marketing and a $1 CAC is an artifact when nothing says how many
    customers produced it."""
    result = fv.efficiency(ltv=100, cac=1)
    assert result["band"] == "unknown"
    assert result["label"] == "Can't measure this yet"


def test_a_large_ratio_from_a_verified_sample_is_not_refused():
    """The ceiling catches tiny denominators, not large ratios as such. The
    sample company has 44 customers behind a $91 acquisition cost and a 20.3x
    ratio; refusing to answer on data that good is its own dishonesty."""
    result = fv.efficiency(ltv=1_848, cac=91, new_customers=44)
    assert result["band"] == "healthy"
    assert "assumes your churn stays" in result["detail"]


def test_a_plausible_ratio_still_reads_healthy():
    result = fv.efficiency(ltv=250, cac=50, new_customers=40)
    assert result["band"] == "healthy"
    assert "$50 to win" in result["detail"] and "$250" in result["detail"]


def test_too_few_customers_to_price_acquisition_from():
    assert fv.efficiency(ltv=250, cac=50, new_customers=3)["band"] == "unknown"


def test_unhealthy_is_named_in_words_not_a_ratio():
    result = fv.efficiency(ltv=250, cac=400, new_customers=40)
    assert result["label"] == "Costs more than it returns"


# --- churn ----------------------------------------------------------------

def test_churn_is_a_count_not_a_percentage():
    assert fv.churn_phrase(4.37) == "You lose about 1 in 23 customers a month"
    assert "%" not in fv.churn_phrase(4.37)


def test_churn_extremes_stay_readable():
    assert "more than half" in fv.churn_phrase(55)
    assert "fewer than 1 in 100" in fv.churn_phrase(0.4)


# --- rule of 40 -----------------------------------------------------------

@pytest.mark.parametrize("mrr,shown", [(2_500, False), (12_000, False),
                                       (83_333, True), (200_000, True)])
def test_rule_of_40_is_withheld_below_a_million_arr(mrr, shown):
    assert fv.rule_of_40_is_meaningful(mrr) is shown


def test_spend_ratio_is_the_replacement_and_works_at_any_size():
    assert fv.spend_ratio_phrase(8_500, 2_500) == "You spend $3.40 for every $1 of revenue"
    assert "no revenue yet" in fv.spend_ratio_phrase(500, 0)


# --- survival -------------------------------------------------------------

def test_survival_says_when_not_just_whether():
    phrase = fv.survival_phrase({"runs": 50, "deaths": 50, "median_death_month": 1})
    assert "Ran out of cash in every run" in phrase
    assert "month 1" in phrase


def test_survival_of_a_healthy_plan_says_so():
    assert "Still going" in fv.survival_phrase({"runs": 50, "deaths": 0})


# --- the two implementations share their numbers --------------------------

def test_thresholds_live_in_one_file_and_both_sides_read_it():
    """The client is local-first, so a few rules exist on both sides. The
    numbers must not: config/founder_view.json is the single source, and
    frontend/src/founderView.js has to reach through it rather than inline a
    second copy that can drift."""
    with io.open(os.path.join(ROOT, "config", "founder_view.json"), encoding="utf-8") as fh:
        raw = json.load(fh)
    keys = [k for k in raw if not k.startswith("_")]

    with io.open(os.path.join(ROOT, "frontend", "src", "founderView.js"), encoding="utf-8") as fh:
        js = fh.read()

    for key in keys:
        assert f"RULES.{key}" in js, f"founderView.js does not read RULES.{key}"

    # The floor is distinctive enough that finding it as a literal proves a copy.
    assert str(raw["rule_of_40_mrr_floor"]) not in js
