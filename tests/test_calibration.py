"""Tests for the sourced calibration store.

These guard the two things that would quietly corrupt founder-facing advice:
a wrong annual/monthly conversion, and an unsourced value passing as sourced.
"""

import pytest

import calibration as cal


# --- unit conversion -------------------------------------------------------

def test_annual_retention_converts_by_compounding_not_division():
    """54% annual retention is 5.01% monthly churn, not (100-54)/12 = 3.83%.

    The acquisition methodology singled this out as the single most likely
    failure in the pipeline, so it is pinned.
    """
    monthly = cal.annual_retention_to_monthly_churn(54)
    assert monthly == pytest.approx(0.0501, abs=0.0002)
    naive = (100 - 54) / 12 / 100
    assert monthly > naive  # compounding always exceeds the naive division


def test_full_retention_is_zero_churn():
    assert cal.annual_retention_to_monthly_churn(100) == pytest.approx(0.0)


def test_conversion_round_trips():
    """Surviving 12 months of the derived monthly churn reproduces the annual
    retention it came from."""
    for annual in (35, 54, 64, 82, 94):
        monthly = cal.annual_retention_to_monthly_churn(annual)
        assert (1 - monthly) ** 12 == pytest.approx(annual / 100, abs=1e-9)


# --- band boundaries -------------------------------------------------------

@pytest.mark.parametrize(
    "arpa,expected",
    [
        (0, "arpa_lt_25"), (24.99, "arpa_lt_25"),
        (25, "arpa_25_100"), (99.99, "arpa_25_100"),
        (100, "arpa_100_250"), (249.99, "arpa_100_250"),
        (250, "arpa_250_500"), (499.99, "arpa_250_500"),
        (500, "arpa_500_1000"), (999.99, "arpa_500_1000"),
        (1000, "arpa_gt_1000"), (50_000, "arpa_gt_1000"),
    ],
)
def test_band_boundaries_are_half_open(arpa, expected):
    assert cal.band_for_arpa(arpa) == expected


def test_every_band_is_reachable():
    reached = {cal.band_for_arpa(a) for a in (10, 40, 150, 300, 700, 2000)}
    assert reached == set(cal.ARPA_BANDS)


# --- sourced data integrity ------------------------------------------------

@pytest.mark.parametrize("kind", ["customer", "gross", "net"])
def test_all_bands_carry_observed_retention(kind):
    for arpa in (10, 40, 150, 300, 700, 2000):
        value = cal.annual_retention(arpa, kind=kind)
        assert value.is_observed, f"{kind} retention missing for ARPA {arpa}"
        assert value.publisher == "ChartMogul"
        assert value.page_or_figure


def test_churn_decreases_as_arpa_rises():
    """A transcription error in the source table would most likely show up as a
    broken monotonic trend, which the report's own narrative asserts."""
    churns = [cal.monthly_churn(a, kind="gross").value for a in (10, 40, 150, 300, 700, 2000)]
    assert churns == sorted(churns, reverse=True)


def test_churn_values_are_plausible_monthly_rates():
    for arpa in (10, 40, 150, 300, 700, 2000):
        churn = cal.monthly_churn(arpa, kind="gross").value
        assert 0.0 < churn < 0.15, "a monthly churn above 15% suggests an annual figure leaked through"


def test_derived_churn_carries_its_derivation():
    churn = cal.monthly_churn(40, kind="gross")
    assert "annual" in churn.page_or_figure
    assert churn.citation().startswith("ChartMogul")


# --- fails safe ------------------------------------------------------------

def test_unknown_metric_returns_none_not_a_default():
    assert cal.annual_retention(40, kind="nonexistent").value is None


def test_none_value_is_never_observed():
    absent = cal.annual_retention(40, kind="nonexistent")
    assert absent.is_observed is False


def test_discretionary_spend_requires_both_components():
    """Half a benchmark is not a benchmark."""
    spend = cal.discretionary_spend_pct_of_mrr()
    marketing = cal.department_spend_pct_of_arr("marketing")
    rnd = cal.department_spend_pct_of_arr("rnd")
    assert spend.is_observed == (marketing.is_observed and rnd.is_observed)
    if spend.is_observed:
        assert spend.value == pytest.approx(marketing.value + rnd.value)


def test_spend_band_flags_extrapolation():
    """Spend figures are printed for $3-5M ARR only; anything else is borrowed."""
    assert cal.spend_band_applies_to(4_000_000) is True
    assert cal.spend_band_applies_to(144_000) is False   # a $12k-MRR founder
    assert cal.spend_band_applies_to(50_000_000) is False


def test_withdrawn_table_stays_withdrawn():
    """The overall department medians failed PDF verification and must not
    reappear; they came from a single model-mediated read."""
    assert "by_department_pct_of_arr" not in cal.load().get("spend_benchmarks", {})


def test_unidentified_parameters_have_no_value():
    for parameter in ("price_elasticity", "marketing_saturation_absolute"):
        assert cal.is_unidentified(parameter)
        assert cal.unidentified_reason(parameter)
        assert cal.load()["unidentified"][parameter]["value"] is None
