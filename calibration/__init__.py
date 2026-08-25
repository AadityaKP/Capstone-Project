"""Sourced calibration constants, replacing hand-tuned literals in the engine.

Two rules the rest of the codebase depends on:

1. A value is either printed in a cited source or it is None. There is no silent
   default. Callers must handle None by degrading honestly - the whole point is
   that an uncalibrated parameter is visible as uncalibrated rather than
   masquerading as a measurement.
2. Every value carries its confidence and its citation. "observed" means printed
   for that exact band; anything else must be rendered differently in the UI,
   the same rule already applied to MAY_CAUSE edges in the causal graph.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

BANDS_PATH = Path(__file__).resolve().parent / "bands.json"

# Follows ChartMogul's own segmentation, which splits $250-500 and $500-1k.
ARPA_BANDS = (
    "arpa_lt_25", "arpa_25_100", "arpa_100_250",
    "arpa_250_500", "arpa_500_1000", "arpa_gt_1000",
)


@dataclass(frozen=True)
class Calibrated:
    """One calibration value plus the provenance needed to render it honestly."""

    value: float | None
    confidence: str
    publisher: str | None = None
    report: str | None = None
    year: int | None = None
    page_or_figure: str | None = None
    n: int | None = None

    @property
    def is_observed(self) -> bool:
        return self.value is not None and self.confidence == "observed"

    def citation(self) -> str | None:
        if self.publisher is None:
            return None
        parts = [self.publisher]
        if self.report:
            parts.append(self.report)
        if self.year:
            parts.append(str(self.year))
        return ", ".join(parts)


@lru_cache(maxsize=1)
def load() -> dict[str, Any]:
    with BANDS_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def band_for_arpa(arpa: float) -> str:
    """ARPA band id. ARPA is the axis the published churn data is segmented on,
    which is why bands are keyed on it rather than on company size."""
    if arpa < 25:
        return "arpa_lt_25"
    if arpa < 100:
        return "arpa_25_100"
    if arpa < 250:
        return "arpa_100_250"
    if arpa < 500:
        return "arpa_250_500"
    if arpa < 1000:
        return "arpa_500_1000"
    return "arpa_gt_1000"


def annual_retention_to_monthly_churn(annual_retention_pct: float) -> float:
    """Annual retention % -> monthly churn fraction.

    The one place this conversion happens. Sources print ANNUAL retention;
    the engine reasons in MONTHLY churn, and conflating the two is the most
    likely way for a wrong number to reach a founder. Compounding, not
    division: 54% annual retention is 5.0% monthly churn, not 46/12 = 3.8%.
    """
    retention = max(1e-6, min(1.0, annual_retention_pct / 100.0))
    return 1.0 - retention ** (1.0 / 12.0)


def _wrap(node: dict[str, Any] | None, value: Any, confidence: str | None = None) -> Calibrated:
    source = (node or {}).get("source") or {}
    return Calibrated(
        value=value,
        confidence=confidence or (node or {}).get("confidence") or "assumed",
        publisher=source.get("publisher"),
        report=source.get("report"),
        year=source.get("year"),
        page_or_figure=source.get("page_or_figure"),
        n=source.get("n"),
    )


def _arpa_band(arpa: float) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    node = load().get("arpa_bands", {})
    band_id = band_for_arpa(arpa)
    for band in node.get("bands", []):
        if band.get("band_id") == band_id:
            return band, node
    return None, node


def annual_retention(arpa: float, kind: str = "customer",
                     percentile: str = "p50") -> Calibrated:
    """Annual retention % as printed. kind: customer | gross | net."""
    band, node = _arpa_band(arpa)
    if band is None:
        return Calibrated(value=None, confidence="assumed")
    raw = band.get(f"annual_{kind}_retention_pct") or {}
    return _wrap(node, raw.get(percentile), band.get("confidence"))


def monthly_churn(arpa: float, kind: str = "customer",
                  percentile: str = "p50") -> Calibrated:
    """Monthly churn fraction derived from published annual retention.

    kind: 'customer' (logo churn), 'gross' (gross MRR churn), 'net' (net MRR
    churn - can be negative when expansion exceeds churn, which is why it is
    never used as a churn input to the engine).
    """
    source = annual_retention(arpa, kind=kind, percentile=percentile)
    if source.value is None:
        return Calibrated(value=None, confidence="assumed")
    return Calibrated(
        value=annual_retention_to_monthly_churn(float(source.value)),
        confidence=source.confidence,
        publisher=source.publisher,
        report=source.report,
        year=source.year,
        page_or_figure=f"{source.page_or_figure} (annual {source.value}% -> monthly)",
        n=source.n,
    )


# The only spend breakdown that survived verification is the $3-5M ARR band.
SPEND_BAND_ARR_MIN = 3_000_000
SPEND_BAND_ARR_MAX = 5_000_000


def department_spend_pct_of_arr(department: str) -> Calibrated:
    """Median spend for one department as a percent of ARR.

    Percent-of-ARR is also percent-of-MRR for a monthly figure: annual spend of
    0.08 * ARR is 0.08 * 12 * MRR, so monthly spend is 0.08 * MRR.

    Sourced from the $3-5M ARR band - the only breakdown printed as text in the
    report. Callers applying it to a company outside that band are extrapolating
    and must say so; see spend_band_applies_to().
    """
    bands = load().get("spend_benchmarks", {}).get("by_arr_band", [])
    if not bands:
        return Calibrated(value=None, confidence="assumed")
    node = bands[0]
    raw = node.get(f"{department}_pct_of_arr")
    value = raw.get("p50") if isinstance(raw, dict) else raw
    return _wrap(node, value)


def spend_band_applies_to(arr: float) -> bool:
    """True when a company sits inside the ARR band the spend figures were
    printed for. False means any use of them is extrapolation."""
    return SPEND_BAND_ARR_MIN <= arr <= SPEND_BAND_ARR_MAX


def total_spend_pct_of_arr(funding: str = "bootstrapped") -> Calibrated:
    node = load().get("spend_benchmarks", {}).get("total_spend_pct_of_arr", {})
    raw = node.get(funding)
    value = raw.get("p50") if isinstance(raw, dict) else raw
    return _wrap(node, value)


def discretionary_spend_pct_of_mrr() -> Calibrated:
    """Median marketing + R&D spend, as a share of monthly revenue.

    This is the benchmark a plan's discretionary spend should be judged against.
    Both components must be observed or the whole figure is withheld: half a
    benchmark is not a benchmark.
    """
    marketing = department_spend_pct_of_arr("marketing")
    rnd = department_spend_pct_of_arr("rnd")
    if not (marketing.is_observed and rnd.is_observed):
        return Calibrated(value=None, confidence="assumed")
    return Calibrated(
        value=float(marketing.value) + float(rnd.value),
        confidence="observed",
        publisher=marketing.publisher,
        report=marketing.report,
        year=marketing.year,
        page_or_figure=f"marketing + R&D, {marketing.page_or_figure}",
        n=marketing.n,
    )


def is_unidentified(parameter: str) -> bool:
    """True for parameters no public dataset can currently fix (price elasticity,
    absolute marketing saturation). Callers should surface these as unidentified
    rather than reporting a point estimate."""
    return parameter in (load().get("unidentified") or {})


def unidentified_reason(parameter: str) -> str | None:
    return ((load().get("unidentified") or {}).get(parameter) or {}).get("reason")
