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

ARPA_BANDS = ("arpa_lt_25", "arpa_25_100", "arpa_100_250", "arpa_250_1000", "arpa_gt_1000")


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
    if arpa < 1000:
        return "arpa_250_1000"
    return "arpa_gt_1000"


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


def band_metric(arpa: float, metric: str, percentile: str = "p50") -> Calibrated:
    """One banded metric, e.g. band_metric(40, 'monthly_gross_mrr_churn').

    Returns Calibrated(value=None) until the band is filled, which is the
    expected state for everything except the spend benchmarks today.
    """
    band_id = band_for_arpa(arpa)
    for band in load().get("bands", []):
        if band.get("band_id") == band_id:
            raw = band.get(metric)
            value = raw.get(percentile) if isinstance(raw, dict) else raw
            return _wrap(band, value)
    return Calibrated(value=None, confidence="assumed")


def department_spend_pct_of_arr(department: str) -> Calibrated:
    """Median spend for one department as a percent of ARR.

    Percent-of-ARR is also percent-of-MRR for a monthly figure: annual spend of
    0.08 * ARR is 0.08 * 12 * MRR, so monthly spend is 0.08 * MRR.
    """
    node = load().get("spend_benchmarks", {}).get("by_department_pct_of_arr", {})
    raw = node.get(department)
    value = raw.get("p50") if isinstance(raw, dict) else raw
    return _wrap(node, value)


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
        page_or_figure="marketing + R&D, Private SaaS Company Spending by Department",
        n=marketing.n,
    )


def is_unidentified(parameter: str) -> bool:
    """True for parameters no public dataset can currently fix (price elasticity,
    absolute marketing saturation). Callers should surface these as unidentified
    rather than reporting a point estimate."""
    return parameter in (load().get("unidentified") or {})


def unidentified_reason(parameter: str) -> str | None:
    return ((load().get("unidentified") or {}).get(parameter) or {}).get("reason")
