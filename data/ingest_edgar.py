"""SEC EDGAR ingestion: a small, clean, fully-provenanced panel of SaaS filers.

Design rules, all of them non-negotiable and all of them enforced below:

1. **Structured API only.** `data.sec.gov/api/xbrl/companyfacts` and the official
   ticker->CIK map. No scraping, no HTML parsing. Descriptive User-Agent, <=10
   requests/second (SEC's published limit; we run at 8/s).
2. **Cache before parse.** Every raw API response lands on disk under
   `data/cache/edgar/` before anything reads it. Re-runs never touch the network.
   The database is therefore rebuildable offline in seconds.
3. **Never impute.** If no tag in the alias list matches, the value is NULL and
   the company-quarter drops out of any analysis needing that field. There is no
   "close enough" tag and no forward-fill. A missing number is a finding.
4. **Every row carries provenance.** No value exists in `facts` without the exact
   us-gaap tag it came from, the accession number of the filing, the form type,
   and the date we retrieved it.

Quality over quantity: the goal is 12-18 companies with complete, correctly
tagged data, not 40 companies half of which need patching. Tag mismatches are an
*exclusion criterion*, not a problem to solve.

Usage
-----
    python data/ingest_edgar.py            # fetch (cached) -> screen -> build
    python data/ingest_edgar.py --fetch    # populate the cache only
    python data/ingest_edgar.py --screen   # coverage report from cache
    python data/ingest_edgar.py --build    # build edgar.db from cache

Set SEC_USER_AGENT to a real contact string before first run; SEC requires a
declared identity and will throttle or block anonymous traffic:

    set SEC_USER_AGENT=Capstone Research (you@example.com)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache" / "edgar"
DB_PATH = DATA_DIR / "edgar.db"
COVERAGE_PATH = DATA_DIR / "coverage_report.md"
FACTS_CSV_PATH = DATA_DIR / "edgar_facts.csv"
RATIOS_CSV_PATH = DATA_DIR / "edgar_ratios.csv"

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# SEC publishes a 10 req/sec ceiling. We run under it deliberately.
REQUESTS_PER_SECOND = 8.0
_MIN_INTERVAL = 1.0 / REQUESTS_PER_SECOND

DEFAULT_USER_AGENT = "Capstone academic research (set SEC_USER_AGENT to override)"


# --------------------------------------------------------------------------
# Candidate universe
# --------------------------------------------------------------------------
# Public SaaS / subscription-software filers. Deliberately wider than the target
# panel so the inclusion criterion does the selecting rather than our priors.
CANDIDATE_TICKERS = [
    # collaboration / work management
    "ASAN", "MNDY", "SMAR", "ATLE", "NOTA",
    # marketing / customer engagement
    "BRZE", "KVYO", "SPT", "SEMR", "KLTR", "ZETA", "AMPL",
    # devtools / data infrastructure
    "GTLB", "CFLT", "DDOG", "MDB", "ESTC", "SUMO", "FSLY", "NET",
    # vertical & mid-market SaaS
    "OLO", "WEAV", "BIGC", "YEXT", "DOMO", "ZUO", "APPF", "PCTY", "PAYC",
    # security SaaS
    "S", "CRWD", "ZS", "OKTA", "TENB", "RPD", "QLYS",
    # CX / support / ops
    "FRSH", "ZM", "DOCN", "TWLO", "BAND", "EGHT", "RNG",
    # analytics / BI / misc subscription
    "AI", "PD", "BL", "WK", "ALRM", "CWAN", "INTA", "JAMF", "PATH",
]


# --------------------------------------------------------------------------
# Tag aliases. Order is priority order. NEVER extended with a "close enough" tag.
# --------------------------------------------------------------------------
CONCEPTS: dict[str, dict[str, Any]] = {
    "revenue": {
        "kind": "duration",
        "tags": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
        ],
    },
    "rnd_expense": {
        "kind": "duration",
        "tags": ["ResearchAndDevelopmentExpense"],
    },
    # The discriminator for the marketing-lever analysis. A filer that only
    # reports SG&A cannot have S&M recovered from XBRL at all -- see `sga_combined`.
    "sm_expense": {
        "kind": "duration",
        "tags": ["SellingAndMarketingExpense"],
    },
    "marketing_expense_narrow": {
        "kind": "duration",
        "tags": ["MarketingAndAdvertisingExpense", "AdvertisingExpense"],
    },
    "ga_expense": {
        "kind": "duration",
        "tags": ["GeneralAndAdministrativeExpense"],
    },
    "sga_combined": {
        "kind": "duration",
        "tags": ["SellingGeneralAndAdministrativeExpense"],
    },
    "cost_of_revenue": {
        "kind": "duration",
        "tags": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    },
    "operating_cash_flow": {
        "kind": "duration",
        "tags": ["NetCashProvidedByUsedInOperatingActivities"],
    },
    "cash": {
        "kind": "instant",
        "tags": [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ],
    },
    "short_term_investments": {
        "kind": "instant",
        "tags": ["ShortTermInvestments", "MarketableSecuritiesCurrent"],
    },
}

# A duration fact is one fiscal quarter if its span lands in this window. 10-Ks
# report 365-day spans and 10-Qs sometimes report cumulative six/nine-month
# spans; both are excluded here rather than being divided down into fake
# quarters.
QUARTER_MIN_DAYS = 80
QUARTER_MAX_DAYS = 100

ACCEPTED_FORMS = {"10-Q", "10-K", "10-K/A", "10-Q/A"}


# --------------------------------------------------------------------------
# Inclusion criterion -- declared in advance, applied mechanically
# --------------------------------------------------------------------------
MIN_CONSECUTIVE_QUARTERS = 16
REQUIRE_SM_SEPARATE = True

INCLUSION_CRITERION = f"""A candidate enters the panel if and only if all three hold:

1. It reports `SellingAndMarketingExpense` separately from G&A in at least one
   filing. Filers that only report `SellingGeneralAndAdministrativeExpense` are
   excluded: S&M is not recoverable from XBRL alone, and imputing it would be
   inventing the marketing lever we are trying to validate.
2. It uses one of the standard revenue tags.
3. It has at least {MIN_CONSECUTIVE_QUARTERS} consecutive quarters in which
   revenue, R&D and S&M are all present.

Nothing is imputed to satisfy these. A company failing any of them is dropped
and the reason recorded."""


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
_last_request_at = 0.0


def _user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)


def _throttled_get(url: str) -> bytes:
    """One GET, rate-limited below SEC's published ceiling."""
    global _last_request_at
    wait = _MIN_INTERVAL - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _user_agent(),
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
            if response.headers.get("Content-Encoding") == "gzip":
                import gzip

                payload = gzip.decompress(payload)
            return payload
    finally:
        _last_request_at = time.monotonic()


def _cached_json(url: str, cache_path: Path, force: bool = False) -> dict[str, Any] | None:
    """Fetch to cache, then parse from cache. The cache is the source of truth."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        try:
            with cache_path.open(encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            print(f"  ! corrupt cache {cache_path.name}, refetching")

    try:
        raw = _throttled_get(url)
    except urllib.error.HTTPError as exc:
        print(f"  ! HTTP {exc.code} for {url}")
        return None
    except Exception as exc:  # noqa: BLE001 - network layer, report and continue
        print(f"  ! {type(exc).__name__}: {exc}")
        return None

    cache_path.write_bytes(raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        print(f"  ! non-JSON response for {url}")
        cache_path.unlink(missing_ok=True)
        return None


# --------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------
def load_ticker_map(force: bool = False) -> dict[str, dict[str, Any]]:
    payload = _cached_json(TICKER_MAP_URL, CACHE_DIR / "company_tickers.json", force=force)
    if not payload:
        raise SystemExit("Could not obtain the SEC ticker->CIK map; cannot continue.")
    mapping: dict[str, dict[str, Any]] = {}
    for entry in payload.values():
        mapping[str(entry["ticker"]).upper()] = {
            "cik": int(entry["cik_str"]),
            "title": entry.get("title", ""),
        }
    return mapping


def fetch_all(tickers: Iterable[str], force: bool = False) -> dict[str, dict[str, Any]]:
    """Populate the cache. Returns {ticker: {cik, title, path}} for what landed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ticker_map = load_ticker_map(force=force)

    resolved: dict[str, dict[str, Any]] = {}
    tickers = list(tickers)
    print(f"Fetching companyfacts for {len(tickers)} candidates "
          f"(<={REQUESTS_PER_SECOND:.0f} req/s, cache-first)...")

    for index, ticker in enumerate(tickers, start=1):
        ticker = ticker.upper()
        entry = ticker_map.get(ticker)
        if entry is None:
            print(f"  [{index:>2}/{len(tickers)}] {ticker:<6} - not in SEC ticker map, skipped")
            continue

        cik = entry["cik"]
        cache_path = CACHE_DIR / f"CIK{cik:010d}.json"
        cached_before = cache_path.exists()
        payload = _cached_json(COMPANYFACTS_URL.format(cik=cik), cache_path, force=force)
        if payload is None:
            print(f"  [{index:>2}/{len(tickers)}] {ticker:<6} - companyfacts unavailable, skipped")
            continue

        size_mb = cache_path.stat().st_size / 1_048_576
        marker = "cached" if cached_before and not force else "fetched"
        print(f"  [{index:>2}/{len(tickers)}] {ticker:<6} CIK {cik:>10}  {size_mb:5.1f} MB  {marker}")
        resolved[ticker] = {"cik": cik, "title": entry["title"], "path": cache_path}

    print(f"Cache holds {len(resolved)} companies at {CACHE_DIR}")
    return resolved


# --------------------------------------------------------------------------
# Parse
# --------------------------------------------------------------------------
def _calendar_quarter(end_iso: str) -> str:
    end = date.fromisoformat(end_iso)
    return f"{end.year}Q{(end.month - 1) // 3 + 1}"


def _iter_facts(payload: dict[str, Any], tag: str) -> Iterable[dict[str, Any]]:
    node = ((payload.get("facts") or {}).get("us-gaap") or {}).get(tag)
    if not node:
        return
    for unit, entries in (node.get("units") or {}).items():
        if unit != "USD":
            continue
        for entry in entries:
            yield entry


def _span_days(start: str, end: str) -> int:
    return (date.fromisoformat(end) - date.fromisoformat(start)).days


def _dedupe_duration_entries(
    payload: dict[str, Any], tags: list[str]
) -> dict[tuple[str, str], dict[str, Any]]:
    """All duration facts for one concept, keyed on (start, end).

    A period restated across filings appears many times. Priority is: an earlier
    tag in the alias list wins outright; within the same tag, the later filing
    wins. So the store reflects the company's latest word on each period, and
    records which filing that was.
    """
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for rank, tag in enumerate(tags):
        for entry in _iter_facts(payload, tag):
            if entry.get("form") not in ACCEPTED_FORMS:
                continue
            start, end = entry.get("start"), entry.get("end")
            if not start or not end or entry.get("val") is None:
                continue
            key = (start, end)
            existing = best.get(key)
            if existing is not None:
                if rank > existing["_rank"]:
                    continue
                if rank == existing["_rank"] and (entry.get("filed") or "") <= existing["filed"]:
                    continue
            best[key] = {
                "tag": tag,
                "_rank": rank,
                "value": float(entry["val"]),
                "period_start": start,
                "period_end": end,
                "fiscal_year": entry.get("fy"),
                "fiscal_period": entry.get("fp"),
                "form": entry.get("form"),
                "accession": entry.get("accn"),
                "filed": entry.get("filed") or "",
            }
    return best


def _quarterize(entries: dict[tuple[str, str], dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """Reduce a concept's duration facts to single fiscal quarters.

    This is the crux of quarterly XBRL. Filers do NOT report Q4 as a three-month
    figure: 10-Qs carry Q1 plus cumulative six- and nine-month spans, and the
    10-K carries only the twelve-month year. Taking three-month facts alone
    therefore yields Q1-Q2-Q3 and a permanent gap at Q4, which breaks every
    consecutive run at three quarters.

    Two sources, in priority order:

      1. **As filed.** Any fact whose own span is one quarter is used verbatim.
      2. **YTD differenced.** Facts sharing a fiscal-year start are cumulative,
         so quarter_n = YTD(n) - YTD(n-1). This is exact arithmetic on two
         reported figures, not imputation: nothing is estimated, interpolated or
         filled from a neighbour, and the result is flagged with `derivation`
         plus the accessions of both operands so a reader can recompute it.

    An as-filed value always beats a derived one for the same period.
    """
    as_filed: dict[tuple[str, str], dict[str, Any]] = {}
    for (start, end), entry in entries.items():
        if QUARTER_MIN_DAYS <= _span_days(start, end) <= QUARTER_MAX_DAYS:
            as_filed[(start, end)] = {**entry, "derivation": None, "derived_from": None}

    derived: dict[tuple[str, str], dict[str, Any]] = {}
    by_start: dict[str, list[dict[str, Any]]] = {}
    for entry in entries.values():
        by_start.setdefault(entry["period_start"], []).append(entry)

    for fy_start, group in by_start.items():
        # One cumulative chain per start date, shortest span first.
        chain = sorted(group, key=lambda e: e["period_end"])
        previous: dict[str, Any] | None = None
        for entry in chain:
            span = _span_days(entry["period_start"], entry["period_end"])
            if span < QUARTER_MIN_DAYS:
                continue
            window_start = previous["period_end"] if previous else fy_start
            quarter_span = _span_days(window_start, entry["period_end"])
            key = (window_start, entry["period_end"])
            if (
                QUARTER_MIN_DAYS <= quarter_span <= QUARTER_MAX_DAYS
                and key not in as_filed
                and key not in derived
                and previous is not None  # a bare quarter with no predecessor is already as-filed
            ):
                derived[key] = {
                    **entry,
                    "value": entry["value"] - previous["value"],
                    "period_start": window_start,
                    "derivation": (
                        f"cumulative {fy_start}..{entry['period_end']} minus "
                        f"cumulative {fy_start}..{previous['period_end']}"
                    ),
                    "derived_from": f"{entry['accession']}|{previous['accession']}",
                }
            previous = entry

    return {**derived, **as_filed}


def extract_company_facts(payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """companyfacts JSON -> {(calendar_quarter, concept): row}."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    for concept, spec in CONCEPTS.items():
        if spec["kind"] == "instant":
            # Balance-sheet items are reported at every period end, Q4 included,
            # so they need no reconstruction.
            for rank, tag in enumerate(spec["tags"]):
                for entry in _iter_facts(payload, tag):
                    if entry.get("form") not in ACCEPTED_FORMS:
                        continue
                    end = entry.get("end")
                    if not end or entry.get("val") is None:
                        continue
                    key = (_calendar_quarter(end), concept)
                    existing = rows.get(key)
                    filed = entry.get("filed") or ""
                    if existing is not None:
                        if rank > existing["_rank"]:
                            continue
                        if rank == existing["_rank"] and filed <= existing["filed"]:
                            continue
                    rows[key] = {
                        "quarter": key[0], "concept": concept, "tag": tag, "_rank": rank,
                        "value": float(entry["val"]), "unit": "USD",
                        "period_start": None, "period_end": end,
                        "fiscal_year": entry.get("fy"), "fiscal_period": entry.get("fp"),
                        "form": entry.get("form"), "accession": entry.get("accn"),
                        "filed": filed, "derivation": None, "derived_from": None,
                    }
            continue

        quarters = _quarterize(_dedupe_duration_entries(payload, spec["tags"]))
        for (start, end), entry in quarters.items():
            key = (_calendar_quarter(end), concept)
            existing = rows.get(key)
            # If two fiscal quarters land in the same calendar quarter, keep the
            # as-filed one, then the later filing.
            if existing is not None:
                if existing["derivation"] is None and entry["derivation"] is not None:
                    continue
                if entry["filed"] <= existing["filed"] and existing["derivation"] == entry["derivation"]:
                    continue
            rows[key] = {
                "quarter": key[0], "concept": concept, "tag": entry["tag"],
                "_rank": entry["_rank"], "value": entry["value"], "unit": "USD",
                "period_start": start, "period_end": end,
                "fiscal_year": entry["fiscal_year"], "fiscal_period": entry["fiscal_period"],
                "form": entry["form"], "accession": entry["accession"],
                "filed": entry["filed"], "derivation": entry["derivation"],
                "derived_from": entry["derived_from"],
            }

    return rows


def _longest_consecutive_run(quarters: Iterable[str]) -> tuple[int, str | None, str | None]:
    """Longest run of calendar-consecutive quarters. Returns (length, first, last)."""
    def as_index(q: str) -> int:
        year, quarter = q.split("Q")
        return int(year) * 4 + int(quarter) - 1

    ordered = sorted(set(quarters), key=as_index)
    if not ordered:
        return 0, None, None

    best_len, best_start, best_end = 1, ordered[0], ordered[0]
    run_len, run_start = 1, ordered[0]
    for previous, current in zip(ordered, ordered[1:]):
        if as_index(current) == as_index(previous) + 1:
            run_len += 1
        else:
            run_len, run_start = 1, current
        if run_len > best_len:
            best_len, best_start, best_end = run_len, run_start, current
    return best_len, best_start, best_end


def screen_company(ticker: str, meta: dict[str, Any]) -> dict[str, Any]:
    """Score one candidate against the declared inclusion criterion."""
    with meta["path"].open(encoding="utf-8") as handle:
        payload = json.load(handle)

    facts = extract_company_facts(payload)
    by_concept: dict[str, set[str]] = {}
    tags_used: dict[str, set[str]] = {}
    for (quarter, concept), row in facts.items():
        by_concept.setdefault(concept, set()).add(quarter)
        tags_used.setdefault(concept, set()).add(row["tag"])

    has_sm = bool(by_concept.get("sm_expense"))
    has_sga_only = bool(by_concept.get("sga_combined")) and not has_sm
    has_revenue = bool(by_concept.get("revenue"))

    core = ["revenue", "rnd_expense", "sm_expense"]
    complete = set.intersection(*[by_concept.get(c, set()) for c in core]) if all(
        by_concept.get(c) for c in core
    ) else set()
    run_len, run_start, run_end = _longest_consecutive_run(complete)

    reasons: list[str] = []
    if REQUIRE_SM_SEPARATE and not has_sm:
        reasons.append(
            "reports SG&A combined only; S&M not separable from XBRL"
            if has_sga_only else "no SellingAndMarketingExpense tag"
        )
    if not has_revenue:
        reasons.append("no standard revenue tag")
    if run_len < MIN_CONSECUTIVE_QUARTERS:
        reasons.append(f"longest complete run {run_len}q < {MIN_CONSECUTIVE_QUARTERS}q")

    return {
        "ticker": ticker,
        "cik": meta["cik"],
        "title": meta["title"],
        "included": not reasons,
        "exclusion_reasons": reasons,
        "has_sm_separate": has_sm,
        "sga_combined_only": has_sga_only,
        "revenue_tag": sorted(tags_used.get("revenue", {"-"}))[0] if has_revenue else None,
        "complete_quarters": len(complete),
        "consecutive_quarters": run_len,
        "run_start": run_start,
        "run_end": run_end,
        "total_facts": len(facts),
        "has_cash": bool(by_concept.get("cash")),
        "has_ocf": bool(by_concept.get("operating_cash_flow")),
        "has_cogs": bool(by_concept.get("cost_of_revenue")),
    }


# --------------------------------------------------------------------------
# Coverage report
# --------------------------------------------------------------------------
def write_coverage_report(scores: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
    ranked = sorted(
        scores,
        key=lambda s: (s["included"], s["consecutive_quarters"], s["complete_quarters"]),
        reverse=True,
    )
    included = [s for s in ranked if s["included"]]
    excluded = [s for s in ranked if not s["included"]]

    lines: list[str] = []
    lines.append("# EDGAR coverage report\n")
    lines.append(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from cached "
        f"`companyfacts` responses in `data/cache/edgar/`. No network access is required to "
        f"regenerate this file.\n"
    )
    lines.append(
        "Nothing is imputed, interpolated or forward-filled; a company-quarter missing any "
        "core field simply does not count toward its run length.\n"
    )
    lines.append(
        "**On Q4.** Filers do not report Q4 as a three-month figure — 10-Qs carry Q1 plus "
        "cumulative six- and nine-month spans, and the 10-K carries only the twelve-month "
        "year. Reading three-month facts alone yields Q1–Q2–Q3 and a permanent gap at Q4, "
        "capping every consecutive run at three quarters. Those quarters are recovered by "
        "differencing two reported year-to-date figures from the same fiscal year: exact "
        "arithmetic on filed numbers, not estimation. Each such row carries its `derivation` "
        "and the accessions of both operands, so `SELECT ... WHERE derivation IS NULL` "
        "restricts any analysis to strictly as-filed values.\n"
    )

    lines.append("## Inclusion criterion (declared before screening)\n")
    lines.append(INCLUSION_CRITERION + "\n")

    lines.append("## Screening outcome\n")
    lines.append(
        f"- Candidates screened: **{len(scores)}**\n"
        f"- Included: **{len(included)}**\n"
        f"- Excluded: **{len(excluded)}**\n"
    )

    lines.append("\n## Panel (included)\n")
    lines.append("| Ticker | Company | CIK | S&M separate | Revenue tag | Consecutive q | Span | Complete q |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s in included:
        span = f"{s['run_start']}–{s['run_end']}" if s["run_start"] else "—"
        lines.append(
            f"| {s['ticker']} | {s['title'][:34]} | {s['cik']} | "
            f"{'yes' if s['has_sm_separate'] else 'no'} | `{s['revenue_tag']}` | "
            f"{s['consecutive_quarters']} | {span} | {s['complete_quarters']} |"
        )

    lines.append("\n## Excluded, with reasons\n")
    lines.append("| Ticker | Company | Consecutive q | Reason |")
    lines.append("|---|---|---|---|")
    for s in excluded:
        lines.append(
            f"| {s['ticker']} | {s['title'][:34]} | {s['consecutive_quarters']} | "
            f"{'; '.join(s['exclusion_reasons'])} |"
        )

    sga_only = [s for s in excluded if s["sga_combined_only"]]
    lines.append(
        f"\n### Note on the S&M / SG&A split\n\n"
        f"{len(sga_only)} of {len(scores)} candidates report `SellingGeneralAndAdministrativeExpense` "
        f"without a separate `SellingAndMarketingExpense`. This is the Tier-2 tag mismatch the "
        f"acquisition plan anticipated. It is resolved by exclusion, not by a better tag map: "
        f"marketing spend genuinely cannot be recovered from those filings, and any split we "
        f"invented would be our assumption wearing an audited company's name.\n"
    )

    lines.append(
        "\n## Scope statement for the review\n\n"
        f"Agent validation on this panel covers R&D fully and marketing spend for the "
        f"**{len(included)} of {len(scores)}** filers that separate S&M from G&A. Hiring is "
        f"available at annual granularity only (headcount is 10-K cover-page prose, not XBRL). "
        f"Pricing and channel are not disclosed by any filer and are not validatable here — "
        f"they are validated against private benchmarks instead, or declared unidentified.\n"
    )

    COVERAGE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {COVERAGE_PATH}")
    return included, excluded


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------
SCHEMA = """
DROP VIEW IF EXISTS ratios;
DROP TABLE IF EXISTS facts;
DROP TABLE IF EXISTS companies;
DROP TABLE IF EXISTS ingest_meta;

CREATE TABLE companies (
    cik              INTEGER PRIMARY KEY,
    ticker           TEXT NOT NULL UNIQUE,
    name             TEXT NOT NULL,
    included         INTEGER NOT NULL,
    exclusion_reason TEXT,
    consecutive_q    INTEGER,
    run_start        TEXT,
    run_end          TEXT
);

-- One row per (company, quarter, concept). `tag` and `accession` are NOT NULL:
-- a value cannot exist in this table without saying where it came from.
CREATE TABLE facts (
    cik            INTEGER NOT NULL,
    ticker         TEXT    NOT NULL,
    fiscal_period  TEXT    NOT NULL,   -- calendar quarter, e.g. 2023Q4
    concept        TEXT    NOT NULL,   -- our normalised name
    tag            TEXT    NOT NULL,   -- the exact us-gaap tag as filed
    value          REAL,
    unit           TEXT    NOT NULL,
    period_start   TEXT,
    period_end     TEXT    NOT NULL,
    fiscal_year    INTEGER,
    fiscal_quarter TEXT,
    form           TEXT    NOT NULL,
    accession      TEXT    NOT NULL,
    filed          TEXT,
    -- NULL for a value taken verbatim from a filing. Otherwise the exact
    -- arithmetic that produced it, with `derived_from` listing the accessions
    -- of both operands so any reader can recompute the number.
    derivation     TEXT,
    derived_from   TEXT,
    retrieved_at   TEXT    NOT NULL,
    PRIMARY KEY (cik, fiscal_period, concept),
    FOREIGN KEY (cik) REFERENCES companies(cik)
);

CREATE INDEX idx_facts_concept ON facts(concept);
CREATE INDEX idx_facts_period  ON facts(fiscal_period);
CREATE INDEX idx_facts_ticker  ON facts(ticker);

CREATE TABLE ingest_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Scale-free ratios. Every comparison against the simulator happens here, never
# in absolute dollars: the panel sits at $100M+ ARR and the simulator at $50k MRR,
# so levels are not comparable and ratios are.
RATIOS_VIEW = """
CREATE VIEW ratios AS
WITH pivoted AS (
    SELECT
        f.cik,
        f.ticker,
        f.fiscal_period,
        MAX(CASE WHEN f.concept = 'revenue'             THEN f.value END) AS revenue,
        MAX(CASE WHEN f.concept = 'rnd_expense'         THEN f.value END) AS rnd,
        MAX(CASE WHEN f.concept = 'sm_expense'          THEN f.value END) AS sm,
        MAX(CASE WHEN f.concept = 'cost_of_revenue'     THEN f.value END) AS cogs,
        MAX(CASE WHEN f.concept = 'operating_cash_flow' THEN f.value END) AS ocf,
        MAX(CASE WHEN f.concept = 'cash'                THEN f.value END) AS cash,
        MAX(CASE WHEN f.concept = 'short_term_investments' THEN f.value END) AS sti
    FROM facts f
    GROUP BY f.cik, f.ticker, f.fiscal_period
),
sequenced AS (
    SELECT
        p.*,
        LAG(p.revenue) OVER (PARTITION BY p.cik ORDER BY p.fiscal_period) AS prev_revenue
    FROM pivoted p
)
SELECT
    cik,
    ticker,
    fiscal_period,
    revenue,
    prev_revenue,
    CASE WHEN prev_revenue > 0
         THEN (revenue - prev_revenue) / prev_revenue END              AS qoq_growth,
    CASE WHEN revenue > 0 THEN sm   / revenue END                      AS sm_pct_revenue,
    CASE WHEN revenue > 0 THEN rnd  / revenue END                      AS rnd_pct_revenue,
    CASE WHEN revenue > 0 AND cogs IS NOT NULL
         THEN 1.0 - (cogs / revenue) END                               AS gross_margin,
    -- Magic number: new ARR won per dollar of prior-quarter S&M.
    CASE WHEN prev_revenue > 0 AND sm > 0
         THEN ((revenue - prev_revenue) * 4.0) / sm END                AS magic_number,
    -- Burn multiple: net cash burned per dollar of new ARR. Negative OCF = burn.
    CASE WHEN (revenue - prev_revenue) > 0 AND ocf < 0
         THEN (-ocf) / ((revenue - prev_revenue) * 4.0) END            AS burn_multiple,
    CASE WHEN prev_revenue > 0 AND revenue > 0 AND cogs IS NOT NULL AND ocf IS NOT NULL
         THEN (((revenue - prev_revenue) / prev_revenue) * 100.0)
              + ((ocf / revenue) * 100.0) END                          AS rule_of_40,
    COALESCE(cash, 0) + COALESCE(sti, 0)                               AS cash_and_investments
FROM sequenced;
"""


def build_database(resolved: dict[str, dict[str, Any]], scores: list[dict[str, Any]]) -> dict[str, int]:
    retrieved_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    by_ticker = {s["ticker"]: s for s in scores}

    DB_PATH.unlink(missing_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.executescript(SCHEMA)

    included_tickers = [s["ticker"] for s in scores if s["included"]]
    fact_rows = 0

    for score in scores:
        connection.execute(
            """INSERT INTO companies
               (cik, ticker, name, included, exclusion_reason, consecutive_q, run_start, run_end)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                score["cik"], score["ticker"], score["title"],
                1 if score["included"] else 0,
                "; ".join(score["exclusion_reasons"]) or None,
                score["consecutive_quarters"], score["run_start"], score["run_end"],
            ),
        )

    # Facts are stored for the included panel only. Excluded companies keep a
    # row in `companies` carrying the reason, so the exclusions stay auditable
    # without polluting the analysis tables.
    for ticker in included_tickers:
        meta = resolved[ticker]
        with meta["path"].open(encoding="utf-8") as handle:
            payload = json.load(handle)
        for (quarter, concept), row in extract_company_facts(payload).items():
            if row["value"] is None or not row["accession"]:
                continue  # never store a value without provenance
            connection.execute(
                """INSERT OR REPLACE INTO facts
                   (cik, ticker, fiscal_period, concept, tag, value, unit, period_start,
                    period_end, fiscal_year, fiscal_quarter, form, accession, filed,
                    derivation, derived_from, retrieved_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    meta["cik"], ticker, quarter, concept, row["tag"], row["value"], row["unit"],
                    row["period_start"], row["period_end"], row["fiscal_year"],
                    row["fiscal_period"], row["form"], row["accession"], row["filed"],
                    row["derivation"], row["derived_from"], retrieved_at,
                ),
            )
            fact_rows += 1

    connection.executescript(RATIOS_VIEW)

    for key, value in {
        "retrieved_at": retrieved_at,
        "source": "SEC EDGAR XBRL companyfacts API",
        "user_agent": _user_agent(),
        "candidates_screened": str(len(scores)),
        "companies_included": str(len(included_tickers)),
        "inclusion_criterion": INCLUSION_CRITERION,
        "min_consecutive_quarters": str(MIN_CONSECUTIVE_QUARTERS),
        "imputation": "none; missing tags stored as absent rows, never filled",
        "derivation_policy": (
            "Quarters not filed as three-month figures (Q4 always, and any quarter a filer "
            "reports only cumulatively) are recovered by differencing two reported "
            "year-to-date values from the same fiscal year. Exact arithmetic on filed "
            "numbers, never interpolation. Every such row carries `derivation` and "
            "`derived_from`; filter on `derivation IS NULL` for as-filed values only."
        ),
    }.items():
        connection.execute("INSERT INTO ingest_meta (key, value) VALUES (?,?)", (key, value))

    connection.commit()

    ratio_rows = connection.execute("SELECT COUNT(*) FROM ratios").fetchone()[0]
    _export_csv(connection, "SELECT * FROM facts ORDER BY ticker, fiscal_period, concept", FACTS_CSV_PATH)
    _export_csv(connection, "SELECT * FROM ratios ORDER BY ticker, fiscal_period", RATIOS_CSV_PATH)
    connection.close()

    print(f"Wrote {DB_PATH}  ({fact_rows} facts, {ratio_rows} company-quarters, "
          f"{len(included_tickers)} companies)")
    return {"facts": fact_rows, "ratios": ratio_rows, "companies": len(included_tickers)}


def _export_csv(connection: sqlite3.Connection, query: str, path: Path) -> None:
    cursor = connection.execute(query)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([d[0] for d in cursor.description])
        writer.writerows(cursor.fetchall())
    print(f"Wrote {path}")


# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="populate the cache only")
    parser.add_argument("--screen", action="store_true", help="coverage report from cache")
    parser.add_argument("--build", action="store_true", help="build edgar.db from cache")
    parser.add_argument("--force", action="store_true", help="ignore cache, refetch")
    args = parser.parse_args(argv)

    run_all = not (args.fetch or args.screen or args.build)

    if _user_agent() == DEFAULT_USER_AGENT:
        print("! SEC_USER_AGENT is unset. SEC asks for a declared contact; set it before\n"
              "  any sizeable run:  set SEC_USER_AGENT=Your Project (you@example.com)\n")

    resolved: dict[str, dict[str, Any]] = {}
    if run_all or args.fetch:
        resolved = fetch_all(CANDIDATE_TICKERS, force=args.force)

    if not resolved:
        # Rebuild the resolved map from whatever the cache already holds.
        ticker_map = load_ticker_map()
        cik_to_ticker = {v["cik"]: (k, v["title"]) for k, v in ticker_map.items()}
        for path in sorted(CACHE_DIR.glob("CIK*.json")):
            cik = int(path.stem.replace("CIK", ""))
            if cik in cik_to_ticker:
                ticker, title = cik_to_ticker[cik]
                resolved[ticker] = {"cik": cik, "title": title, "path": path}

    if args.fetch and not (args.screen or args.build):
        return 0

    if not resolved:
        print("Cache is empty; run with --fetch first.")
        return 1

    print(f"\nScreening {len(resolved)} cached companies...")
    scores = []
    for ticker, meta in sorted(resolved.items()):
        try:
            scores.append(screen_company(ticker, meta))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the panel
            print(f"  ! {ticker}: {type(exc).__name__}: {exc}")

    included, excluded = write_coverage_report(scores)
    print(f"  included {len(included)} / screened {len(scores)}")

    if run_all or args.build:
        build_database(resolved, scores)

    return 0


if __name__ == "__main__":
    sys.exit(main())
