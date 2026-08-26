"""Per-company extract of the EDGAR panel, for shortlisting.

Reads data/edgar.db and writes a human-readable preview: one summary table
ranking every company, then a per-company block showing the columns that exist
and a handful of rows spread across that company's span.

The point is a shortlisting decision, so the summary carries the things that
actually decide it rather than everything available:

- **scale** - the simulator seeds at $50k MRR; the panel sits at $100M+ ARR.
  Companies whose early quarters are small are the closest thing to the target
  regime, so `qtrs_lt_25m` counts quarters under $25M revenue (~$100M ARR).
- **completeness** - how many quarters have every core field, and what share of
  values were derived rather than filed verbatim.
- **shape** - median QoQ growth, S&M and R&D as a share of revenue, gross
  margin. All scale-free, because absolute dollars are not comparable to the
  simulator and never will be.

Rows are sampled evenly across each company's span rather than taken from the
end, so an early-stage stretch is visible if one exists.

Usage:
    python data/extract_panel_sample.py              # 5 rows per company
    python data/extract_panel_sample.py --rows 8
    python data/extract_panel_sample.py --out somewhere.md
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

DATA_DIR = Path(__file__).resolve().parent
DB_PATH = DATA_DIR / "edgar.db"
OUT_MD = DATA_DIR / "panel_extract.md"
OUT_CSV = DATA_DIR / "panel_shortlist.csv"

SMALL_REVENUE_CEILING = 25_000_000  # ~$100M ARR, the low end of the panel

CORE_CONCEPTS = ("revenue", "rnd_expense", "sm_expense")
USEFUL_CONCEPTS = ("cost_of_revenue", "operating_cash_flow", "cash", "short_term_investments")

RATIO_COLUMNS = [
    ("fiscal_period", "quarter", None),
    ("revenue", "revenue $M", lambda v: f"{v / 1e6:,.1f}" if v is not None else "—"),
    ("qoq_growth", "QoQ %", lambda v: f"{v * 100:+.1f}" if v is not None else "—"),
    ("sm_pct_revenue", "S&M %", lambda v: f"{v * 100:.1f}" if v is not None else "—"),
    ("rnd_pct_revenue", "R&D %", lambda v: f"{v * 100:.1f}" if v is not None else "—"),
    ("gross_margin", "GM %", lambda v: f"{v * 100:.1f}" if v is not None else "—"),
    ("rule_of_40", "R40", lambda v: f"{v:.0f}" if v is not None else "—"),
    ("magic_number", "magic", lambda v: f"{v:.2f}" if v is not None else "—"),
    ("burn_multiple", "burn×", lambda v: f"{v:.2f}" if v is not None else "—"),
    ("cash_and_investments", "cash $M", lambda v: f"{v / 1e6:,.0f}" if v is not None else "—"),
]


def _median(values: list[float]) -> float | None:
    clean = [v for v in values if v is not None]
    return median(clean) if clean else None


def _fmt(value: float | None, scale: float = 1.0, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "—"
    return f"{value * scale:,.{digits}f}{suffix}"


def _evenly_spaced(rows: list[Any], count: int) -> list[Any]:
    """`count` rows spread across the span, always including first and last."""
    if len(rows) <= count:
        return rows
    if count == 1:
        return [rows[-1]]
    step = (len(rows) - 1) / (count - 1)
    return [rows[round(i * step)] for i in range(count)]


def collect(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    connection.row_factory = sqlite3.Row

    companies = connection.execute(
        "SELECT cik, ticker, name, consecutive_q, run_start, run_end "
        "FROM companies WHERE included = 1 ORDER BY ticker"
    ).fetchall()

    out: list[dict[str, Any]] = []
    for company in companies:
        ticker = company["ticker"]

        rows = connection.execute(
            "SELECT * FROM ratios WHERE ticker = ? AND revenue IS NOT NULL "
            "ORDER BY fiscal_period",
            (ticker,),
        ).fetchall()
        if not rows:
            continue

        revenues = [r["revenue"] for r in rows if r["revenue"] is not None]

        concept_counts = {
            r["concept"]: r["n"]
            for r in connection.execute(
                "SELECT concept, COUNT(*) n FROM facts WHERE ticker = ? GROUP BY concept",
                (ticker,),
            )
        }
        derived, total = connection.execute(
            "SELECT SUM(CASE WHEN derivation IS NOT NULL THEN 1 ELSE 0 END), COUNT(*) "
            "FROM facts WHERE ticker = ?",
            (ticker,),
        ).fetchone()

        missing = [c for c in CORE_CONCEPTS + USEFUL_CONCEPTS if not concept_counts.get(c)]

        out.append({
            "ticker": ticker,
            "name": company["name"],
            "cik": company["cik"],
            "quarters": len(rows),
            "consecutive_q": company["consecutive_q"],
            "span_start": rows[0]["fiscal_period"],
            "span_end": rows[-1]["fiscal_period"],
            "rev_min": min(revenues) if revenues else None,
            "rev_median": _median(revenues),
            "rev_max": max(revenues) if revenues else None,
            "qtrs_lt_25m": sum(1 for v in revenues if v < SMALL_REVENUE_CEILING),
            "median_qoq": _median([r["qoq_growth"] for r in rows]),
            "median_sm": _median([r["sm_pct_revenue"] for r in rows]),
            "median_rnd": _median([r["rnd_pct_revenue"] for r in rows]),
            "median_gm": _median([r["gross_margin"] for r in rows]),
            "median_r40": _median([r["rule_of_40"] for r in rows]),
            "derived_pct": (derived / total * 100.0) if total else 0.0,
            "missing": missing,
            "rows": rows,
        })
    return out


def write_markdown(panel: list[dict[str, Any]], rows_per_company: int, path: Path) -> None:
    lines: list[str] = []
    lines.append("# EDGAR panel — per-company extract\n")
    lines.append(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from "
        f"`data/edgar.db`. Regenerate with `python data/extract_panel_sample.py`.\n"
    )
    lines.append(
        f"**{len(panel)} companies.** Every figure is computed from XBRL tags as filed, or "
        f"from exact arithmetic on filed figures where a quarter was only reported "
        f"cumulatively — see `derived %` below and `docs/data_provenance.md` §R3.7. "
        f"Nothing is interpolated or forward-filled.\n"
    )

    lines.append("## How to read this for shortlisting\n")
    lines.append(
        f"- **`qtrs <$25M`** is the column that matters most for this project. The simulator "
        f"seeds at $50k MRR; most of this panel is $100M+ ARR. Companies with a long stretch "
        f"of small quarters are the closest available analogue to the target regime, and are "
        f"the ones worth keeping for retrodiction.\n"
        f"- **`derived %`** is the share of that company's values reconstructed by differencing "
        f"year-to-date figures rather than read verbatim. Filter to `derivation IS NULL` in "
        f"SQL if you want as-filed values only.\n"
        f"- **`gaps`** lists core or useful concepts with no rows at all. A company with gaps "
        f"is not unusable, but any analysis needing that field drops it.\n"
        f"- Everything is scale-free apart from the revenue columns. Absolute dollars are not "
        f"comparable to the simulator and should never be compared to it directly.\n"
    )

    lines.append("## Summary — all companies\n")
    header = ("| Ticker | Company | Qtrs | Span | rev min $M | rev med $M | rev max $M | "
              "qtrs <$25M | QoQ % | S&M % | R&D % | GM % | R40 | derived % | gaps |")
    lines.append(header)
    lines.append("|" + "---|" * 15)
    for c in sorted(panel, key=lambda x: (-x["qtrs_lt_25m"], -x["quarters"])):
        lines.append(
            f"| **{c['ticker']}** | {c['name'][:26]} | {c['quarters']} | "
            f"{c['span_start']}–{c['span_end']} | "
            f"{_fmt(c['rev_min'], 1e-6)} | {_fmt(c['rev_median'], 1e-6)} | "
            f"{_fmt(c['rev_max'], 1e-6, 0)} | "
            f"{c['qtrs_lt_25m']} | {_fmt(c['median_qoq'], 100, 1)} | "
            f"{_fmt(c['median_sm'], 100)} | {_fmt(c['median_rnd'], 100)} | "
            f"{_fmt(c['median_gm'], 100)} | {_fmt(c['median_r40'], 1, 0)} | "
            f"{c['derived_pct']:.0f} | {', '.join(c['missing']) or '—'} |"
        )

    small = [c for c in panel if c["qtrs_lt_25m"] >= 8]
    lines.append(
        f"\n**{len(small)} of {len(panel)} companies have 8+ quarters under $25M revenue.** "
        f"Those are the ones carrying anything close to early-stage dynamics; the rest are "
        f"steady-state large-cap SaaS throughout their filing history.\n"
    )

    lines.append("\n---\n")
    lines.append(f"## Per company — {rows_per_company} rows spread across each span\n")
    lines.append(
        "Rows are sampled evenly from first to last filed quarter, not taken from the end, "
        "so an early-stage stretch shows up if the company has one.\n"
    )

    for c in sorted(panel, key=lambda x: x["ticker"]):
        lines.append(f"\n### {c['ticker']} — {c['name']}\n")
        lines.append(
            f"CIK {c['cik']} · {c['quarters']} quarters with revenue "
            f"({c['span_start']}–{c['span_end']}) · longest complete run "
            f"{c['consecutive_q']}q · {c['derived_pct']:.0f}% derived · "
            f"{c['qtrs_lt_25m']} quarters under $25M revenue"
            + (f" · **gaps: {', '.join(c['missing'])}**" if c["missing"] else "")
            + "\n"
        )
        lines.append("| " + " | ".join(label for _, label, _ in RATIO_COLUMNS) + " |")
        lines.append("|" + "---|" * len(RATIO_COLUMNS))
        for row in _evenly_spaced(c["rows"], rows_per_company):
            cells = []
            for key, _, formatter in RATIO_COLUMNS:
                value = row[key]
                cells.append(str(value) if formatter is None else formatter(value))
            lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def write_csv(panel: list[dict[str, Any]], path: Path) -> None:
    """Same summary, sortable. For filtering rather than reading."""
    fields = ["ticker", "name", "cik", "quarters", "consecutive_q", "span_start", "span_end",
              "rev_min", "rev_median", "rev_max", "qtrs_lt_25m", "median_qoq", "median_sm",
              "median_rnd", "median_gm", "median_r40", "derived_pct", "missing"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for c in sorted(panel, key=lambda x: (-x["qtrs_lt_25m"], -x["quarters"])):
            writer.writerow([
                c["ticker"], c["name"], c["cik"], c["quarters"], c["consecutive_q"],
                c["span_start"], c["span_end"],
                round(c["rev_min"] or 0), round(c["rev_median"] or 0), round(c["rev_max"] or 0),
                c["qtrs_lt_25m"],
                round(c["median_qoq"], 5) if c["median_qoq"] is not None else "",
                round(c["median_sm"], 5) if c["median_sm"] is not None else "",
                round(c["median_rnd"], 5) if c["median_rnd"] is not None else "",
                round(c["median_gm"], 5) if c["median_gm"] is not None else "",
                round(c["median_r40"], 3) if c["median_r40"] is not None else "",
                round(c["derived_pct"], 1),
                ";".join(c["missing"]),
            ])
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=5, help="sample rows per company")
    parser.add_argument("--out", type=Path, default=OUT_MD)
    parser.add_argument("--csv", type=Path, default=OUT_CSV)
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"{DB_PATH} not found. Build it first: python data/ingest_edgar.py")
        return 1

    with sqlite3.connect(DB_PATH) as connection:
        panel = collect(connection)

    if not panel:
        print("No included companies with revenue in the database.")
        return 1

    write_markdown(panel, args.rows, args.out)
    write_csv(panel, args.csv)
    print(f"{len(panel)} companies extracted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
