# Dataset card — EDGAR SaaS quarterly panel (frozen)

Consolidates `data/coverage_report.md` (screening, generated 2026-08-26) and
`validation/edgar_data_audit.md` (capability audit, 2026-08-30). Card written
2026-09-01. **This panel is the project's calibration and validation benchmark;
all agent experience data is simulator-generated, never drawn from this panel.**

## Source and construction

- **Source:** SEC EDGAR `companyfacts` XBRL API (cached responses in
  `data/cache/edgar/`, 41 CIKs; no network needed to regenerate).
- **Pipeline:** `data/ingest_edgar.py` → `data/edgar.db` (tables `companies`,
  `facts`) → `data/edgar_facts.csv` (long form) and `data/edgar_ratios.csv`
  (per company-quarter, the analysis panel).
- **Q4 handling:** filers report Q4 only cumulatively; Q4 rows are recovered by
  exact differencing of two filed year-to-date figures (24.4% of fact values,
  each row carrying `derivation` + `derived_from`). Nothing is imputed,
  interpolated or forward-filled. `WHERE derivation IS NULL` restricts to
  strictly as-filed values.

## Inclusion criteria (declared before screening)

1. Reports `SellingAndMarketingExpense` separately from G&A (filers reporting
   only combined SG&A are excluded — S&M is not recoverable from XBRL).
2. Uses a standard revenue tag (`RevenueFromContractWithCustomer…`).
3. Has ≥16 consecutive quarters with revenue, R&D and S&M all present.

Screened 41 candidates → **39 included, 2 excluded** (MNDY: non-standard tags;
PAYC: SG&A combined only).

## Coverage

- **Companies (39):** QLYS EGHT PCTY RPD TWLO BL YEXT BAND ALRM RNG TENB WK
  APPF ZS DOMO ESTC MDB OKTA CRWD FSLY PD ZM DDOG NET SPT ASAN AI DOCN KLTR
  PATH ZETA AMPL FRSH INTA S WEAV BRZE GTLB KVYO
- **Quarter range:** 2010Q2–2026Q2. **Complete core quarters** (revenue + QoQ
  growth + S&M% + R&D% all present): **1,288** (1,473 company-quarter rows in
  `edgar_ratios.csv`, 1,332 with revenue).
- **Frequency:** quarterly. Simulator output is aggregated to quarters for any
  comparison; no monthly EDGAR series exists or is fabricated.

## Fields

Seven directly ingested concepts (rows / companies in `facts`): revenue
1,332/39 · sm_expense 1,391/39 · rnd_expense 1,395/39 · ga_expense 1,385/39 ·
cost_of_revenue 1,303/39 · cash 1,452/39 · operating_cash_flow 1,292/39
(plus short_term_investments 442/20).

Derived quantities used in validation (in `edgar_ratios.csv`): QoQ revenue
growth, S&M % of revenue, R&D % of revenue (together: discretionary spend
intensity), gross margin, magic number, burn multiple, Rule of 40,
cash-and-investments.

## What this panel can / cannot validate

| Can validate (used in) | Cannot validate (declared, not forced) |
|---|---|
| Revenue-growth distribution, persistence, volatility, deceleration with scale (E1–E3, E5) | Churn, CAC, LTV, customer counts (not in XBRL; ChartMogul benchmarks give range checks only) |
| Spend structure: S&M%, R&D% of revenue (E4) | Price/ARPU levels or pricing actions |
| Gross margin range (E7b) | Product quality; any monthly dynamics |
| Drawdown/recovery shape of revenue (E6) | Hiring at quarterly resolution (headcount is annual 10-K prose) |
| Real-state initialization for the counterfactual backtest (C1); observational allocation direction (C2) | Macro block (rates, unemployment, confidence — out of EDGAR scope) |

**Known biases:** survivor-biased, right-tail population (successful post-IPO
SaaS; smallest observed quarter ≈ $15.8M revenue, two orders of magnitude above
the simulator's $50k-MRR anchor) — all comparisons are therefore scale-free;
absolute dollars are never compared.

## Freeze — content hashes (SHA-256, 2026-09-01)

| File | SHA-256 |
|---|---|
| `data/edgar_ratios.csv` | `0CED2D6599C534C65FC60C07C81243078B0535C2971E451C2C38977662AAA4CE` |
| `data/edgar_facts.csv` | `5CEE17F8151246F74F9A72D581A07BABDAACADAD0E6629AC0F3E1B0457FDBB0D` |
| `data/edgar.db` | `D9BE3195EE1D2421BD0A4C1B53B8C6EDDBB03181316D347954FCCA9345AE65BA` |

Verify with `Get-FileHash data\edgar_ratios.csv -Algorithm SHA256` (PowerShell)
or `sha256sum data/edgar_ratios.csv`. Any analysis against a file whose hash
differs is not against the frozen panel.
