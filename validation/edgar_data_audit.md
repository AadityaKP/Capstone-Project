# EDGAR capability audit — what the data can and cannot support

Audit date: 2026-08-30. Basis: the repository's **already-built** EDGAR pipeline
(`data/ingest_edgar.py`, `data/edgar.db`, `data/edgar_ratios.csv`,
`data/coverage_report.md`, `data/panel_extract.md`), re-verified against the SQLite
store this session. No new network access was required; the cached
SEC `companyfacts` responses for 41 CIKs are in `data/cache/edgar/`.

## 1. What exists on disk (verified)

- `edgar.db`: `companies` (41 rows: 39 included, 2 excluded with reasons),
  `facts` (10,065 rows), keyed (cik, fiscal_period, concept).
- Concepts and coverage (rows / companies): cash 1,452/39 · rnd_expense 1,395/39 ·
  sm_expense 1,391/39 · ga_expense 1,385/39 · revenue 1,332/39 · cost_of_revenue
  1,303/39 · operating_cash_flow 1,292/39 · short_term_investments 442/20.
- `edgar_ratios.csv`: 1,473 company-quarters; **1,288 complete core quarters**
  (revenue + QoQ growth + S&M% + R&D%) across **39 companies**, 2010Q2–2026Q2.
  Also: gross_margin (1,258), magic_number (1,288), rule_of_40 (1,179),
  burn_multiple (294), cash_and_investments (1,473).
- 24.4% of fact values are *derived* — Q4 recovered by exact differencing of
  year-to-date figures, with `derivation`/`derived_from` recorded per row.
  `WHERE derivation IS NULL` restricts to strictly as-filed values.
- Inclusion criteria were declared before screening (S&M reported separately from G&A;
  standard revenue tag; ≥16 consecutive complete quarters). 2/41 candidates excluded
  (MNDY: non-standard tags; PAYC: SG&A combined only).

**Assessment: the cohort and inclusion criteria are sound and already documented.
This validation adopts the 39-company panel as-is rather than re-screening.** The
cohort is SaaS/software-only, which matches the simulator's domain claim exactly.

## 2. Field-by-field mapping to the simulator

| EDGAR field | XBRL concept(s) | Freq | Coverage | Reliability | Simulator mapping | Class |
|---|---|---|---|---|---|---|
| Revenue | RevenueFromContractWithCustomer(Ex/In)cludingAssessedTax | Q | 1,332 q / 39 co | high (24% derived, exact arithmetic) | MRR ≈ quarterly revenue ÷ 3; QoQ growth vs sim quarterly growth | **Direct** |
| S&M expense | SellingAndMarketingExpense | Q | 1,391 q / 39 co | high (panel screened for it) | marketing spend; S&M % of revenue vs sim marketing/MRR | **Direct** |
| R&D expense | ResearchAndDevelopmentExpense | Q | 1,395 q / 39 co | high | r_and_d_spend; R&D % of revenue | **Direct** |
| Cash (+STI) | CashAndCashEquivalents…(+ShortTermInvestments) | Q | 1,452 q (STI only 20 co) | high; STI gap noted per company | cash | **Direct** (levels), scale caveat |
| Cost of revenue | CostOfRevenue etc. | Q | 1,303 q | high | gross margin (sim: gross_margin flag; None in research profile) | **Direct ratio** |
| G&A | GeneralAndAdministrativeExpense | Q | 1,385 q | high | part of burn proxy | **Proxy** (burn = G&A + S&M + R&D + CoR vs sim burn) |
| Operating cash flow | NetCashProvidedByUsedInOperatingActivities | Q | 1,292 q | moderate (YTD differencing) | burn multiple; net burn | **Proxy** |
| Rule of 40 | derived | Q | 1,179 q | derived | sim rule_of_40 (definitions differ: EDGAR uses op-margin-based; sim uses −burn/MRR) | **Comparative only — definitions differ; compare shapes, not levels** |
| Headcount | 10-K cover prose | **A** | annual only | low for quarterly use | hiring action | **Unavailable** (quarterly) |
| Churn / retention | not in XBRL | — | none | — | churn_* | **Unavailable** (ChartMogul benchmarks give plausible *ranges* only) |
| CAC / LTV / customer counts | not in XBRL | — | none | — | cac, ltv | **Unavailable** (magic number = ΔARR/S&M is the closest published analogue, computed in ratios) |
| Price / ARPU / pricing actions | not in XBRL | — | none | — | price, pricing action | **Unavailable** |
| Product quality | not observable | — | none | — | product_quality | **Unavailable** |
| Competitors, confidence, unemployment, interest rate | not EDGAR | — | — | — | macro block | **Out of EDGAR scope** (deliberately not validated here) |

## 3. Temporal compatibility

Simulator is monthly; EDGAR is quarterly. **Rule adopted: simulator traces are
aggregated to quarters (revenue summed over 3 months; ratios computed on quarterly
sums; growth as quarter-over-quarter). No monthly EDGAR series is fabricated,
no interpolation anywhere.**

## 4. Scale compatibility — the binding constraint

The simulator's research anchor is $50k MRR ($0.6M ARR). The panel's *smallest*
observed quarter is ≈$15.8M revenue (≈$5.3M/month MRR) — two orders of magnitude
larger; only 2 of 39 companies (QLYS 9, EGHT 8) have 8+ quarters under $25M.
Consequences, adopted as design rules:

1. **Never compare absolute dollars.** All environment validation uses scale-free
   quantities: growth rates, expense-to-revenue ratios, margins, persistence,
   volatility, correlation signs.
2. The EDGAR panel represents *successful post-IPO SaaS* — a survivor-biased,
   right-tail population. It bounds what realistic SaaS dynamics look like; it is
   not a sample of startups. Distributional comparisons are therefore judged on
   overlap and shape, with the bias stated, not on exact equality.
3. The real-company backtest initializes the simulator **at EDGAR scale** with the
   scale-aware physics flags (`scale_aware_marketing`, `scale_aware_rnd`,
   `gross_margin`, real `monthly_burn` from opex, `scale_absolutes` = mrr/50k),
   because the legacy absolute constants are meaningless at $30M/quarter. This is
   itself disclosed: the backtest tests the *scale-aware* configuration.

## 5. What EDGAR validation can therefore support

Feasible and supported by on-disk data:
- distributional validity of quarterly revenue growth (median/IQR/percentiles, KS,
  Wasserstein) — sim vs 1,288 EDGAR quarters;
- growth persistence (lag-1 autocorrelation within company), volatility, and growth
  deceleration with scale;
- expense-structure validity: S&M% and R&D% of revenue levels and dispersion vs the
  simulator's chosen spend ratios per policy;
- structural correlations: growth vs S&M intensity ("magic number" direction),
  growth vs margin, cash trajectory vs burn;
- drawdown/recovery: frequency, depth and duration of revenue-decline episodes in
  EDGAR vs simulator shock-recovery behaviour (comparative — EDGAR shocks are not
  the simulator's shock types);
- real-company counterfactual: initialize from company-quarter states (revenue,
  cash, opex; assumed price/churn/CAC from calibrated benchmarks, labelled
  assumed), roll 4 quarters under hold / heuristic / boardroom arms, compare with
  the company's actual next 4 quarters. **Model-based counterfactual evidence only.**

Not supportable with EDGAR (declared, not forced): churn, CAC, LTV, price levels or
pricing actions, product quality, customer counts, monthly dynamics, hiring at
quarterly resolution, macro block realism.
