# EDGAR coverage report

Generated 2026-08-26 08:32 UTC from cached `companyfacts` responses in `data/cache/edgar/`. No network access is required to regenerate this file.

Nothing is imputed, interpolated or forward-filled; a company-quarter missing any core field simply does not count toward its run length.

**On Q4.** Filers do not report Q4 as a three-month figure — 10-Qs carry Q1 plus cumulative six- and nine-month spans, and the 10-K carries only the twelve-month year. Reading three-month facts alone yields Q1–Q2–Q3 and a permanent gap at Q4, capping every consecutive run at three quarters. Those quarters are recovered by differencing two reported year-to-date figures from the same fiscal year: exact arithmetic on filed numbers, not estimation. Each such row carries its `derivation` and the accessions of both operands, so `SELECT ... WHERE derivation IS NULL` restricts any analysis to strictly as-filed values.

## Inclusion criterion (declared before screening)

A candidate enters the panel if and only if all three hold:

1. It reports `SellingAndMarketingExpense` separately from G&A in at least one
   filing. Filers that only report `SellingGeneralAndAdministrativeExpense` are
   excluded: S&M is not recoverable from XBRL alone, and imputing it would be
   inventing the marketing lever we are trying to validate.
2. It uses one of the standard revenue tags.
3. It has at least 16 consecutive quarters in which
   revenue, R&D and S&M are all present.

Nothing is imputed to satisfy these. A company failing any of them is dropped
and the reason recorded.

## Screening outcome

- Candidates screened: **41**
- Included: **39**
- Excluded: **2**


## Panel (included)

| Ticker | Company | CIK | S&M separate | Revenue tag | Consecutive q | Span | Complete q |
|---|---|---|---|---|---|---|---|
| QLYS | QUALYS, INC. | 1107843 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 62 | 2011Q1–2026Q2 | 62 |
| EGHT | 8X8 INC /DE/ | 1023731 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 61 | 2011Q2–2026Q2 | 61 |
| PCTY | Paylocity Holding Corp | 1591698 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 54 | 2013Q1–2026Q2 | 54 |
| RPD | Rapid7, Inc. | 1560327 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 49 | 2014Q2–2026Q2 | 49 |
| TWLO | TWILIO INC | 1447669 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 45 | 2015Q2–2026Q2 | 45 |
| BL | BLACKLINE, INC. | 1666134 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 44 | 2015Q3–2026Q2 | 44 |
| YEXT | Yext, Inc. | 1614178 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 41 | 2016Q2–2026Q2 | 41 |
| BAND | Bandwidth Inc. | 1514416 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 39 | 2016Q3–2026Q1 | 39 |
| ALRM | Alarm.com Holdings, Inc. | 1459200 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 38 | 2017Q1–2026Q2 | 38 |
| RNG | RingCentral, Inc. | 1384905 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 38 | 2017Q1–2026Q2 | 38 |
| TENB | Tenable Holdings, Inc. | 1660280 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 38 | 2017Q1–2026Q2 | 38 |
| WK | WORKIVA INC | 1445305 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 38 | 2017Q1–2026Q2 | 38 |
| APPF | APPFOLIO INC | 1433195 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 37 | 2017Q2–2026Q2 | 37 |
| ZS | Zscaler, Inc. | 1713683 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 37 | 2017Q2–2026Q2 | 37 |
| DOMO | DOMO, INC. | 1505952 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 36 | 2017Q3–2026Q2 | 36 |
| ESTC | Elastic N.V. | 1707753 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 35 | 2017Q4–2026Q2 | 35 |
| MDB | MongoDB, Inc. | 1441816 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 35 | 2017Q4–2026Q2 | 35 |
| OKTA | Okta, Inc. | 1660134 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 35 | 2017Q4–2026Q2 | 35 |
| CRWD | CrowdStrike Holdings, Inc. | 1535527 | yes | `RevenueFromContractWithCustomerIncludingAssessedTax` | 33 | 2018Q2–2026Q2 | 33 |
| FSLY | Fastly, Inc. | 1517413 | yes | `RevenueFromContractWithCustomerIncludingAssessedTax` | 33 | 2018Q2–2026Q2 | 33 |
| PD | PagerDuty, Inc. | 1568100 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 33 | 2018Q2–2026Q2 | 33 |
| ZM | Zoom Communications, Inc. | 1585521 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 33 | 2018Q2–2026Q2 | 33 |
| DDOG | Datadog, Inc. | 1561550 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 32 | 2018Q3–2026Q2 | 32 |
| NET | Cloudflare, Inc. | 1477333 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 32 | 2018Q3–2026Q2 | 32 |
| SPT | Sprout Social, Inc. | 1517375 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 30 | 2019Q1–2026Q2 | 30 |
| ASAN | Asana, Inc. | 1477720 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 27 | 2019Q4–2026Q2 | 27 |
| AI | C3.ai, Inc. | 1577526 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 26 | 2020Q1–2026Q2 | 26 |
| DOCN | DigitalOcean Holdings, Inc. | 1582961 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 26 | 2020Q1–2026Q2 | 26 |
| KLTR | KALTURA INC | 1432133 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 25 | 2020Q2–2026Q2 | 25 |
| PATH | UiPath, Inc. | 1734722 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 25 | 2020Q2–2026Q2 | 25 |
| ZETA | Zeta Global Holdings Corp. | 1851003 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 25 | 2020Q2–2026Q2 | 25 |
| AMPL | Amplitude, Inc. | 1866692 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 24 | 2020Q3–2026Q2 | 24 |
| FRSH | Freshworks Inc. | 1544522 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 24 | 2020Q3–2026Q2 | 24 |
| INTA | Intapp, Inc. | 1565687 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 24 | 2020Q3–2026Q2 | 24 |
| S | SentinelOne, Inc. | 1583708 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 24 | 2020Q3–2026Q2 | 24 |
| WEAV | Weave Communications, Inc. | 1609151 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 24 | 2020Q3–2026Q2 | 24 |
| BRZE | Braze, Inc. | 1676238 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 23 | 2020Q4–2026Q2 | 23 |
| GTLB | Gitlab Inc. | 1653482 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 23 | 2020Q4–2026Q2 | 23 |
| KVYO | Klaviyo, Inc. | 1835830 | yes | `RevenueFromContractWithCustomerExcludingAssessedTax` | 16 | 2022Q3–2026Q2 | 16 |

## Excluded, with reasons

| Ticker | Company | Consecutive q | Reason |
|---|---|---|---|
| MNDY | monday.com Ltd. | 0 | no SellingAndMarketingExpense tag; no standard revenue tag; longest complete run 0q < 16q |
| PAYC | Paycom Software, Inc. | 0 | reports SG&A combined only; S&M not separable from XBRL; longest complete run 0q < 16q |

### Note on the S&M / SG&A split

1 of 41 candidates report `SellingGeneralAndAdministrativeExpense` without a separate `SellingAndMarketingExpense`. This is the Tier-2 tag mismatch the acquisition plan anticipated. It is resolved by exclusion, not by a better tag map: marketing spend genuinely cannot be recovered from those filings, and any split we invented would be our assumption wearing an audited company's name.


## Scope statement for the review

Agent validation on this panel covers R&D fully and marketing spend for the **39 of 41** filers that separate S&M from G&A. Hiring is available at annual granularity only (headcount is 10-K cover-page prose, not XBRL). Pricing and channel are not disclosed by any filer and are not validatable here — they are validated against private benchmarks instead, or declared unidentified.

