# EDGAR panel — per-company extract

Generated 2026-08-26 10:43 UTC from `data/edgar.db`. Regenerate with `python data/extract_panel_sample.py`.

**39 companies.** Every figure is computed from XBRL tags as filed, or from exact arithmetic on filed figures where a quarter was only reported cumulatively — see `derived %` below and `docs/data_provenance.md` §R3.7. Nothing is interpolated or forward-filled.

## How to read this for shortlisting

- **`qtrs <$25M`** is the column that matters most for this project. The simulator seeds at $50k MRR; most of this panel is $100M+ ARR. Companies with a long stretch of small quarters are the closest available analogue to the target regime, and are the ones worth keeping for retrodiction.
- **`derived %`** is the share of that company's values reconstructed by differencing year-to-date figures rather than read verbatim. Filter to `derivation IS NULL` in SQL if you want as-filed values only.
- **`gaps`** lists core or useful concepts with no rows at all. A company with gaps is not unusable, but any analysis needing that field drops it.
- Everything is scale-free apart from the revenue columns. Absolute dollars are not comparable to the simulator and should never be compared to it directly.

## Summary — all companies

| Ticker | Company | Qtrs | Span | rev min $M | rev med $M | rev max $M | qtrs <$25M | QoQ % | S&M % | R&D % | GM % | R40 | derived % | gaps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **QLYS** | QUALYS, INC. | 62 | 2011Q1–2026Q2 | 17.7 | 72.9 | 182 | 9 | 4.1 | 22.9 | 19.6 | 79.1 | 43 | 19 | — |
| **EGHT** | 8X8 INC /DE/ | 65 | 2010Q2–2026Q2 | 16.8 | 83.2 | 190 | 8 | 3.9 | 45.1 | 14.7 | 67.8 | 9 | 15 | — |
| **PCTY** | Paylocity Holding Corp | 56 | 2012Q3–2026Q2 | 15.8 | 124.9 | 470 | 6 | 5.9 | 26.4 | 12.2 | 64.1 | 24 | 25 | short_term_investments |
| **RPD** | Rapid7, Inc. | 49 | 2014Q2–2026Q2 | 17.9 | 98.9 | 218 | 4 | 5.6 | 48.0 | 26.3 | 70.6 | 15 | 23 | — |
| **BL** | BLACKLINE, INC. | 46 | 2015Q1–2026Q2 | 18.0 | 92.9 | 188 | 4 | 4.5 | 48.9 | 16.0 | 75.9 | 21 | 25 | short_term_investments |
| **SPT** | Sprout Social, Inc. | 30 | 2019Q1–2026Q2 | 23.4 | 67.5 | 124 | 2 | 6.5 | 45.1 | 23.8 | 76.5 | 10 | 27 | — |
| **S** | SentinelOne, Inc. | 24 | 2020Q3–2026Q2 | 20.7 | 141.4 | 277 | 2 | 9.3 | 65.0 | 38.8 | 69.3 | 2 | 25 | — |
| **WEAV** | Weave Communications, Inc. | 24 | 2020Q3–2026Q2 | 21.4 | 42.6 | 68 | 2 | 4.8 | 43.2 | 20.5 | 68.0 | 9 | 28 | short_term_investments |
| **TWLO** | TWILIO INC | 45 | 2015Q2–2026Q2 | 38.0 | 548.1 | 1,499 | 0 | 8.2 | 25.8 | 26.9 | 51.5 | 14 | 27 | short_term_investments |
| **YEXT** | Yext, Inc. | 41 | 2016Q2–2026Q2 | 27.1 | 92.2 | 114 | 0 | 3.2 | 61.3 | 17.5 | 74.6 | 1 | 26 | short_term_investments |
| **BAND** | Bandwidth Inc. | 39 | 2016Q3–2026Q1 | 38.6 | 120.7 | 210 | 0 | 4.3 | 14.5 | 14.8 | 43.6 | 13 | 26 | — |
| **ALRM** | Alarm.com Holdings, Inc. | 38 | 2017Q1–2026Q2 | 74.2 | 193.8 | 278 | 0 | 3.6 | 11.8 | 25.2 | 64.2 | 20 | 23 | short_term_investments |
| **RNG** | RingCentral, Inc. | 38 | 2017Q1–2026Q2 | 112.2 | 431.6 | 657 | 0 | 4.5 | 48.4 | 15.2 | 72.2 | 19 | 25 | short_term_investments |
| **TENB** | Tenable Holdings, Inc. | 38 | 2017Q1–2026Q2 | 40.5 | 143.8 | 269 | 0 | 5.6 | 50.4 | 22.1 | 79.5 | 21 | 17 | — |
| **WK** | WORKIVA INC | 38 | 2017Q1–2026Q2 | 49.4 | 116.7 | 255 | 0 | 4.7 | 42.2 | 27.2 | 75.5 | 14 | 11 | short_term_investments |
| **APPF** | APPFOLIO INC | 37 | 2017Q2–2026Q2 | 35.9 | 95.8 | 281 | 0 | 6.8 | 18.6 | 18.2 | 60.8 | 24 | 28 | short_term_investments |
| **ZS** | Zscaler, Inc. | 37 | 2017Q2–2026Q2 | 33.0 | 230.5 | 850 | 0 | 9.2 | 60.3 | 23.3 | 77.8 | 34 | 28 | short_term_investments |
| **DOMO** | DOMO, INC. | 36 | 2017Q3–2026Q2 | 25.9 | 67.5 | 80 | 0 | 3.5 | 55.1 | 30.9 | 74.1 | -3 | 27 | — |
| **ESTC** | Elastic N.V. | 35 | 2017Q4–2026Q2 | 37.0 | 223.9 | 451 | 0 | 6.8 | 45.6 | 31.2 | 73.6 | 12 | 29 | — |
| **MDB** | MongoDB, Inc. | 35 | 2017Q4–2026Q2 | 41.5 | 266.5 | 695 | 0 | 8.0 | 52.6 | 32.4 | 71.9 | 10 | 29 | short_term_investments |
| **OKTA** | Okta, Inc. | 35 | 2017Q4–2026Q2 | 66.9 | 382.8 | 765 | 0 | 8.1 | 53.9 | 26.7 | 73.3 | 23 | 25 | — |
| **CRWD** | CrowdStrike Holdings, Inc. | 33 | 2018Q2–2026Q2 | 47.3 | 487.8 | 1,386 | 0 | 10.6 | 41.3 | 26.7 | 73.7 | 45 | 26 | — |
| **FSLY** | Fastly, Inc. | 33 | 2018Q2–2026Q2 | 34.4 | 102.5 | 183 | 0 | 4.6 | 35.6 | 26.2 | 54.8 | 1 | 25 | — |
| **PD** | PagerDuty, Inc. | 33 | 2018Q2–2026Q2 | 25.0 | 85.4 | 125 | 0 | 5.8 | 53.1 | 30.3 | 84.0 | 15 | 25 | — |
| **ZM** | Zoom Communications, Inc. | 33 | 2018Q2–2026Q2 | 60.1 | 1,073.8 | 1,247 | 0 | 2.6 | 32.4 | 13.4 | 76.1 | 41 | 26 | short_term_investments |
| **DDOG** | Datadog, Inc. | 32 | 2018Q3–2026Q2 | 51.1 | 421.3 | 1,121 | 0 | 8.6 | 29.0 | 42.4 | 79.3 | 35 | 28 | short_term_investments |
| **NET** | Cloudflare, Inc. | 32 | 2018Q3–2026Q2 | 50.1 | 244.2 | 696 | 0 | 9.2 | 48.5 | 27.6 | 76.7 | 25 | 28 | short_term_investments |
| **ASAN** | Asana, Inc. | 27 | 2019Q4–2026Q2 | 38.1 | 150.2 | 206 | 0 | 5.7 | 71.1 | 50.1 | 89.6 | -6 | 28 | short_term_investments |
| **AI** | C3.ai, Inc. | 26 | 2020Q1–2026Q2 | 40.5 | 68.2 | 109 | 0 | 3.6 | 65.7 | 61.7 | 66.1 | -33 | 26 | — |
| **DOCN** | DigitalOcean Holdings, Inc | 26 | 2020Q1–2026Q2 | 72.8 | 167.5 | 281 | 0 | 5.5 | 9.6 | 20.2 | 58.8 | 38 | 20 | — |
| **KLTR** | KALTURA INC | 25 | 2020Q2–2026Q2 | 28.7 | 43.9 | 47 | 0 | 1.0 | 26.8 | 28.3 | 64.2 | 1 | 25 | — |
| **PATH** | UiPath, Inc. | 25 | 2020Q2–2026Q2 | 113.1 | 289.7 | 481 | 0 | 7.1 | 58.7 | 25.9 | 83.5 | 17 | 24 | — |
| **ZETA** | Zeta Global Holdings Corp. | 25 | 2020Q2–2026Q2 | 77.1 | 175.1 | 443 | 0 | 10.4 | 33.2 | 10.1 | 60.9 | 24 | 27 | short_term_investments |
| **AMPL** | Amplitude, Inc. | 24 | 2020Q3–2026Q2 | 26.4 | 69.2 | 101 | 0 | 4.2 | 54.3 | 30.5 | 71.9 | 6 | 25 | — |
| **FRSH** | Freshworks Inc. | 24 | 2020Q3–2026Q2 | 66.2 | 149.3 | 237 | 0 | 5.4 | 56.3 | 22.3 | 82.9 | 24 | 25 | — |
| **INTA** | Intapp, Inc. | 24 | 2020Q3–2026Q2 | 48.1 | 98.1 | 153 | 0 | 4.4 | 33.9 | 27.1 | 69.0 | 17 | 28 | short_term_investments |
| **BRZE** | Braze, Inc. | 23 | 2020Q4–2026Q2 | 39.3 | 115.1 | 211 | 0 | 7.6 | 49.1 | 23.1 | 67.2 | 10 | 28 | short_term_investments |
| **GTLB** | Gitlab Inc. | 23 | 2020Q4–2026Q2 | 42.2 | 139.6 | 264 | 0 | 8.5 | 66.0 | 35.1 | 88.3 | 13 | 28 | — |
| **KVYO** | Klaviyo, Inc. | 16 | 2022Q3–2026Q2 | 119.2 | 228.7 | 371 | 0 | 5.8 | 42.5 | 24.8 | 75.6 | 24 | 27 | short_term_investments |

**2 of 39 companies have 8+ quarters under $25M revenue.** Those are the ones carrying anything close to early-stage dynamics; the rest are steady-state large-cap SaaS throughout their filing history.


---

## Per company — 5 rows spread across each span

Rows are sampled evenly from first to last filed quarter, not taken from the end, so an early-stage stretch shows up if the company has one.


### AI — C3.ai, Inc.

CIK 1577526 · 26 quarters with revenue (2020Q1–2026Q2) · longest complete run 26q · 26% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q1 | 41.3 | — | 56.1 | 29.9 | 73.5 | — | — | — | 50 |
| 2021Q3 | 52.4 | +0.2 | 70.3 | 51.0 | 75.1 | 2 | 0.01 | — | 1,099 |
| 2023Q1 | 66.7 | +6.8 | 65.2 | 82.6 | 66.6 | -71 | 0.39 | 3.05 | 772 |
| 2024Q4 | 94.3 | +8.2 | 59.0 | 59.1 | 61.3 | -33 | 0.51 | 1.36 | 730 |
| 2026Q2 | 51.6 | -3.1 | 95.6 | 91.6 | 21.9 | -109 | -0.13 | — | 575 |

### ALRM — Alarm.com Holdings, Inc.

CIK 1459200 · 38 quarters with revenue (2017Q1–2026Q2) · longest complete run 38q · 23% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q1 | 74.2 | — | 13.9 | 19.6 | 64.1 | — | — | — | 63 |
| 2019Q2 | 121.7 | +8.3 | 12.8 | 23.4 | 63.4 | 28 | 2.39 | — | 151 |
| 2021Q3 | 192.3 | +1.8 | 11.7 | 23.0 | 58.2 | 22 | 0.61 | — | 700 |
| 2024Q1 | 223.3 | -1.3 | 11.4 | 29.5 | 65.7 | 21 | -0.46 | — | 748 |
| 2026Q2 | 277.7 | +4.7 | 11.9 | 25.6 | 65.6 | 20 | 1.52 | — | 479 |

### AMPL — Amplitude, Inc.

CIK 1866692 · 24 quarters with revenue (2020Q3–2026Q2) · longest complete run 24q · 25% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q3 | 26.4 | — | 43.5 | 21.2 | 70.5 | — | — | — | 119 |
| 2022Q1 | 53.1 | +7.4 | 53.0 | 31.1 | 69.7 | -8 | 0.52 | 0.57 | 300 |
| 2023Q3 | 70.6 | +4.2 | 54.5 | 30.9 | 75.5 | 16 | 0.30 | — | 321 |
| 2024Q4 | 78.1 | +3.9 | 54.4 | 44.1 | 74.7 | 8 | 0.27 | — | 241 |
| 2026Q2 | 100.9 | +7.9 | 49.4 | 33.4 | 68.5 | 33 | 0.59 | — | 141 |

### APPF — APPFOLIO INC

CIK 1433195 · 37 quarters with revenue (2017Q2–2026Q2) · longest complete run 37q · 28% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q2 | 35.9 | — | 20.0 | 11.2 | 61.8 | — | — | — | 7 |
| 2019Q3 | 67.9 | +6.8 | 18.6 | 15.6 | 61.8 | 24 | 1.36 | — | 20 |
| 2021Q4 | 95.6 | -0.2 | 20.9 | 20.5 | 59.1 | 9 | -0.04 | — | 58 |
| 2024Q1 | 187.4 | +9.1 | 13.0 | 20.2 | — | — | 2.55 | — | 59 |
| 2026Q2 | 281.1 | +7.2 | 15.6 | 18.1 | — | — | 1.72 | — | 217 |

### ASAN — Asana, Inc.

CIK 1477720 · 27 quarters with revenue (2019Q4–2026Q2) · longest complete run 27q · 28% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2019Q4 | 38.1 | — | 94.3 | 104.3 | 86.0 | — | — | — | 15 |
| 2021Q2 | 76.7 | +12.1 | 74.1 | 52.1 | 89.7 | 2 | 0.58 | 0.22 | 264 |
| 2023Q1 | 150.2 | +6.2 | 76.4 | 54.1 | 89.9 | -14 | 0.31 | 0.88 | 527 |
| 2024Q4 | 183.9 | +2.6 | 56.9 | 45.3 | 89.2 | -5 | 0.18 | 0.80 | 197 |
| 2026Q2 | 205.1 | -0.2 | 45.1 | 32.2 | 87.6 | 19 | -0.02 | — | 194 |

### BAND — Bandwidth Inc.

CIK 1514416 · 39 quarters with revenue (2016Q3–2026Q1) · longest complete run 39q · 26% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2016Q3 | 38.6 | — | 6.3 | 6.2 | 44.3 | — | — | — | 6 |
| 2019Q1 | 53.3 | +1.9 | 15.7 | 14.5 | 46.1 | -15 | 0.47 | 2.31 | 198 |
| 2021Q2 | 120.7 | +6.3 | 16.7 | 13.6 | 44.3 | 5 | 1.42 | 0.04 | 310 |
| 2023Q3 | 152.0 | +4.2 | 16.5 | 16.3 | 39.1 | 19 | 0.98 | — | 139 |
| 2026Q1 | 208.8 | +0.5 | 11.8 | 18.4 | 37.3 | 5 | 0.18 | — | 50 |

### BL — BLACKLINE, INC.

CIK 1666134 · 46 quarters with revenue (2015Q1–2026Q2) · longest complete run 44q · 25% derived · 4 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2015Q1 | 18.0 | — | — | — | — | — | — | — | 0 |
| 2017Q4 | 50.0 | +10.1 | 52.2 | 12.1 | 78.0 | 16 | 0.70 | — | 31 |
| 2020Q3 | 90.2 | +8.3 | 47.2 | 16.4 | 81.2 | 32 | 0.65 | — | 408 |
| 2023Q3 | 150.7 | +4.2 | 40.9 | 17.7 | 75.6 | 29 | 0.40 | — | 236 |
| 2026Q2 | 187.8 | +2.5 | 36.5 | 16.7 | 76.0 | 27 | 0.27 | — | 243 |

### BRZE — Braze, Inc.

CIK 1676238 · 23 quarters with revenue (2020Q4–2026Q2) · longest complete run 23q · 28% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q4 | 39.3 | — | 48.7 | 18.8 | 63.3 | — | — | — | 30 |
| 2022Q2 | 77.5 | +10.0 | 59.4 | 27.9 | 66.6 | 33 | 0.61 | — | 91 |
| 2023Q3 | 115.1 | +13.1 | 52.5 | 25.3 | 69.2 | -2 | 0.88 | 0.33 | 77 |
| 2024Q4 | 152.1 | +4.5 | 49.1 | 21.6 | 69.8 | -3 | 0.35 | 0.44 | 61 |
| 2026Q2 | 211.0 | +2.8 | 42.3 | 21.8 | 65.7 | 16 | 0.26 | — | 145 |

### CRWD — CrowdStrike Holdings, Inc.

CIK 1535527 · 33 quarters with revenue (2018Q2–2026Q2) · longest complete run 33q · 26% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2018Q2 | 47.3 | — | 77.4 | 37.2 | 59.0 | — | — | — | 49 |
| 2020Q2 | 178.1 | +17.1 | 49.5 | 22.8 | 73.7 | 72 | 1.18 | — | 1,005 |
| 2022Q2 | 487.8 | +13.2 | 39.7 | 25.3 | 74.0 | 57 | 1.17 | — | 2,153 |
| 2024Q2 | 921.0 | +9.0 | 38.0 | 25.5 | 75.6 | 51 | 0.86 | — | 3,702 |
| 2026Q2 | 1,385.6 | +6.1 | 35.3 | 29.5 | 75.3 | 49 | 0.66 | — | 4,553 |

### DDOG — Datadog, Inc.

CIK 1561550 · 32 quarters with revenue (2018Q3–2026Q2) · longest complete run 32q · 28% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2018Q3 | 51.1 | — | 49.2 | 27.7 | 76.3 | — | — | — | 64 |
| 2020Q3 | 154.7 | +10.5 | 36.9 | 36.5 | 78.0 | 34 | 1.03 | — | 199 |
| 2022Q3 | 436.5 | +7.5 | 29.7 | 47.0 | 78.6 | 27 | 0.94 | — | 295 |
| 2024Q2 | 645.3 | +5.6 | 29.0 | 42.6 | 80.9 | 31 | 0.73 | — | 411 |
| 2026Q2 | 1,121.5 | +11.4 | 27.8 | 42.6 | 78.6 | 40 | 1.48 | — | 435 |

### DOCN — DigitalOcean Holdings, Inc.

CIK 1582961 · 26 quarters with revenue (2020Q1–2026Q2) · longest complete run 26q · 20% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q1 | 72.8 | — | 13.0 | 26.8 | 52.4 | — | — | — | 91 |
| 2021Q3 | 111.4 | +7.3 | 11.9 | 26.9 | 61.0 | 43 | 2.29 | — | 590 |
| 2023Q1 | 165.1 | +1.3 | 9.9 | 22.7 | 54.9 | 23 | 0.52 | — | 613 |
| 2024Q4 | 204.9 | +3.2 | 9.5 | 19.7 | 61.5 | 38 | 1.33 | — | 428 |
| 2026Q2 | 281.2 | +9.0 | 8.0 | 20.5 | 55.0 | 48 | 4.13 | — | 767 |

### DOMO — DOMO, INC.

CIK 1505952 · 36 quarters with revenue (2017Q3–2026Q2) · longest complete run 36q · 27% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q3 | 25.9 | — | 121.3 | 77.9 | 58.9 | — | — | — | 48 |
| 2019Q4 | 44.8 | +7.5 | 66.5 | 39.3 | 67.7 | -36 | 0.42 | 1.57 | 116 |
| 2022Q1 | 70.0 | +7.5 | 56.3 | 33.6 | 73.5 | 9 | 0.50 | — | 84 |
| 2024Q1 | 80.2 | +0.6 | 49.2 | 26.3 | 76.3 | 7 | 0.05 | — | 61 |
| 2026Q2 | 79.4 | -0.3 | 47.3 | 23.5 | 73.7 | 6 | -0.02 | — | 39 |

### EGHT — 8X8 INC /DE/

CIK 1023731 · 65 quarters with revenue (2010Q2–2026Q2) · longest complete run 61q · 15% derived · 8 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2010Q2 | 16.8 | — | — | 7.3 | — | — | — | — | 18 |
| 2014Q2 | 37.9 | +5.9 | 50.5 | 9.0 | — | — | 0.44 | — | 182 |
| 2018Q2 | 83.2 | +4.9 | 48.7 | 15.7 | — | — | 0.38 | — | 144 |
| 2022Q2 | 187.6 | +3.4 | 44.5 | 18.6 | — | — | 0.30 | — | 142 |
| 2026Q2 | 190.2 | +2.7 | 30.9 | 14.9 | 61.2 | 12 | 0.34 | — | 91 |

### ESTC — Elastic N.V.

CIK 1707753 · 35 quarters with revenue (2017Q4–2026Q2) · longest complete run 35q · 29% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q4 | 37.0 | — | 45.6 | 32.9 | 75.8 | — | — | — | 0 |
| 2019Q4 | 101.1 | +12.7 | 53.4 | 38.1 | 71.6 | 13 | 0.84 | — | 305 |
| 2022Q1 | 223.9 | +8.7 | 46.9 | 32.0 | 72.6 | 11 | 0.68 | — | 864 |
| 2024Q2 | 335.0 | +2.1 | 45.3 | 28.0 | 73.8 | 20 | 0.19 | — | 1,084 |
| 2026Q2 | 450.7 | +0.2 | 41.2 | 26.7 | 75.4 | 34 | 0.02 | — | 1,370 |

### FRSH — Freshworks Inc.

CIK 1544522 · 24 quarters with revenue (2020Q3–2026Q2) · longest complete run 24q · 25% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q3 | 66.2 | — | 51.6 | 20.0 | 80.1 | — | — | — | 56 |
| 2022Q1 | 114.6 | +8.7 | 62.3 | 26.8 | 80.5 | 10 | 0.51 | — | 603 |
| 2023Q3 | 153.6 | +5.8 | 59.1 | 22.7 | 82.9 | 21 | 0.37 | — | 1,165 |
| 2024Q4 | 194.6 | +4.3 | 46.6 | 21.1 | 84.9 | 26 | 0.35 | — | 1,070 |
| 2026Q2 | 237.4 | +3.8 | 44.8 | 18.5 | 84.8 | 28 | 0.33 | — | 664 |

### FSLY — Fastly, Inc.

CIK 1517413 · 33 quarters with revenue (2018Q2–2026Q2) · longest complete run 33q · 25% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2018Q2 | 34.4 | — | 34.8 | 23.5 | 54.4 | — | — | — | 61 |
| 2020Q2 | 74.7 | +18.7 | 33.1 | 22.3 | 60.2 | 7 | 1.90 | 0.19 | 384 |
| 2022Q2 | 102.5 | +0.1 | 45.6 | 37.8 | 44.9 | -16 | 0.01 | 30.66 | 63 |
| 2024Q2 | 132.4 | -0.9 | 40.0 | 26.5 | 55.1 | -5 | -0.09 | — | 312 |
| 2026Q2 | 183.3 | +6.0 | 30.9 | 22.9 | 63.3 | 27 | 0.73 | — | 337 |

### GTLB — Gitlab Inc.

CIK 1653482 · 23 quarters with revenue (2020Q4–2026Q2) · longest complete run 23q · 28% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q4 | 42.2 | — | 82.6 | 45.2 | 89.0 | — | — | — | 283 |
| 2022Q2 | 87.4 | +12.4 | 76.3 | 36.4 | 88.7 | -20 | 0.58 | 0.73 | 887 |
| 2023Q3 | 139.6 | +10.0 | 66.0 | 35.1 | 89.5 | 29 | 0.55 | — | 273 |
| 2024Q4 | 196.0 | +7.4 | 48.6 | 31.3 | 88.7 | -83 | 0.56 | 3.29 | 177 |
| 2026Q2 | 264.2 | +1.4 | 45.2 | 27.1 | 85.8 | 58 | 0.13 | — | 335 |

### INTA — Intapp, Inc.

CIK 1565687 · 24 quarters with revenue (2020Q3–2026Q2) · longest complete run 24q · 28% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q3 | 48.1 | — | 31.9 | 24.8 | 64.7 | — | — | — | 58 |
| 2022Q1 | 69.7 | +7.7 | 41.3 | 29.3 | 62.7 | 5 | 0.69 | 0.10 | 43 |
| 2023Q3 | 101.6 | +7.4 | 33.9 | 28.1 | 68.9 | 19 | 0.81 | — | 142 |
| 2024Q4 | 121.2 | +2.0 | 33.7 | 27.5 | 73.2 | 23 | 0.24 | — | 286 |
| 2026Q2 | 152.5 | +4.4 | 33.7 | 28.2 | 77.6 | 35 | 0.51 | — | 163 |

### KLTR — KALTURA INC

CIK 1432133 · 25 quarters with revenue (2020Q2–2026Q2) · longest complete run 25q · 25% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q2 | 28.7 | — | 22.7 | 22.6 | 62.5 | — | — | — | 23 |
| 2021Q4 | 42.7 | -0.6 | 32.4 | 31.2 | 62.7 | -26 | -0.08 | — | 144 |
| 2023Q2 | 43.9 | +1.4 | 29.0 | 29.6 | 65.2 | -8 | 0.19 | 1.70 | 70 |
| 2024Q4 | 45.6 | +3.0 | 27.1 | 28.4 | 70.8 | 12 | 0.43 | — | 81 |
| 2026Q2 | 46.9 | +5.1 | 27.4 | 27.1 | 73.6 | 1 | 0.71 | 0.22 | 31 |

### KVYO — Klaviyo, Inc.

CIK 1835830 · 16 quarters with revenue (2022Q3–2026Q2) · longest complete run 16q · 27% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2022Q3 | 119.2 | — | 51.6 | 25.3 | 72.6 | — | — | — | 368 |
| 2023Q3 | 175.8 | +6.8 | 95.5 | 80.5 | 66.5 | 20 | 0.27 | — | 723 |
| 2024Q3 | 235.1 | +5.8 | 42.5 | 23.7 | 76.9 | 22 | 0.52 | — | 827 |
| 2025Q2 | 293.1 | +4.7 | 43.2 | 24.7 | 75.7 | 24 | 0.42 | — | 936 |
| 2026Q2 | 370.6 | +3.5 | 37.6 | 24.8 | 72.6 | 29 | 0.36 | — | 833 |

### MDB — MongoDB, Inc.

CIK 1441816 · 35 quarters with revenue (2017Q4–2026Q2) · longest complete run 35q · 29% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q4 | 41.5 | — | 67.6 | 40.0 | 73.3 | — | — | — | 243 |
| 2019Q4 | 109.4 | +10.1 | 52.1 | 36.0 | 70.6 | -0 | 0.71 | 0.29 | 151 |
| 2022Q1 | 266.5 | +17.5 | 54.1 | 33.6 | 71.6 | 26 | 1.10 | — | 474 |
| 2024Q2 | 450.6 | -1.6 | 48.7 | 32.4 | 72.8 | 12 | -0.14 | — | 816 |
| 2026Q2 | 687.6 | -1.1 | 36.3 | 29.1 | 72.2 | 28 | -0.12 | — | 1,036 |

### NET — Cloudflare, Inc.

CIK 1477333 · 32 quarters with revenue (2018Q3–2026Q2) · longest complete run 32q · 28% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2018Q3 | 50.1 | — | 48.9 | 29.6 | 77.6 | — | — | — | 102 |
| 2020Q3 | 114.2 | +14.5 | 49.0 | 27.1 | 76.3 | 16 | 1.03 | — | 112 |
| 2022Q3 | 253.9 | +8.2 | 45.7 | 30.1 | 75.6 | 25 | 0.67 | — | 138 |
| 2024Q2 | 401.0 | +5.9 | 43.5 | 25.6 | 77.8 | 25 | 0.51 | — | 157 |
| 2026Q2 | 696.1 | +8.8 | 39.7 | 22.9 | 71.8 | 26 | 0.82 | — | 1,664 |

### OKTA — Okta, Inc.

CIK 1660134 · 35 quarters with revenue (2017Q4–2026Q2) · longest complete run 35q · 25% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q4 | 66.9 | — | 71.1 | 28.7 | 68.4 | — | — | — | 224 |
| 2019Q4 | 153.0 | +8.9 | 57.0 | 27.3 | 73.3 | 16 | 0.58 | — | 1,366 |
| 2022Q1 | 382.8 | +9.2 | 58.1 | 38.5 | 68.9 | 13 | 0.58 | — | 2,502 |
| 2024Q2 | 617.0 | +2.0 | 38.2 | 26.4 | 76.0 | 37 | 0.20 | — | 2,320 |
| 2026Q2 | 765.0 | +0.5 | 36.3 | 21.3 | 77.8 | 37 | 0.06 | — | 2,589 |

### PATH — UiPath, Inc.

CIK 1734722 · 25 quarters with revenue (2020Q2–2026Q2) · longest complete run 25q · 24% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q2 | 113.1 | — | 80.4 | 23.6 | 87.9 | — | — | — | 297 |
| 2021Q4 | 220.8 | +12.9 | 78.3 | 27.9 | 80.5 | 1 | 0.59 | 0.25 | 1,878 |
| 2023Q2 | 289.6 | -6.1 | 55.4 | 26.0 | 84.9 | 17 | -0.47 | — | 1,781 |
| 2024Q4 | 354.7 | +12.1 | 52.8 | 27.3 | 82.0 | 20 | 0.82 | — | 1,569 |
| 2026Q2 | 418.4 | -13.0 | 40.1 | 22.2 | 81.6 | 18 | -1.49 | — | 1,307 |

### PCTY — Paylocity Holding Corp

CIK 1591698 · 56 quarters with revenue (2012Q3–2026Q2) · longest complete run 54q · 25% derived · 6 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2012Q3 | 15.8 | — | — | — | — | — | — | — | 0 |
| 2016Q1 | 70.6 | +27.9 | 25.1 | 9.6 | 61.4 | — | 3.48 | — | 90 |
| 2019Q3 | 121.9 | +1.2 | 30.3 | 11.8 | 65.0 | 8 | 0.16 | — | 101 |
| 2022Q4 | 256.4 | +4.5 | 29.5 | 16.0 | 64.9 | 29 | 0.58 | — | 120 |
| 2026Q2 | 415.6 | -11.6 | 25.0 | 13.4 | 65.3 | 15 | -2.10 | — | 272 |

### PD — PagerDuty, Inc.

CIK 1568100 · 33 quarters with revenue (2018Q2–2026Q2) · longest complete run 33q · 25% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2018Q2 | 25.0 | — | 53.1 | 30.9 | 84.5 | — | — | — | 40 |
| 2020Q2 | 49.8 | +8.4 | 53.7 | 30.2 | 86.0 | 8 | 0.58 | 0.01 | 351 |
| 2022Q2 | 85.4 | +8.7 | 53.4 | 36.7 | 81.6 | 5 | 0.60 | 0.11 | 467 |
| 2024Q2 | 111.2 | +0.0 | 43.6 | 33.8 | 82.6 | 26 | 0.00 | — | 593 |
| 2026Q2 | 121.0 | -3.1 | 32.7 | 24.8 | 84.3 | 34 | -0.39 | — | 444 |

### QLYS — QUALYS, INC.

CIK 1107843 · 62 quarters with revenue (2011Q1–2026Q2) · longest complete run 62q · 19% derived · 9 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2011Q1 | 17.7 | — | 39.6 | 26.9 | 83.8 | — | — | — | 0 |
| 2014Q4 | 36.6 | +6.5 | 32.6 | 19.2 | 79.4 | 52 | 0.75 | — | 127 |
| 2018Q3 | 71.7 | +5.1 | 21.6 | 17.4 | 77.0 | 49 | 0.91 | — | 327 |
| 2022Q3 | 125.6 | +4.7 | 19.9 | 20.3 | 79.3 | 38 | 0.91 | — | 194 |
| 2026Q2 | 182.2 | +3.7 | 22.5 | 16.8 | 83.4 | 36 | 0.64 | — | 250 |

### RNG — RingCentral, Inc.

CIK 1384905 · 38 quarters with revenue (2017Q1–2026Q2) · longest complete run 38q · 25% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q1 | 112.2 | — | 48.3 | 15.2 | 75.7 | — | — | — | 150 |
| 2019Q2 | 215.2 | +6.8 | 48.1 | 15.2 | 75.1 | 18 | 0.53 | — | 568 |
| 2021Q3 | 414.6 | +9.3 | 54.3 | 20.3 | 73.4 | 20 | 0.63 | — | 345 |
| 2024Q1 | 584.2 | +2.3 | 46.7 | 13.8 | 70.8 | 19 | 0.19 | — | 203 |
| 2026Q2 | 657.0 | +2.0 | 41.8 | 12.6 | 71.9 | 33 | 0.19 | — | 112 |

### RPD — Rapid7, Inc.

CIK 1560327 · 49 quarters with revenue (2014Q2–2026Q2) · longest complete run 49q · 23% derived · 4 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2014Q2 | 17.9 | — | 64.7 | 35.4 | 76.5 | — | — | — | 0 |
| 2017Q2 | 47.4 | +4.9 | 57.2 | 25.0 | 72.4 | — | 0.32 | — | 84 |
| 2020Q2 | 98.9 | +4.8 | 45.5 | 26.4 | 70.6 | 5 | 0.41 | — | 316 |
| 2023Q2 | 190.4 | +4.0 | 43.8 | 26.9 | 69.5 | 20 | 0.35 | — | 294 |
| 2026Q2 | 210.9 | +0.6 | 36.1 | 22.3 | 68.9 | 18 | 0.06 | — | 703 |

### S — SentinelOne, Inc.

CIK 1583708 · 24 quarters with revenue (2020Q3–2026Q2) · longest complete run 24q · 25% derived · 2 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q3 | 20.7 | — | 78.9 | 65.2 | 63.5 | — | — | — | 174 |
| 2022Q1 | 65.6 | +17.2 | 64.2 | 65.0 | 63.1 | 9 | 0.91 | 0.15 | 1,670 |
| 2023Q3 | 149.4 | +12.0 | 65.8 | 36.2 | 70.1 | 4 | 0.65 | 0.19 | 732 |
| 2024Q4 | 210.6 | +5.9 | 58.7 | 33.4 | 74.7 | 2 | 0.38 | 0.15 | 660 |
| 2026Q2 | 276.7 | +2.0 | 47.8 | 34.6 | 71.8 | 16 | 0.17 | — | 657 |

### SPT — Sprout Social, Inc.

CIK 1517375 · 30 quarters with revenue (2019Q1–2026Q2) · longest complete run 30q · 27% derived · 2 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2019Q1 | 23.4 | — | 44.7 | 27.2 | 75.0 | — | — | — | 22 |
| 2020Q4 | 37.3 | +10.9 | 43.6 | 20.9 | 74.3 | 10 | 0.90 | 0.01 | 164 |
| 2022Q3 | 65.3 | +6.3 | 49.6 | 24.9 | 76.6 | 8 | 0.48 | — | 95 |
| 2024Q3 | 102.6 | +3.3 | 46.3 | 25.6 | 77.4 | 12 | 0.27 | — | 83 |
| 2026Q2 | 123.8 | +1.9 | 38.3 | 21.5 | 77.6 | 9 | 0.20 | — | 120 |

### TENB — Tenable Holdings, Inc.

CIK 1660280 · 38 quarters with revenue (2017Q1–2026Q2) · longest complete run 38q · 17% derived · 0 quarters under $25M revenue

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q1 | 40.5 | — | 64.6 | 30.8 | 89.0 | — | — | — | 0 |
| 2019Q2 | 85.4 | +6.3 | 65.6 | 25.4 | 83.7 | 4 | 0.36 | 0.10 | 297 |
| 2021Q3 | 138.7 | +6.5 | 49.3 | 22.1 | 80.5 | 21 | 0.49 | — | 652 |
| 2024Q1 | 216.0 | +1.2 | 46.2 | 20.2 | 77.3 | 25 | 0.11 | — | 511 |
| 2026Q2 | 268.5 | +2.5 | 39.4 | 21.2 | 77.5 | 19 | 0.24 | — | 298 |

### TWLO — TWILIO INC

CIK 1447669 · 45 quarters with revenue (2015Q2–2026Q2) · longest complete run 45q · 27% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2015Q2 | 38.0 | — | 37.3 | 24.7 | 55.7 | — | — | — | 122 |
| 2018Q1 | 129.1 | +12.0 | 25.4 | 29.1 | 53.9 | 25 | 1.69 | — | 118 |
| 2020Q4 | 548.1 | +22.3 | 32.8 | 29.0 | 51.5 | 25 | 2.23 | — | 934 |
| 2023Q3 | 1,033.7 | -0.4 | 25.4 | 23.4 | 50.0 | 20 | -0.06 | — | 678 |
| 2026Q2 | 1,499.1 | +6.6 | 14.5 | 18.2 | 48.4 | 31 | 1.70 | — | 823 |

### WEAV — Weave Communications, Inc.

CIK 1609151 · 24 quarters with revenue (2020Q3–2026Q2) · longest complete run 24q · 28% derived · 2 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q3 | 21.4 | — | 41.5 | 25.3 | 56.9 | — | — | — | 59 |
| 2022Q1 | 33.3 | +4.5 | 48.7 | 21.7 | 58.7 | -8 | 0.35 | 0.73 | 129 |
| 2023Q3 | 43.5 | +4.5 | 40.9 | 19.8 | 68.7 | 12 | 0.42 | — | 63 |
| 2024Q4 | 54.2 | +3.4 | 40.5 | 19.9 | 72.1 | 16 | 0.33 | — | 52 |
| 2026Q2 | 67.5 | +3.1 | 40.6 | 17.8 | 72.0 | 18 | 0.30 | — | 48 |

### WK — WORKIVA INC

CIK 1445305 · 38 quarters with revenue (2017Q1–2026Q2) · longest complete run 38q · 11% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q1 | 51.9 | — | 36.1 | 29.9 | 72.6 | — | — | — | 52 |
| 2019Q2 | 73.5 | +5.0 | 38.4 | 29.7 | 71.9 | 31 | 0.50 | — | 95 |
| 2021Q3 | 112.7 | +6.7 | 40.8 | 26.5 | 76.6 | 21 | 0.62 | — | 291 |
| 2024Q1 | 175.7 | +5.4 | 47.0 | 25.9 | 76.4 | 20 | 0.44 | — | 296 |
| 2026Q2 | 255.3 | +3.2 | 42.7 | 22.5 | 80.4 | 34 | 0.29 | — | 252 |

### YEXT — Yext, Inc.

CIK 1614178 · 41 quarters with revenue (2016Q2–2026Q2) · longest complete run 41q · 26% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2016Q2 | 27.1 | — | 62.1 | 17.6 | 67.4 | — | — | — | 37 |
| 2018Q4 | 58.6 | +6.7 | 74.6 | 15.6 | 74.6 | -32 | 0.34 | 1.53 | 28 |
| 2021Q2 | 92.0 | -0.2 | 60.0 | 15.1 | 76.2 | 38 | -0.01 | — | 272 |
| 2023Q4 | 101.2 | -1.4 | 44.8 | 18.1 | 78.2 | -3 | -0.13 | — | 182 |
| 2026Q2 | 107.9 | -3.6 | 27.2 | 19.9 | 72.9 | 31 | -0.56 | — | 92 |

### ZETA — Zeta Global Holdings Corp.

CIK 1851003 · 25 quarters with revenue (2020Q2–2026Q2) · longest complete run 25q · 27% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2020Q2 | 77.1 | — | 21.8 | 10.6 | 62.0 | — | — | — | 41 |
| 2021Q4 | 134.8 | +17.1 | 48.5 | 10.5 | 63.7 | 33 | 1.21 | — | 104 |
| 2023Q2 | 171.8 | +9.0 | 42.2 | 10.1 | 63.9 | 21 | 0.78 | — | 117 |
| 2024Q4 | 314.7 | +17.3 | 26.4 | 7.7 | 60.0 | 31 | 2.24 | — | 366 |
| 2026Q2 | 442.8 | +11.7 | 23.5 | 9.5 | 59.1 | 27 | 1.79 | — | 310 |

### ZM — Zoom Communications, Inc.

CIK 1585521 · 33 quarters with revenue (2018Q2–2026Q2) · longest complete run 33q · 26% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2018Q2 | 60.1 | — | 60.4 | 10.4 | 80.6 | — | — | — | 31 |
| 2020Q2 | 328.2 | +74.3 | 37.0 | 8.0 | 68.4 | 153 | 4.60 | — | 489 |
| 2022Q2 | 1,073.8 | +0.2 | 33.8 | 13.4 | 75.6 | 49 | 0.03 | — | 1,407 |
| 2024Q2 | 1,141.2 | -0.5 | 30.5 | 18.0 | 76.1 | 51 | -0.06 | — | 1,886 |
| 2026Q2 | 1,239.0 | -0.6 | 26.6 | 18.4 | 77.9 | 41 | -0.10 | — | 891 |

### ZS — Zscaler, Inc.

CIK 1713683 · 37 quarters with revenue (2017Q2–2026Q2) · longest complete run 37q · 28% derived · 0 quarters under $25M revenue · **gaps: short_term_investments**

| quarter | revenue $M | QoQ % | S&M % | R&D % | GM % | R40 | magic | burn× | cash $M |
|---|---|---|---|---|---|---|---|---|---|
| 2017Q2 | 33.0 | — | 62.8 | 23.6 | 78.8 | — | — | — | 87 |
| 2019Q3 | 86.1 | +8.8 | 57.3 | 20.0 | 79.9 | 30 | 0.57 | — | 78 |
| 2021Q4 | 230.5 | +17.0 | 66.7 | 28.3 | 77.4 | 57 | 0.87 | — | 372 |
| 2024Q1 | 525.0 | +5.7 | 52.7 | 23.3 | 77.7 | 33 | 0.41 | — | 1,439 |
| 2026Q2 | 850.5 | +4.3 | 43.7 | 27.3 | 77.3 | 28 | 0.37 | — | 982 |
