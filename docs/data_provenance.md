# Data provenance and change record

What external data entered this project, how it was obtained, exactly where it landed in
the code, and what measurably changed as a result.

Scope: branch `founder-calibration`, through commit `0eafa08` (2026-08-25).
Every claim below is checkable with a command in §7.

**Reading guide.** The project distinguishes three kinds of number, and the distinction is
enforced in code, not just documented:

| Kind | Meaning | Example |
|---|---|---|
| **observed** | printed in a cited external source | marketing = 8% of ARR |
| **assumed** | chosen by us, no source | `SATURATION_ACQUISITION_RATE = 0.20` |
| **unidentified** | no dataset exists; must not be estimated | price elasticity |

`calibration/__init__.py` returns `None` rather than a default for anything not observed,
so an uncalibrated parameter is visible as uncalibrated instead of silently passing as a
measurement.

---

## 1. Provenance summary

| Parameter | Status | Source | Obtained | Lands in |
|---|---|---|---|---|
| Discretionary spend ceiling | **observed** | SaaS Capital 2026 | 2026-08-25, public web page | `advise_service.py:148` |
| Marketing saturation (`gamma`, `beta`) | **derived, not sourced** | dimensional fix, no data | 2026-08-25 | `business_logic.py:106` |
| Saturation acquisition rate | **assumed** | none | — | `business_logic.py:103` |
| Spend-ceiling multiple | **assumed** (product judgement) | none | — | `advise_service.py:73` |
| Churn by ARPA band | **pending** | ChartMogul (email gate) | not yet obtained | `bands.json` → all `null` |
| Price → churn coupling | **pending** | ChartMogul retention | not yet obtained | not yet wired |
| Price elasticity | **unidentified** | no public dataset exists | — | `bands.json` → `unidentified` |

Only **one** external source has been used so far. Everything else is either pending,
assumed-and-labelled, or declared unobtainable.

---

## 2. Source 1 — SaaS Capital (the only external data currently in the build)

**Publisher.** SaaS Capital
**Report.** 2026 Spending Benchmarks for Private B2B SaaS Companies (15th annual survey)
**URL.** `https://www.saas-capital.com/blog-posts/spending-benchmarks-for-private-b2b-saas-companies/`
**Access.** Public blog page. No email gate, no account, no scraping — a single HTTP fetch
of one page.
**Date obtained.** 2026-08-25
**Method.** `WebFetch` tool: fetches the URL, converts to markdown, and answers a prompt
against it using a model. **This is a model-mediated extraction, not a manual read.**
**Population.** "more than 1,000 SaaS companies", survey completed March 2026.

### Figures taken

| Figure | Value | Section of page |
|---|---|---|
| Total spend, bootstrapped | 96% of ARR | How Much Do Private SaaS Companies Spend? |
| Total spend, equity-backed | 101% of ARR | same |
| Marketing | 8% of ARR | Private SaaS Company Spending by Department |
| R&D | 22% of ARR | same |
| Sales | 15% of ARR | same |
| Customer support/success | 9% of ARR | same |
| G&A | 15% of ARR | same |
| Hosting / DevOps | 5% / 4% of ARR | same |
| Pro services / other CoGS | 5% / 3% of ARR | same |
| $3–5M ARR band | 12% selling, 8% marketing, 24% R&D, 15% G&A | SaaS Spending by ARR Levels |

Stored verbatim in `calibration/bands.json` under `spend_benchmarks`, each with
`confidence: "observed"` and a `source` block naming publisher, report, year, section and
n.

### Unit conversion, stated explicitly

Percent-of-ARR converts directly to percent-of-MRR for a **monthly** figure:

```
annual marketing spend = 0.08 × ARR = 0.08 × 12 × MRR
monthly marketing spend = 0.08 × MRR
```

So the published 8% marketing + 22% R&D means a median company spends **30% of MRR per
month** on marketing and product combined. This conversion is recorded in `bands.json`
`notes` and in the loader docstring. It is the single most likely place for a silent error,
which is why it is written down in three places.

### ⚠ Verification gap — action required

The extraction methodology in `calibration_acquisition.md` §3 requires **two passes**: an
extraction pass and a separate verification pass in a fresh context. These figures went
through **one pass only**, because `WebFetch` performs a single model-mediated read.

**Before these numbers are defended as sourced, open the URL and confirm each figure in
the table above against the page.** They currently gate a live product guard. Single-pass
extraction has a real error rate on exactly this content — banded percentages in small
multiples — and the failure mode is silent.

### Licensing position

Fitting shipped constants from published benchmark tables is normal practice; republishing
the tables is not. This document cites specific figures with full attribution for an
internal audit record. The product does not display these tables, does not ship the source
material, and does not reproduce their charts. `*.pdf` and `calibration/sources/` are
gitignored so source material cannot be committed by accident.

---

## 3. What is NOT from data

Recorded here so no reader mistakes them for measurements.

### `SATURATION_ACQUISITION_RATE = 0.20` — assumed
`env/business_logic.py:103`. The share of its existing customer base a company could
plausibly add in one month at full marketing saturation. No public dataset fixes this.
It is the one free parameter in the reparameterised marketing curve, and it is labelled
assumed in code, in `bands.json`, and here.
*Check available:* MicroConf's growth-by-MRR-band data (source 6, pending) would say
whether 20%/month at saturation is plausible for sub-$1M ARR companies.

### `DISCRETIONARY_SPEND_MEDIAN_MULTIPLE = 2.0` — product judgement
`backend/advise_service.py:73`. How far above the published median a plan may sit before
being clamped. The **median** is observed; this **multiple** is not. It is a decision about
how prescriptive the product should be, and is stated as such in the code comment.

### Price elasticity — unidentified, deliberately not estimated
`apply_pricing_effect` draws `uniform(-0.9, -0.2)`. That range is folk knowledge, not
measurement; no public SaaS price-elasticity dataset exists. Recorded in `bands.json`
under `unidentified` with the reason and what would unblock it (first-party A/B price
tests via the `decisions` table). **It has not been replaced with a plausible-looking
number**, which would have been the easy and wrong move.

---

## 4. Changes made as a result

### 4.1 Discretionary spend ceiling — traced to data

- **Commit** `0eafa08`
- **Code** `backend/advise_service.py:148` `_apply_spend_ceiling()`
- **Chain** SaaS Capital page → `bands.json` `spend_benchmarks.by_department_pct_of_arr`
  → `calibration.discretionary_spend_pct_of_mrr()` → ceiling = `MRR × 30% × 2.0`
- **Behaviour** plans above the ceiling scale marketing and R&D down proportionally, so
  the board's product/marketing balance is preserved and only the magnitude changes
- **Fails safe** if the benchmark is ever absent, `is_observed` is false and **nothing is
  capped** — an uncalibrated guard is worse than no guard
- **Audit trail** every analysis writes `trace.spend_ceiling` with the ceiling, whether it
  bound, the scale factor, the median used, and the citation string

### 4.2 Marketing saturation — a fix the data did not provide

This change is **not** traceable to a dataset, and the distinction matters.

`gamma = uniform(15_000, 50_000)` was a spend level with no reference to who was being
bought — dimensionally wrong, not merely uncalibrated. No public dataset fixes it: open
marketing-mix datasets (Robyn, Meridian, the arXiv synthetic generator) are synthetic and
retail-shaped, so fitting to them would substitute another model's assumptions for ours.

Reparameterised at `env/business_logic.py:106` so both anchors scale with the company:

```
acquirable = current_customers × SATURATION_ACQUISITION_RATE
beta       = acquirable × price          # max new MRR per month
gamma      = (acquirable / 2) × CAC      # spend that buys half of them
```

Opt-in via `initial_config={"scale_aware_marketing": True}`; **off by default**, so prior
runs reproduce exactly.

---

## 5. Measured impact

### 5.1 Marketing response — new MRR per dollar spent

Mean of 60 draws per cell, seeded. Response as a share of MRR, at equivalent *relative*
spend (brand channel, spend ≈ 25% of MRR):

| Company | Before | After |
|---|---|---|
| $12k MRR | 33.4% | 21.4% |
| $50k MRR | 25.5% | 21.3% |
| $200k MRR | 21.8% | 21.3% |

Before, identical relative spend produced materially different returns purely because of
company size. After, it does not. The dead zone is also gone: $12k-MRR brand spend at
$1,000 moved 0.9% → 2.6%, and the inverse distortion — small-company ppc paying 51.7% of
MRR at $7,500 — fell to 10.8%.

### 5.2 Advice-quality audit

`advice_audit.py`, six company profiles, real advise path, no mocks.

| Build | Violations |
|---|---|
| Before any fixes | 9 |
| After Oracle prompt carries burn/runway | 6 |
| After hiring guard on final action | 2 |
| After sourced spend ceiling (`0eafa08`) | **1** |

The one remaining: `pre_revenue` reads `LOW` risk at 4.0 months runway.

**What the data revealed about the engine.** Before the ceiling, the board recommended
26% of MRR on marketing and 40% on R&D — **66% against a 30% published median**. The
ceiling now binds on 4 of 6 profiles, all landing at exactly 60%. That is evidence the
over-spending tendency is being *clamped*, not *cured*: the proposal generator still wants
to exceed 2× the median in most cases. Per-ARPA targets from ChartMogul would let the
generator aim correctly instead of being corrected after the fact.

### 5.3 Confidence limits on these numbers

- Audit figures are **one run per profile**. LLM output varies; `advice_audit.py --repeat N`
  measures the spread and **has not been run**. Treat the direction as established and the
  magnitudes as provisional.
- §5.1 figures are seeded and reproducible exactly.
- 36/36 tests pass at `0eafa08`.

---

## 6. Integrity controls

Mechanisms that stop an unsourced number from passing as sourced:

1. **No silent defaults.** `calibration/__init__.py` returns `Calibrated(value=None)` for
   anything absent. Callers must handle it; `_apply_spend_ceiling` declines to cap rather
   than invent a ceiling.
2. **`is_observed` gates use.** A value is usable only if printed for that exact band.
3. **Provenance travels with the value.** `Calibrated.citation()` carries publisher,
   report and year to the UI.
4. **Half a benchmark is not a benchmark.** `discretionary_spend_pct_of_mrr()` requires
   *both* marketing and R&D to be observed, or returns `None`.
5. **Every engine change is default-off.** `scale_aware_marketing`,
   `include_burn_context`, `hiring_runway_guard_months`, `scale_absolutes` all default to
   research behaviour, so prior results remain reproducible.
6. **The same rule already applies to the causal graph.** Seeded `MAY_CAUSE` priors are
   rendered as "the board's working assumption", distinct from observed `CONFIRMED_CAUSE`
   edges — corrected in `109254a` after being found to overstate them.

---

## 7. How to verify every claim here

```bash
git log --oneline founder-calibration          # commits cited above
git show 0eafa08 --stat                        # what the calibration commit touched
```

```bash
venv\Scripts\python.exe -c "import calibration as c; d=c.discretionary_spend_pct_of_mrr(); print(d.value, d.confidence, d.citation())"
```

```bash
venv\Scripts\python.exe -c "import calibration as c; b=c.band_metric(40,'monthly_gross_mrr_churn'); print(b.value, b.confidence)"
```

Expect `30.0 observed SaaS Capital, …` and `None assumed` — the second proving unfilled
bands stay empty rather than defaulting.

```bash
venv\Scripts\python.exe advice_audit.py
venv\Scripts\python.exe -m pytest tests/ -q
```

The SaaS Capital page itself is public and can be opened directly to check §2.

---

## 8. Open provenance gaps

| Gap | Blocked on | Consequence while open |
|---|---|---|
| SaaS Capital figures single-pass | your manual check against the page | a live guard rests on unverified extraction |
| Churn by ARPA band | 3 ChartMogul PDFs (email gate) | `bands.json` bands all `null`; churn stays hand-tuned |
| Price → churn coupling | ChartMogul retention report | price changes still touch churn zero |
| `SATURATION_ACQUISITION_RATE` sanity check | MicroConf PDF | the one free parameter stays assumed |
| Real founder-scale rows | Flippa ToS decision | memory corpus stays simulator-derived |
| Price elasticity | first-party A/B tests | lever unidentified; must not be presented as estimated |
| Audit variance | `--repeat N` run | magnitudes provisional |

Two of these are not data problems and will not close by acquiring anything: price
elasticity requires running experiments, and audit variance requires spending compute.
