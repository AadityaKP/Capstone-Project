# Data provenance and change record

What external data entered this project, how it was obtained, exactly where it landed in
the code, and what measurably changed as a result.

Scope: branch `founder-calibration`, 2026-08-25.

**Revision 2.** Source PDFs obtained and a verification pass run against them. One
revision-1 figure did not survive and has been withdrawn (§2.3). ChartMogul retention data
added (§2b). ChartMogul scraping was declined on Terms-of-Service grounds (§2c).

**Reading guide.** The project distinguishes three kinds of number, and the distinction is
enforced in code, not just documented:

| Kind | Meaning | Example |
|---|---|---|
| **observed** | printed in a cited source page | gross retention 64% at $25–100 ARPA |
| **assumed** | chosen by us, no source | `SATURATION_ACQUISITION_RATE = 0.20` |
| **unidentified** | no dataset exists; must not be estimated | price elasticity |

`calibration/__init__.py` returns `None` rather than a default for anything not observed,
so an uncalibrated parameter is visible as uncalibrated instead of silently passing as a
measurement.

---

## 1. Provenance summary

| Parameter | Status | Source | Verified | Lands in |
|---|---|---|---|---|
| Total spend % of ARR | **observed** | SaaS Capital 2026, p2 | yes — 2nd pass vs PDF | `bands.json` |
| Spend by department | **observed, $3–5M ARR band only** | SaaS Capital 2026, p4 | yes — 2nd pass vs PDF | spend ceiling |
| ~~Overall department medians~~ | **WITHDRAWN** | — | **failed verification** | removed |
| Retention / churn by ARPA | **observed** | ChartMogul 2023, p30 | yes — read from PDF text | Oracle prompt, trace |
| Marketing saturation | **derived, not sourced** | dimensional fix | n/a | `business_logic.py:106` |
| Saturation acquisition rate | **assumed** | none | n/a | `business_logic.py:103` |
| Spend-ceiling multiple | **assumed** (product judgement) | none | n/a | `advise_service.py:73` |
| Price → churn coupling | **pending** | report not supplied | — | not wired |
| Price elasticity | **unidentified** | no public dataset exists | n/a | `bands.json` |

---

## 2. Source 1 — SaaS Capital

**Publisher.** SaaS Capital
**Report.** 2026 Spending Benchmarks for Private B2B SaaS Companies (15th annual survey)
**Population.** "more than 1,000 SaaS companies", survey completed March 2026
**First access.** 2026-08-25, public blog page, single HTTP fetch, no gate, no scraping
**Verification.** 2026-08-25, against the report PDF supplied by the project owner

### 2.1 Confirmed

- **96%** of ARR total median spend (bootstrapped), **101%** (equity-backed) — page 2
- **$3–5M ARR band:** 5% hosting, 3% DevOps, 5% pro services CoGS, 3.5% other CoGS,
  10% customer support/success, 12% selling, **8% marketing**, **24% R&D**, 15% G&A — page 4

### 2.2 Unit conversion, stated explicitly

Percent-of-ARR converts directly to percent-of-MRR for a **monthly** figure:

```
annual marketing spend  = 0.08 × ARR = 0.08 × 12 × MRR
monthly marketing spend = 0.08 × MRR
```

So marketing 8% + R&D 24% means a median $3–5M ARR company spends **32% of MRR per month**
on marketing and product combined.

### 2.3 Verification pass — one figure withdrawn

The second pass used `pypdf` text extraction rather than a model read, so the comparison is
mechanical rather than another judgement call.

**Withdrawn:** the *overall* department median table reported in revision 1 — sales 15%,
marketing 8%, customer support 9%, R&D 22%, G&A 15%, hosting 5%, DevOps 4%, pro services
5%, other CoGS 3%.

**These figures do not appear anywhere in the PDF's text.** Pages 5–10 are chart images
titled "Spend by Company Size" with no extractable content. They came from a single
model-mediated read of the web page and could not be confirmed, so they have been removed
from `bands.json`.

This is precisely the failure the two-pass requirement exists to catch, and it had reached
a live product guard. Recorded here rather than quietly corrected.

**Consequence.** The spend ceiling now derives from the $3–5M ARR band: **32% of MRR**,
replacing the withdrawn 30%. That band is two orders of magnitude larger than the target
user, so every analysis records `trace.spend_ceiling.extrapolated = true` alongside
`source_band: "$3-5M ARR"`. The number is observed; its application to a founder is
extrapolation, and the two are not conflated.

### 2.4 Licensing position

Fitting shipped constants from published benchmark tables is normal practice; republishing
the tables is not. This document cites specific figures with full attribution for an
internal audit record. The product does not display these tables, does not ship the source
material, and does not reproduce their charts. `*.pdf` and `calibration/sources/` are
gitignored so source material cannot be committed by accident.

---

## 2b. Source 2 — ChartMogul (retention by ARPA band)

**Publisher.** ChartMogul
**Reports.** SaaS Benchmarks Report, 2023 edition (60pp); SaaS Retention Report: The New
Normal, 2024, 2nd edition (28pp, 2,500+ SaaS businesses)
**Access.** PDFs supplied by the project owner, obtained through ChartMogul's own
email-gated download. **Not scraped** — see §2c.
**Date read.** 2026-08-25
**Method.** `pypdf` text extraction from local files. No model intermediary, so there is
no extraction error surface requiring a second pass.

### 2b.1 Figures taken — 2023 Benchmarks Report, page 30

Table "Retention benchmarks by ARPA per month range", band headers printed in order
(`<$25`, `$25-100`, `$100-250`, `$250-500`, `$500-1k`, `>$1k`), four percentile rows each
for net, gross and customer retention — 72 values, stored verbatim in `bands.json` under
`arpa_bands` as **annual retention percentages exactly as printed**.

### 2b.2 The annual → monthly conversion

Nothing is converted in the data file. The conversion lives in one inspectable function,
`calibration.annual_retention_to_monthly_churn()`:

```
monthly_churn = 1 − (annual_retention)^(1/12)
```

Compounding, not division. **54% annual retention is 5.01% monthly churn**; the naive
`(100 − 54)/12` gives 3.83%, understating churn by 31%. This is the error the acquisition
methodology singled out as most likely, which is why the raw unit is preserved in the data
file and the conversion has exactly one call site.

Derived medians:

| ARPA band | monthly logo churn | monthly gross MRR churn |
|---|---|---|
| <$25 | 5.01% | 5.61% |
| $25–100 | 3.40% | 3.65% |
| $100–250 | 2.70% | 2.70% |
| $250–500 | 2.59% | 2.48% |
| $500–1k | 2.15% | 1.95% |
| >$1k | 1.95% | 1.64% |

### 2b.3 Band-boundary deviation from the target schema

The acquisition plan specified `arpa_250_1000` as one band. ChartMogul publishes `$250-500`
and `$500-1k` separately. Merging two published medians would produce a number no source
prints, so **the source's segmentation wins** and the schema was widened to six bands.

### 2b.4 Age caveat

The retention table is the **2023** edition. The 2024 retention report states NRR is
declining across all ARR segments since, so these figures are likely optimistic for 2026.
Its own ARPA chart (page 17) is an image with no extractable text and could not be used to
update them. Recorded in `bands.json` under `arpa_bands.caveat`.

### 2b.5 A discrepancy inside the source

Page 34 prose says top-quartile <$25 ARPA net retention "only hit 70%", while the table on
page 30 prints **68%**. The table is treated as authoritative. Noted so a reader comparing
the report to our data is not surprised.

### 2b.6 What could NOT be extracted

- NRR by ARPA from the 2024 report — chart image, page 17
- Price→churn coupling (GRR/NRR by price point) — that figure is from the *AI Churn Wave*
  report, which was not supplied

---

## 2c. ChartMogul Terms of Service — scraping declined

Checked `https://chartmogul.com/terms/` on 2026-08-25, before any automated collection.
Clause 8 (Forbidden Use):

> "use any robot, spider, site search/retrieval application, or other automated device,
> process, or means to access, retrieve, scrape, or index any portion of ChartMogul or its
> content"

**No scraping was performed.** The prohibition is explicit and unconditional. All
ChartMogul data in §2b comes from PDFs the project owner downloaded through ChartMogul's
own distribution route, read locally.

Reading a legitimately obtained file is a distinct act from automated site access. Citing
figures from it for internal calibration, with attribution and without republishing the
tables, sits on the same footing as the SaaS Capital material (§2.4).

---

## 3. What is NOT from data

### `SATURATION_ACQUISITION_RATE = 0.20` — assumed
`env/business_logic.py:103`. The share of its existing customer base a company could
plausibly add in one month at full marketing saturation. No public dataset fixes it. It is
the one free parameter in the reparameterised marketing curve, labelled assumed in code, in
`bands.json`, and here.
*Check still available:* MicroConf's growth-by-MRR-band data would say whether 20%/month at
saturation is plausible for sub-$1M ARR companies. Not supplied.

### `DISCRETIONARY_SPEND_MEDIAN_MULTIPLE = 2.0` — product judgement
`backend/advise_service.py:73`. How far above the published median a plan may sit before
being clamped. The **median** is observed; this **multiple** is not.

### Price elasticity — unidentified, deliberately not estimated
`apply_pricing_effect` draws `uniform(-0.9, -0.2)`. That range is folk knowledge; no public
SaaS price-elasticity dataset exists. Recorded in `bands.json` under `unidentified` with
what would unblock it. **It has not been replaced with a plausible-looking number.**

---

## 4. Changes made as a result

### 4.1 Churn benchmark in the Oracle prompt — traced to ChartMogul

- **Chain** ChartMogul p30 → `bands.json` `arpa_bands` → `annual_retention_to_monthly_churn()`
  → `Oracle(churn_benchmark_pct=…)` → prompt line
- **Why it matters** churn in isolation is uninterpretable: 5%/month is poor at $500 ARPA
  and unremarkable at $10. The prompt now carries the published median for the company's
  own price point, turning an absolute number into a judgement.
- **Fails safe** `None` when no band covers the company; the prompt line is simply omitted
  rather than showing an invented comparison
- **Audit trail** `trace.churn_benchmark` records company rate, median, band, citation and
  the annual→monthly derivation

### 4.2 Discretionary spend ceiling — traced to SaaS Capital

- **Code** `backend/advise_service.py` `_apply_spend_ceiling()`
- **Ceiling** `MRR × 32% × 2.0`
- **Behaviour** plans above it scale marketing and R&D down proportionally, preserving the
  board's balance and correcting only magnitude
- **Fails safe** if the benchmark is absent, nothing is capped — an uncalibrated guard is
  worse than no guard
- **Honesty** records `extrapolated: true` and `source_band` on every analysis

### 4.3 Marketing saturation — a fix the data did not provide

Not traceable to a dataset, and the distinction matters. `gamma = uniform(15_000, 50_000)`
was a spend level with no reference to who was being bought — dimensionally wrong, not
merely uncalibrated. Open marketing-mix datasets are synthetic and retail-shaped, so
fitting to them would substitute another model's assumptions for ours.

Reparameterised at `env/business_logic.py:106` so both anchors scale with the company:

```
acquirable = current_customers × SATURATION_ACQUISITION_RATE
beta       = acquirable × price          # max new MRR per month
gamma      = (acquirable / 2) × CAC      # spend that buys half of them
```

Opt-in via `initial_config={"scale_aware_marketing": True}`; **off by default**.

---

## 5. Measured impact

### 5.1 Marketing response

Mean of 60 seeded draws per cell. Response as a share of MRR at equivalent *relative* spend
(brand, ≈25% of MRR):

| Company | Before | After |
|---|---|---|
| $12k MRR | 33.4% | 21.4% |
| $50k MRR | 25.5% | 21.3% |
| $200k MRR | 21.8% | 21.3% |

Identical relative spend previously produced materially different returns purely because of
company size. The dead zone is also gone: $12k-MRR brand spend at $1,000 moved 0.9% → 2.6%,
and the inverse distortion (small-company ppc paying 51.7% of MRR at $7,500) fell to 10.8%.

### 5.2 Advice-quality audit

`advice_audit.py`, six profiles, real advise path, no mocks.

| Build | Violations |
|---|---|
| Before any fixes | 9 |
| Oracle prompt carries burn/runway | 6 |
| Hiring guard on final action | 2 |
| Sourced spend ceiling | 1 |
| **Churn benchmark in prompt** | **0** |

The published churn benchmark closed the final violation without a hard-coded rule. The
planned fix had been a risk floor ("under 6 months runway cannot be LOW"); given the
company's churn *and* what companies at its price point actually achieve, the model
reaches the right read unaided.

Risk now discriminates, which it did not at the start of this work:

| Profile | Runway | Risk before | Risk after |
|---|---|---|---|
| cash_crisis | 1.1 mo | LOW | **HIGH** |
| pre_revenue | 4.0 mo | LOW | **MEDIUM** |
| small_struggling | 7.5 mo | LOW | **MEDIUM** |
| small_healthy | 50 mo | LOW | LOW |
| scaling | 100 mo | LOW | LOW |

Every profile returned `LOW` before, including the company with 1.1 months of cash.

### 5.3 What the data revealed about the engine

Before the ceiling, the board recommended 26% of MRR on marketing and 40% on R&D — **66%
against a 32% published median**. The ceiling binds on most profiles, which is evidence the
over-spending tendency is being *clamped*, not *cured*.

### 5.4 Reproducibility — measured, not assumed

`advice_audit.py --repeat` was run after the calibration work: **5 repeats of
`small_struggling` and 4 of `cash_crisis`, nine runs total.**

| Profile | n | risk | marketing % | R&D % | hires |
|---|---|---|---|---|---|
| small_struggling | 5 | {MEDIUM} | {15} | {40} | {0} |
| cash_crisis | 4 | {HIGH} | {0} | {0} | {0} |

Every field collapses to a single value. The Oracle runs at `temperature: 0` and retrieval
is deterministic, so identical inputs produce identical advice. **The audit magnitudes are
reproducible, not indicative** — an earlier caveat in this document said otherwise and is
withdrawn.

Scope of the claim: determinism was measured within one process against one Ollama
instance. It is not a guarantee across model reloads or a different Ollama build.

§5.1 marketing figures are seeded and reproducible exactly.

### 5.5 Test coverage

The calibration store and the founder guards had no tests while already gating
founder-facing advice. 48 were added (suite 36 → **84**), all deterministic, running in
about 1.3s:

- the annual→monthly conversion pinned against the naive division, plus a round-trip check
  that surviving 12 months of the derived rate reproduces the published annual figure
- a **monotonicity check across the six ARPA bands**, which would catch a mis-transcribed
  row in the source table
- every fails-safe path: absent values return `None`, half a benchmark is withheld,
  extrapolation outside the printed ARR band is flagged, and the withdrawn department
  table is asserted to stay withdrawn

---

## 6. Integrity controls

1. **No silent defaults.** `calibration/__init__.py` returns `Calibrated(value=None)` for
   anything absent; `_apply_spend_ceiling` declines to cap rather than invent a ceiling.
2. **`is_observed` gates use.** A value is usable only if printed for that exact band.
3. **Provenance travels with the value.** `Calibrated.citation()` carries publisher, report
   and year to the UI; `page_or_figure` carries the derivation.
4. **Half a benchmark is not a benchmark.** `discretionary_spend_pct_of_mrr()` requires both
   components observed, or returns `None`.
5. **Raw units are preserved.** Retention is stored as published (annual); conversion is
   code with one call site, not a value baked into the data file.
6. **Extrapolation is labelled.** `spend_band_applies_to()` distinguishes a figure printed
   for a company's band from one borrowed from another.
7. **Every engine change is default-off**, so prior research results remain reproducible.
8. **The same rule already applies to the causal graph.** Seeded `MAY_CAUSE` priors render
   as "the board's working assumption", distinct from observed `CONFIRMED_CAUSE` edges.

---

## 7. How to verify every claim here

```bash
git log --oneline founder-calibration
```

```bash
venv\Scripts\python.exe -c "import calibration as c; d=c.monthly_churn(40,'gross'); print(round(d.value*100,2), d.confidence, d.page_or_figure)"
```

Expect `3.65 observed p30, 'Retention benchmarks by ARPA per month range' (annual 64% -> monthly)`.

```bash
venv\Scripts\python.exe -c "import calibration as c; print(c.annual_retention_to_monthly_churn(54)*100)"
```

Expect `5.01…` — the compounding conversion, not `3.83`.

```bash
venv\Scripts\python.exe -c "import calibration as c; print(c.spend_band_applies_to(144000))"
```

Expect `False` — a $12k-MRR founder is outside the band the spend figures were printed for.

```bash
venv\Scripts\python.exe advice_audit.py
venv\Scripts\python.exe -m pytest tests/ -q
```

Source PDFs are gitignored; re-verification requires the files from the project owner.

---

## 8. Open provenance gaps

| Gap | Blocked on | Consequence while open |
|---|---|---|
| Price → churn coupling | ChartMogul *AI Churn Wave* report, not supplied | price changes still touch churn zero |
| Retention data is 2023 | a current edition | figures likely optimistic; NRR has since declined |
| Spend figures are $3–5M ARR | no published per-ARPA or small-ARR spend split | ceiling extrapolates, and says so |
| `SATURATION_ACQUISITION_RATE` | MicroConf report, not supplied | the one free parameter stays assumed |
| CAC payback, gross margin | not printed in any supplied source | CAC still comes from founder input only |
| Real founder-scale rows | Flippa ToS decision | memory corpus stays simulator-derived |
| Price elasticity | first-party A/B tests | lever unidentified; must not be presented as estimated |

Price elasticity will not close by acquiring anything — it requires running experiments.

Audit variance has since been **closed by measurement**: see §5.4. Nine runs, zero
variation.

**None of these gaps blocks the current build.** They gate two future pieces of work:
a trustworthy price lever, and regeneration of the memory corpus at founder scale.

---

# Revision 3 — 2026-08-26

Dashboard completion (defect D5 and the D1/D3 residue), one derived calibration value,
two gated engine flags, and a new external dataset. Everything below was verified by
execution, not by reading.

## R3.1 Gross margin — derived, not observed

**Problem.** `startup_env.step()` does `state.cash += state.mrr`: revenue booked to cash at
100% margin, no cost of revenue. Confirmed by execution — `cash_t1` equals
`cash_t0 + mrr − salary` to the cent. Harmless in research runs where only relative
comparisons matter; visibly wrong the moment a founder is shown a 12-month cash projection.

**Fix.** `calibration.gross_margin_pct()` returns **83.5%**, with `confidence="derived"`.

No source page prints a gross margin. What SaaS Capital 2026 p4 prints, for the $3–5M ARR
band, is the four cost-of-revenue components: hosting 5% + DevOps 3% + pro-services CoGS 5%
+ other CoGS 3.5% = 16.5% of ARR. Summing printed components is arithmetic; calling the
result *observed* would not be, so it is labelled `derived` and carries its own composition
in `page_or_figure`. All four components must be observed or the whole figure is withheld —
a partial CoGS sum overstates margin, which is the direction that flatters a cash
projection, so it fails closed.

`bands.json` still lists `gross_margin_pct` under `still_missing`, and that stays accurate:
the figure is not printed. It is now derivable, which is a different claim.

**Wiring.** `StartupEnv(initial_config={"gross_margin": ...})`. `None` reproduces the original
behaviour exactly. Only the what-if projection sets it.

## R3.2 Two gated engine flags, both proven behaviour-neutral

| Flag | Default | Why |
|---|---|---|
| `gross_margin` | `None` | above |
| `scheduled_shocks` | `True` | The hard shocks at months 24/48/72 are a fixture of the 120-month research episode. A founder at month 20 projecting a year forward would inherit one at month 4 of their forecast for no reason they could see. The what-if turns them off and says so. |

**Identical-output proof.** Five episodes x 60 months of the `heuristic` policy through
`simulation_runner.run_simulation`, SHA-256 of the full result payload, with and without
every change in this revision:

```
with changes:    47b9ba5814e758414687b95d85a7562d3084c647ed5daf9cbb1adb7ed4cf170c
without changes: 47b9ba5814e758414687b95d85a7562d3084c647ed5daf9cbb1adb7ed4cf170c
```

## R3.3 The seed fix, and the larger problem it does not solve

`StartupEnv.reset(seed=N)` called `super().reset(seed=seed)`, which seeds `self.np_random` —
an object the physics never touches. `env/business_logic.py` draws from the **global
`random` module** throughout. Measured: two `reset(seed=42)` runs with identical actions
produced different trajectories.

`reset()` now seeds the module when given a seed. `simulation_runner.py` already did exactly
this immediately afterwards, so this is provably a no-op for existing runs (see the hash
above) and makes the documented contract true for new code.

**This buys reproducibility, not isolation, and the distinction matters.** The stream is
still global, and its per-step draw count is *state-dependent*: `apply_recession_cascade`
draws only when `unemployment > 8 AND interest_rate > 7`. Measured — 7 draws/step normally,
8 once that condition holds. Two policies reaching different macro states therefore
desynchronise and stop sharing a world.

Demonstrated: identical seed, identical actions, varying only how much RNG a *policy*
consumed, competitor counts diverged 5 to 6 versus 5 to 9 over ten months.

**Reach.** Over 120-month episodes the condition is met in **20 of 20 seeds**, first at a
median of ~month 33 — inside the months 25–60 window `post_shock_avg_rule40` is computed
over. Every cross-policy comparison in `results/` is therefore confounded: the policies were
not run against the same worlds. **Not fixed here.** The fix changes `business_logic.py` and
invalidates recorded results, so it is a decision, not a patch.

**Why the what-if is nonetheless clean.** Over a 12-month horizon from founder-typical
conditions the condition is reached in **0 of 50 rollouts** — it needs roughly three
interest-rate and four unemployment shocks inside one year. `run_whatif` re-checks per
request rather than trusting that measurement and returns `shock_tape_shared`; the UI says
so when it is ever False.

## R3.4 What-if projection — defect D5

`backend/whatif_service.py`, `POST /api/whatif`, `frontend/src/whatif.jsx`.

Three policies over the same horizon, seeds and shock tape: the board's plan held, the
founder's current spend held, and the heuristic C-suite recomputed monthly. Four panels
(MRR, cash, churn, Rule of 40) as median + 25–75 IQR band, a summary table, an optional
competitor shock at month 6, and the caveat rendered directly beneath the charts.

Pure simulation — the plan comes from an analysis the client already holds, so there is no
LLM call. **50 seeds x 3 policies in 0.11 s** (0.29 s with the shock counterfactual),
against the 30 s budget.

Two defects found and fixed while building it:

- **The rule-based arm ran unscaled.** `baseline_agents` dollar amounts are tuned for a
  ~$50k-MRR company; against a $12k-MRR founder they proposed more marketing than the
  company earned, and the arm "won" on revenue purely by burning cash it did not have. It
  now takes `absolute_scale(mrr)`, the same G11 correction `Boardroom` already applies.
  Scaled, it loses — and survives only 60% of runs.
- **"Months to recover" reported 0 for a drawdown that never happened.**
  `inject_hard_shock("competitor_surge")` cuts price and raises churn but removes no
  revenue, so a growing company never dips below its pre-shock MRR. Recovery is now
  reported only once a dip is actually observed, and the shock's cost is measured instead
  by re-running each policy without it on the same seeds.

## R3.5 Dashboard honesty fixes

- **D3 — `monthly_burn_override` removed.** It was sent by the client, accepted by
  `FounderConfig`, and read by nothing. Burn reaches the engine through virtual headcount,
  which is correct; the dead field only made the contract look load-bearing.
- **D1 residue — `trace.assumed_fields`.** `build_env_state` set 14 of 18 `EnvState` fields;
  `valuation_multiple`, `unemployment`, `innovation_factor` and `months_in_depression` fell
  to pydantic schema defaults — defaults invisible even in the code constructing the state,
  so nothing could report them. They are now set explicitly, and the server enumerates
  everything it assumed. The UI's "estimated inputs" count was hardcoded on the client and
  could not see any of these four; it now reads the server's list. Verified on a live
  analysis: six fields reported where the client previously showed one.
- **A false caption.** "Expected, in simulation" described a single LLM enum
  (`GROWTH|STAGNATION|DECLINE`) as "a scenario range from a calibrated simulation". No range
  and no simulation were involved. Rewritten, and it now points at the projection, which is
  the thing that claim was actually describing.

## R3.6 Two findings recorded, not fixed

- **`scale_aware_marketing` diverges CAC.** In the scale-aware path a dying company drives
  `estimated_new_users` toward 0, so `raw_cac = spend / ~0` explodes; `gamma` is
  `(acquirable/2) * CAC`, and `gamma ** alpha` then raises `OverflowError` for brand
  (`alpha` up to 3.0). Latent — the flag is default-off and nothing in the product used it.
  The what-if therefore runs on the default curve. There is no upper bound on CAC anywhere
  in `business_logic`.
- **The competitor shock is nearly inert on revenue.** `competitor_surge` sets
  `competitors += 3` (5 to 8), but `compute_new_mrr` steps only at `>= 4` and `>= 10`, so
  5 to 8 changes nothing; and in the default curve new MRR does not depend on price, so the
  25% price cut does not reduce it either. Measured cost over 12 months: **-1.6% to -3.2%**
  of terminal MRR. This is a property of the model, surfaced rather than hidden — the panel
  reports "no drop" instead of inventing a recovery figure.

## R3.7 SEC EDGAR panel — new dataset

`data/ingest_edgar.py` produces `data/edgar.db`, `data/coverage_report.md`, plus CSV exports.

**39 companies, 1,332 company-quarters with revenue, 2010Q2–2026Q2, 10,065 facts.**
Structured XBRL `companyfacts` API only, no scraping, 8 req/s under SEC's published 10/s
ceiling, `SEC_USER_AGENT` supplied by the operator. Every raw response is cached before
parsing, so the database rebuilds offline in seconds and re-runs never re-hit the API.

**Inclusion criterion, declared before screening:** S&M tagged separately from G&A (filers
reporting only `SellingGeneralAndAdministrativeExpense` are excluded — marketing spend is
genuinely unrecoverable from those filings, and any split we invented would be our
assumption wearing an audited company's name), a standard revenue tag, and at least 16
consecutive quarters with revenue, R&D and S&M all present. 2 of 41 candidates excluded.

**Every row carries provenance**: the exact us-gaap tag, accession number, form, filing date
and retrieval date. `tag` and `accession` are `NOT NULL` — a value cannot exist in the table
without saying where it came from.

**The Q4 problem, and why the fix is not imputation.** Filers do not report Q4 as a
three-month figure: 10-Qs carry Q1 plus cumulative six- and nine-month spans, and the 10-K
carries only the twelve-month year. Reading three-month facts alone yields Q1–Q2–Q3 and a
permanent gap at Q4, capping every consecutive run at three quarters — which is exactly what
the first build produced, passing only 4 of 41 candidates. Those quarters are recovered by
differencing two reported year-to-date figures from the same fiscal year. That is exact
arithmetic on filed numbers, not estimation, and each such row records its `derivation` and
the accessions of **both** operands, so `WHERE derivation IS NULL` restricts any analysis to
strictly as-filed values. 76% of rows are as-filed, 24% derived.

**Verified against a known figure.** Twilio's four 2023 quarters, Q4 derived, sum to
$4,153,945,000 against a reported FY2023 revenue of $4.154B, and the derived Q4 of $1.076B
matches their reported Q4. Spot-checked ratios are independently plausible: CrowdStrike
Rule-of-40 50.6, Datadog gross margin 80.9%, Asana 89.7%, Bandwidth 37.3% (CPaaS, correctly
low).

**Known biases, unchanged from the acquisition plan and still declared:** survivorship (only
companies that reached IPO — the bankruptcy dynamic the simulator models is structurally
absent), scale (~$100M+ ARR against the simulator's $50k MRR), and lifecycle stage. All
comparisons must happen in the scale-free `ratios` view, never in absolute dollars.

## R3.8 Environment drift

`scipy` is **not installed** and is **not in `requirements.txt`**, so
`experiments/thesis_analysis.significance_test` silently degrades to `_mann_whitney_fallback`
— a normal approximation with no tie correction. Recorded results in
`results/.../primary_significance_tests.csv` carry `method=scipy_mannwhitneyu`, meaning
scipy was present when they were produced and is not now. Separately, those rows show
`n_a=1, n_b=1, p=1.0`: one episode per arm, so the test was vacuous regardless of method.
Neither is fixed here; both gate any statistical claim.

## R3.9 How to verify this revision

```bash
venv\Scripts\python.exe -c "import calibration as c; g=c.gross_margin_pct(); print(g.value, g.confidence, g.page_or_figure)"
```

```bash
venv\Scripts\python.exe -m pytest tests/ -q
```

```bash
venv\Scripts\python.exe data/ingest_edgar.py --screen --build
```

```bash
curl -s -X POST http://127.0.0.1:8000/api/whatif -H "Content-Type: application/json" -d "{\"config\":{\"initial_mrr\":12000,\"initial_cash\":60000,\"average_price\":50}}"
```
