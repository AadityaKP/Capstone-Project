# Calibration plan — code work and data work combined

Merges the dataset research in `calibration_acquisition.md` with the code roadmap.
Branch: `founder-calibration`.

The organising principle from the research holds up: **three of five parameters have real
public data, two do not, and the two that don't are not blocked on data.** Price
elasticity needs first-party experiments. Marketing saturation was never a data problem —
it was a dimensional error, and it is now fixed.

---

## Status

| # | Parameter | Blocked on | State |
|---|---|---|---|
| 1 | Marketing saturation (`gamma`, `beta`) | nothing — dimensional fix | **done**, opt-in |
| 2 | Discretionary spend ceiling | SaaS Capital (public) | **done**, sourced |
| 3 | Churn by ARPA band | ChartMogul (email gate) | schema ready, **needs you** |
| 4 | Price → churn coupling | ChartMogul retention report | schema ready, **needs you** |
| 5 | Price elasticity | first-party A/B tests | **unidentified**, correctly marked |

---

## Done in this pass

### 1. `calibration/bands.json` + loader

Target schema from the research, with two deliberate deviations:

- **Spend benchmarks live outside the ARPA bands.** SaaS Capital segments by funding type
  and ARR, not ARPA. Forcing those numbers into ARPA bands would attribute them to a
  segmentation their source never used.
- **An `unidentified` block** records parameters no dataset can fix, with the reason and
  what would unblock them. Price elasticity is in it.

The loader (`calibration/__init__.py`) enforces the rule that makes this worth having: a
value is either printed in a cited source or it is `None`. There is no silent default.
`Calibrated.is_observed` gates use, and `citation()` carries provenance to the UI.

### 2. Marketing saturation — reparameterised

`gamma = uniform(15_000, 50_000)` was a spend level with no reference to who was being
bought. Now:

```
acquirable = current_customers * SATURATION_ACQUISITION_RATE
beta       = acquirable * price          # max new MRR per month
gamma      = (acquirable / 2) * CAC      # spend that buys half of them
```

Response as a share of MRR is now consistent across company sizes at equivalent relative
spend — brand at ~25% of MRR returns ~21% of MRR at $12k, $50k and $200k alike. Before,
the same spend returned 33%, 25% and 22%, and small-company ppc paid an absurd 52%.

`SATURATION_ACQUISITION_RATE = 0.20` is the one free parameter left. **No dataset fixes
it. It is assumed and labelled assumed.** MicroConf (source 6) is the closest check —
growth by MRR band for sub-$1M ARR companies would tell us whether 20%/month at saturation
is plausible.

Opt-in via `initial_config={"scale_aware_marketing": True}`; off by default so existing
runs reproduce exactly.

**This changes simulation, not today's advice.** The advise path calls `Boardroom.decide()`
once and never steps the environment, so the curve only bites during episode generation —
which is exactly what corpus regeneration needs.

### 3. Discretionary spend ceiling — sourced

Median private B2B SaaS spends **8% of ARR on marketing and 22% on R&D**
(SaaS Capital 2026, n>1000). Percent-of-ARR is percent-of-MRR monthly, so the median plan
is 30% of MRR. The ceiling is 2× that; over it, marketing and product scale down
proportionally so the board's balance is kept and only the magnitude is corrected.

The multiple is a product judgement, not a measurement, and says so in code.

Worth noting what the benchmark reveals: the engine had been recommending **26% of MRR on
marketing and 40% on R&D — 66% against a 30% median**, more than double.

---

## What I need from you

### A. Three PDFs behind email gates

I cannot pass an email gate. Download and put them anywhere in the repo (they are
gitignored):

1. ChartMogul SaaS Benchmarks Report — `chartmogul.com/reports/saas-benchmarks-report/`
2. ChartMogul Retention: The AI Churn Wave — `/reports/saas-retention-the-ai-churn-wave/`
3. ChartMogul Retention: The New Normal — `/reports/saas-retention-the-new-normal/`
4. MicroConf State of Independent SaaS — `stateofindiesaas.com` (free, no gate, but a
   direct download)

Then run the two-pass extraction from `calibration_acquisition.md` §3 and hand me the
JSON, or hand me the PDFs and I will run it here. **Keep the two passes** — the
monthly/annual confusion the research flags would land directly in founder-facing risk
numbers, and a single pass reliably makes it.

Fills: churn by ARPA band (bands 3), GRR/NRR by price point (band 4), and a sanity check
on `SATURATION_ACQUISITION_RATE`.

### B. A decision on Flippa and Acquire.com

I have not touched either, and I would not without you deciding two things:

- **Flippa**: listings are public, but bulk scraping is governed by their terms, not by
  whether the HTML is reachable. Check the ToS and `robots.txt` and tell me if it is
  permitted. If yes, the 12-month revenue series is genuinely valuable — real
  (state → state') transitions at exactly the scale the memory corpus lacks.
- **Acquire.com**: your own research notes the financials sit behind authentication.
  I will not work around an auth boundary, and logged-out data is asking price plus
  description, which is not worth the effort. My recommendation: skip it.

Whatever comes from either source needs the selection bias recorded in corpus metadata —
every row is a company that chose to sell, so growth skews flat-to-declining. Use for
distribution shape, never for outcome labels.

### C. A judgement call on price elasticity

`apply_pricing_effect` currently draws `elasticity = uniform(-0.9, -0.2)` and touches
churn zero. Two things follow, and the second is yours:

1. **Structural (mine):** a price change should carry a churn penalty. The retention data
   in source 2 gives the shape — products above $250/month see 70% GRR versus 45% for
   $50–249. I can wire the coupling with the coefficient read from `bands.json`, inert
   until the band is filled.
2. **Product (yours):** until first-party price tests exist, do we (a) keep the random
   draw and label the price lever unidentified in the UI, or (b) freeze price changes out
   of the recommendation set entirely? (b) is more honest and less useful. I lean (a)
   with a visible caveat, matching how seeded causal edges are now handled — but it is
   your call, since it decides whether founders see a lever we cannot justify.

---

## Sequence

1. **B (decision)** — cheap, unblocks or kills the Flippa work before anyone spends time.
2. **A (PDFs)** → extraction → fills bands 3 and 4 → churn and price→churn become sourced.
3. **Corpus regeneration** with `scale_aware_marketing: True` and founder-scale
   `environment_config`. Do this *after* A, so the regenerated corpus carries calibrated
   churn rather than needing a second overnight run.
4. **C** once first-party price changes accumulate in the `decisions` table.

## Guardrails

- `advice_audit.py` after every change; it is the regression test for advice.
- `pytest tests/ -q` — 36 tests.
- Every engine change stays behind a default-off flag. `scale_aware_marketing` joins
  `include_burn_context`, `hiring_runway_guard_months` and `scale_absolutes`.
- Do not ship the source PDFs or reproduce their tables in the product. Fitting shipped
  constants from them is fine; republishing is not.
