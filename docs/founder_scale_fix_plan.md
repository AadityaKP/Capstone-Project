# Founder scale fix — analysis and plan

Branch: `founder-scale-fix` (forked from `founder-calibration`).

Everything below was verified against the code on this branch and reproduced by
running the engine, not inferred from screenshots. Numbers in tables are measured
output; appendix A says how to re-run any of them.

**Status: all six steps are done.** Sections 0-4 describe the state the product
shipped in and are kept as the record of what was wrong; §2.1's four-configuration
table is now reachable through the supported payload rather than by patching, so
`founder_scale_probes.py configs` prints the before and after side by side. Two
defects not in the original analysis surfaced while fixing it - a divergent CAC
loop (§2.2) and an efficiency ceiling that refused well-measured data (§5, step 5)
- and both are recorded where they were found rather than quietly patched.

---

## 0. The reproduction

One `run_whatif` call with a real founder's numbers — $2,500 MRR, $5,000 cash,
$500/mo costs, $25 price, 10%/mo churn, 8 months old, 20 seeds:

```
recommended  survival=  0%   12mo MRR=$3,404   cash=$-936
   mrr: [3404, 3404, 3404, 3404, 3404, 3404, 3404, 3404, 3404, 3404, 3404, 3404]
hold         survival=  0%   12mo MRR=$2,546   cash=$-884
   mrr: [2546, 2546, 2546, 2546, 2546, 2546, 2546, 2546, 2546, 2546, 2546, 2546]
rule_based   survival=  0%   12mo MRR=$2,482   cash=$-2,302
   mrr: [2482, 2482, 2482, 2482, 2482, 2482, 2482, 2482, 2482, 2482, 2482, 2482]
churn (all three policies, all twelve months): 4.37
```

Every symptom in the screenshots, exactly: 0% survival, terminal cash a few
hundred dollars below zero, churn identical to two decimals across plans, and a
"12-month projection" that is one number drawn twelve times. The seeds all die in
month 0 or month 1 and `_rollout` forward-fills the corpse to the horizon.

The arithmetic chain in the original diagnosis is confirmed exactly. Month 0 for
the `hold` arm: $2,500 MRR churns to $2,384, plus $50 of expansion = $2,434
recognised to cash; then `salary_burn = headcount * 8000` removes $8,000; cash
goes $5,000 → −$566; `terminated = cash <= 0` fires. Rule of 40 = growth −2.6 +
margin (−8000/2434 × 100 = −328.7) = **−331.3**.

---

## 1. Root cause: the $8k salary slot is a lossy encoding of the founder's costs

The founder's real monthly costs never reach the server. `FounderConfig` has no
cost field. `api.js` converts costs into *virtual headcount* instead:

```js
// derive.js:61
export function virtualHeadcount({ costs, marketingSpend }) {
  const nonMarketing = Math.max((costs || 0) - (marketingSpend || 0), 8000);
  return Math.max(1, Math.round(nonMarketing / 8000));
}
```

Two floors, either of which is fatal on its own: `Math.max(..., 8000)` and
`Math.max(1, ...)`. **Every company with monthly costs between $0 and $12,000
maps to headcount 1 and is charged exactly $8,000/month.** A founder-only company
spending $500 is charged 16× its actual burn. `EnvState.headcount` is `ge=1`, so
there is no encoding for "no payroll" even in principle.

This is not one constant. `headcount * 8000` is a load-bearing protocol with
**seven independent implementations** across five subsystems:

| Site | Role |
|---|---|
| `env/startup_env.py:213` | the cash physics — the only one that can kill the company |
| `boardroom/boardroom.py:579,587` | the board's runway estimate and hiring gate |
| `oracle/oracle.py:361` | the Oracle's runway |
| `oracle/prompt_builder.py:9,74` | **what the LLM is told the burn is** |
| `agents/baseline_agents.py:27` | CFO's runway gate (the `rule_based` comparator) |
| `agents/causal_proposal_agents.py:104` | causal proposal runway |
| `frontend/src/derive.js:63` | the inverse mapping, costs → headcount |
| `advice_audit.py:20`, `tests/test_founder_guards.py:25` | audit and test mirrors |
| `boardroom/boardroom.py:772` | **an eighth, found only by running the product** |

`prompt_builder.py:74` deserves stating plainly: the prompt sent to the model
says `- Monthly burn: 8,000` for a founder whose real burn is $500. **The
strategist is given a false cost base and asked why the company is in trouble.**
That is a direct, mechanical cause of the generic advice, independent of anything
about onboarding.

### 1.2 The eighth site, and why grep missed it

Seven sites were found by searching for `headcount * 8000`. The eighth was found
by running the finished product and asking why the plan was empty:

```python
# boardroom.py, _resolve_conflicts
base_burn = state.headcount * cost_per_employee
```

The constant is a variable here, so no search for the literal could find it - and
it is the wrong variable. `cost_per_employee` is the **one-time recruiting cost**
of a new hire, $10,000 by default, used as such three lines above. Used again as
a monthly salary it charged the $500/month founder $10,000 a month:

```
total_needed = 500 (marketing) + 600 (R&D) + 10,000 (the hire) + 10,000 (base_burn)
shortfall    = 21,100 - 5,000 cash = 16,100     <- entirely fictional
```

The resolver then zeroed the whole plan to cover it, and the founder was shown
**hold / wait / hold** and told nothing. That is the same screen the original
analysis complains about in its §6 - "three of four cards this month are hold /
wait / hold" - diagnosed there as a product decision about empty states. It was
not. It was this line.

Fixing the burn exposed a second flaw underneath it. Cutting a hire rounds **down**
(`math.floor($6,100 / $10,000)` is zero), so a headcount costing more than the
remaining shortfall can never be cut at all: the plan sheds every dollar of
marketing and R&D to protect a hire it still cannot afford. On the real-burn path
the cut now rounds up - a hire you cannot pay for is not a hire - and the
overshoot from removing a whole salary is given back to the marketing budget that
was cut first, rather than banked as a saving nobody asked for.

Measured on the same founder, end to end through `/api/advise` with the real
model: the plan went from `marketing $0 / R&D $0 / hires 0` to **`marketing $425 /
R&D $1,000 / hires 0`**, and the strategist's own risk bullets moved from generic
to specific - "High churn rate (10.0%) compared to industry benchmark (3.7%)".
Research runs keep the original expression and the original rounding, both gated
on `state.monthly_burn is not None`.

### 1.1 There are three financial models, not two

The original diagnosis says two. There are three, and they disagree pairwise:

| Model | Burn it believes | Verdict on the founder above |
|---|---|---|
| `derive.js` runway (Home) | `costs − mrr` = −$2,000 | runway **∞** |
| advisor stack (board, oracle, prompt) | `headcount × 8000` = $8,000 | runway 0.9 months |
| physics (`startup_env.step`) | `headcount × 8000` = $8,000 | **dead in month 0** |

Home says infinity, the board is told 0.9 months, and the simulator kills the
company — about the same company, on the same screen, in the same session.

---

## 2. The correction that matters most: marketing is too *generous*, not too weak

The original diagnosis, and `docs/founder_roadmap.md` §B1 before it, both say
realistic founder marketing spend sits so far left of `gamma` that
`hill_response` returns near-nothing. **For `ppc` — the default channel, and the
one the board actually recommends — the opposite is true**, by a wide margin.

Mean new MRR over 4,000 draws per cell, current absolute constants against the
scale-aware curve that already exists in `marketing_curve_params`:

| Company | Spend | Channel | Absolute → new MRR | as % of MRR | Scale-aware | % of MRR |
|---|---|---|---|---|---|---|
| $2,500 | $125 (5%) | ppc | **$516** | **20.6%** | $106 | 4.2% |
| $2,500 | $625 (25%) | ppc | **$1,431** | **57.2%** | $217 | 8.7% |
| $2,500 | $625 (25%) | brand | $35 | 1.4% | $373 | 14.9% |
| $12,000 | $600 (5%) | ppc | $1,394 | 11.6% | $270 | 2.2% |
| $12,000 | $3,000 (25%) | brand | $618 | 5.2% | $372 | 3.1% |
| $50,000 | $2,500 (5%) | ppc | $3,382 | 6.8% | $1,125 | 2.2% |
| $50,000 | $12,500 (25%) | ppc | $8,210 | 16.4% | $2,741 | 5.5% |

Read the "% of MRR" column down: under the absolute constants, response *rises as
the company gets smaller* — 20.6% for a $2,500 company against 6.8% for a $50,000
company at the same spend share. That is backwards, and it is `beta`, not
`gamma`, that causes it. `beta` is drawn as $10k–100k of new MRR regardless of
company size, so a $2,500 company sitting at 4% of a $30,000 ceiling still
collects $1,255 — half its entire revenue — for $500 of ads. The half-saturation
point being far away does not matter when the ceiling is twelve times the
company's revenue.

Under the scale-aware curve the "% of MRR" column is **size-invariant** at equal
unit economics ($12k and $50k both return 2.2% / 3.4% / 5.5%), which is the
property `test_marketing_response_is_size_invariant_at_equal_unit_economics`
already asserts in `tests/test_founder_guards.py`.

The roadmap's B1 table reached the wrong conclusion because it measured response
as a share of *potential* rather than as a share of *MRR*. 1.1% of a $75,000
ceiling is $825/month; for a $12,000 company that is not "essentially nothing".

### 2.1 The two fixes must ship together

Measured, same founder, same seeds, four configurations:

| | recommended (12mo MRR) | hold | rule_based | survival |
|---|---|---|---|---|
| **A** as shipped today | $3,404 flat | $2,546 flat | $2,482 flat | **0%** |
| **B** real burn only | **$10,802** | $3,192 | $5,691 | 100% |
| **C** real burn + scale-aware | $4,431 | $3,828 | $8,422 | 100% |
| **D** scale-aware only | $2,591 flat | $2,464 flat | $2,908 flat | **0%** |

- **D proves the burn constant is the sole cause of death.** Fixing marketing
  alone changes nothing; everything still dies in month 0.
- **B proves the burn fix alone produces a fantasy.** $250/month of ads takes a
  $2,500 company to $10,802 — 4.3× in a year. That is a *worse* failure than the
  current one, because it looks plausible and a founder might act on it.
- **C is the only defensible configuration.** $2,500 → $4,431 under the board's
  plan, $3,828 doing nothing, $8,422 under a playbook that spends far more.
  Ordering sensible, magnitudes arguable.

Anyone shipping the burn fix without the marketing fix will believe they have
succeeded, because the charts will finally move.

### 2.2 A loop that only exists once the curve is scale-aware

Turning on `scale_aware_marketing` overflowed the float32 observation. It is not
a numerical nuisance; it is a divergent feedback loop the absolute constants
could not create, because `gamma` was drawn independently of anything on the
state:

```
gamma = (acquirable / 2) * state.cac    <- gamma reads CAC
   -> spend far left of gamma, response is a fraction of a customer
   -> raw_cac = spend / 1e-50
   -> state.cac explodes, gamma moves further right      <- and CAC writes gamma
```

Measured, $1/month of brand spend on the founder above:

| month | cac | gamma |
|---|---|---|
| 0 | 50 | 500 |
| 1 | 5.47e4 | 5.33e5 |
| 2 | 1.05e15 | 9.99e15 |
| 3 | 2.98e44 | 2.77e45 |
| 4 | 1.22e128 | `OverflowError` from `hill_response` |

The fix is not a clamp. `compute_cac` had two wrong branches, and both are about
the same mistake - treating an unmeasurable month as a measurement:

- a month that acquired a fraction of a customer returned `spend / 1e-50`, a
  division artifact rather than a cost per customer;
- a month that acquired nobody returned `0.0`, which wrote `cac = 0` and made
  marketing look free the following month. That was a standing tailwind under
  the `hold` arm, which spends nothing.

So CAC is now re-estimated only in a month that actually acquired at least one
customer; otherwise the previous estimate stands, which is also what a founder
would say about a month in which nobody signed up. The guard is on exactly when
the curve that closes the loop is on (`stable_cac` defaults to
`scale_aware_marketing`), so research runs are untouched.

`hill_response` still raises `OverflowError` for `gamma ** alpha` above roughly
1e103. With the guard it is unreachable, so it is left alone and pinned by
`test_without_the_guard_cac_is_the_runaway_this_prevents` rather than papered
over.

---

## 3. R&D is multiplied by exactly zero

Not "cannot move churn enough to matter" — cannot move it at all:

```python
# business_logic.py:240
gain *= (1.0 - state.innovation_factor)
```

and `advise_service.DEFAULT_INNOVATION_FACTOR = 1.0` for every founder. The
product is zero. Measured, starting from the founder default:

| R&D spend | product_quality after | innovation_factor after |
|---|---|---|
| $200 | 0.5 → **0.5** | 1.0 → **1.0** |
| $5,000 | 0.5 → **0.5** | 1.0 → **1.0** |
| $500,000 | 0.5 → **0.5** | 1.0 → **1.0** |

`product_quality` is the only input to `compute_churn_rate` that a plan can move,
and nothing can move it. **The CPO's entire lever is dead for every founder**,
which is why churn is identical to two decimal places across all three arms. Even
starting from `innovation_factor = 0.8`, $50,000 of R&D moves quality by +0.0017
— a 0.08% relative change in churn.

Two further reasons churn looks inert, both real:

- `tenure_decay = exp(-0.15 * months_elapsed * 0.4)` depends on nothing but the
  clock. Over a 12-month horizon from month 8 it drifts 4.64% → 2.26%
  identically under every plan. Confirmed as diagnosed.
- `compute_expansion_mrr` returns `mrr * 0.02 * upsell` — a free 2%/month
  tailwind regardless of spend. It is why the `hold` arm grows 53% over 12 months
  on zero spend in configuration C. The do-nothing baseline is not doing nothing.

---

## 4. Where the original diagnosis is wrong

Three items, called out so nobody spends a day on work that is already done or
points the wrong way.

1. **The shock button already works.** `whatif_service` does not depend on the
   24/48/72 cadence: `_make_env` passes `scheduled_shocks: False`, and `_rollout`
   calls `business_logic.inject_hard_shock(env.state, SHOCK_TYPE)` directly at
   `SHOCK_MONTH = 6` (`whatif_service.py:156`). An explicit `shock` parameter is
   already threaded through `run_whatif`. Fix #4 on the original list is
   complete. What is actually wrong is that the shock is *invisible*, because
   every run is already dead by month 1.
2. **Marketing is too generous, not too weak** — §2. Acting on the original
   framing would push `gamma` down and make configuration B worse.
3. **Onboarding does not pre-fill numeric defaults.** `NumInput` renders
   `value ?? ""` (`Onboarding.jsx:35`); there is no `defaultValue` anywhere. The
   `10` and `15` in the screenshot came from `state.onboardingDraft`, which
   `SAVE_DRAFT` persists to localStorage on every keystroke and which is cleared
   only by `CREATE_COMPANY` or `RESET_ALL`. The real defect is worse than a bad
   default: **a half-finished onboarding from a previous session silently
   repopulates the form, with no way to start over.** That is also the most likely
   explanation for "advice says $1.5k marketing, the screenshot shows $10" — the
   analysed company carries a different draft's numbers. It needs a repro before
   anything is changed, and so does the 10% → 20% churn discrepancy, which no
   code path in `derive.js` or `copy.js` produces.

One item the diagnosis missed that belongs with §2 of its own argument:
`whatYouSell` is collected at onboarding, persisted, and rendered on the Company
page — and **never sent to the server**. `buildAdvisePayload` drops it, and
`build_prompt` contains no qualitative context of any kind. The single field most
determining whether advice can be specific is discarded at the API boundary.

---

## 5. The plan

Six steps. Steps 1 and 2 must land in the same commit; the rest are independently
shippable, in order.

### Step 1 — `monthly_burn` as a first-class field, one source of truth &nbsp;· DONE

The founder's costs become a real quantity instead of a quantised headcount.

```python
# env/schemas.py
monthly_burn: float | None = Field(
    default=None,
    description="Fixed monthly operating cost ($). None falls back to the "
                "engine's headcount-slot convention.",
)
```

```python
# env/business_logic.py
SALARY_SLOT_USD = 8000.0

def monthly_burn(state: EnvState) -> float:
    """The company's fixed monthly operating cost.

    None means "not supplied" and falls back to the original headcount-slot
    convention, so every recorded research run reproduces byte-identically.
    """
    if state.monthly_burn is not None:
        return float(state.monthly_burn)
    return state.headcount * SALARY_SLOT_USD
```

Then replace all seven call sites in §1 with `business_logic.monthly_burn(state)`.
`None` preserves research behaviour exactly — the same default-off discipline as
`include_burn_context` and `scale_absolutes`.

Hiring: `apply_hiring_cost` keeps charging the one-time cost; the ongoing cost
becomes `state.monthly_burn += hires * SALARY_SLOT_USD` when `monthly_burn` is
set. `headcount` survives as a display field and stops implying money.

Transport, so the number actually arrives:

- `FounderConfig` and `WhatIfRequest`: add `monthly_costs: float | None`.
- `api.js buildAdvisePayload`: send `monthly_costs: v.costs` directly. Send the
  founder's real team size as `initial_headcount` (or 1), not `virtualHeadcount`.
- `build_env_state`: map `monthly_costs` → `monthly_burn`.
- Delete `virtualHeadcount` from `derive.js` and its mirrors in `advice_audit.py`
  and `tests/test_founder_guards.py`.

Files: `env/schemas.py`, `env/business_logic.py`, `env/startup_env.py`,
`boardroom/boardroom.py`, `oracle/oracle.py`, `oracle/prompt_builder.py`,
`agents/baseline_agents.py`, `agents/causal_proposal_agents.py`,
`backend/schemas.py`, `backend/advise_service.py`, `backend/whatif_service.py`,
`frontend/src/api.js`, `frontend/src/derive.js`, `advice_audit.py`.
Effort: **M** code, **M** validation.
Done when: a founder with $500/mo costs is charged $500/mo; the LLM prompt states
their real burn; and `pytest tests/ -q` passes with `monthly_burn=None`
reproducing every existing result byte-identically. **All three hold.** Seven
call sites became one helper; the prompt now reads `- Monthly burn: 500`; and its
runway clause was rewritten, because `_runway_months` returned the fragment
`"cash-flow positive"` into `"{} months of cash"` - rare while every company
was charged $8k/month, and the common case now that real costs arrive.

### Step 2 — turn on scale-aware marketing in the product path &nbsp;· DONE

```python
# whatif_service._make_env
"scale_aware_marketing": True,
```

One line. The physics, the parameterisation and the tests already exist
(`marketing_curve_params`, `test_marketing_curve_anchors_scale_with_the_company`);
the flag has simply never been switched on outside a unit test.

Known consequence, to be stated rather than discovered: the scale-aware path
draws **one** uniform per step where the absolute path draws three, so every
what-if trajectory changes. There are no recorded research results behind
`/api/whatif`, so this is safe — but `test_projection_is_reproducible` and any
stored fixtures need regenerating in the same commit.

Effort: **S** code, **M** validation.
Done when: configuration C in §2.1 reproduces, and no arm returns more new MRR in
a month than the company's entire revenue for a spend below 25% of MRR. **Both
hold**, and the largest month-on-month median MRR gain on any arm is 15.4%. It
also surfaced §2.2, which had to ship with it.

Two things visible for the first time once runs survive past month 1: the
`rule_based` arm now grows fastest ($6,886) while only 10% of its seeds survive,
which is the over-spending comparator being correctly punished rather than
winning; and the competitor shock at month 6 finally lands, at -12.8% / -12.1% /
-5.8% against the same seeds. The shock parameter was never broken - it was
unreachable.

### Step 3 — make R&D able to move product quality &nbsp;· DONE

Separate scarring from investment. `innovation_factor` keeps the `(1 − f)`
headroom term because it is a scarring variable. `product_quality` gets its own
headroom, and the saturation constant becomes scale-relative:

```python
scale = R_AND_D_SATURATION_MULTIPLE * max(state.mrr, MIN_SCALE_MRR)  # was 100_000
gain = (spend / (spend + scale)) * 0.05
state.innovation_factor += gain * (1.0 - state.innovation_factor)      # scarring, unchanged
state.product_quality   += gain * (1.0 - state.product_quality) * 0.5  # investable
```

`R_AND_D_SATURATION_MULTIPLE` is a free parameter no public dataset fixes. It is
**assumed**, exactly like `SATURATION_ACQUISITION_RATE`, and must be declared in
`calibration/bands.json` and surfaced in the projection's assumptions list.

Gate it behind an `initial_config` flag so research runs are untouched.

Effort: **S** code, **M** validation.
Done when: the `recommended` and `hold` churn series diverge visibly over 12
months, and the divergence is attributable to R&D spend alone. **Both hold.**

The half-saturation point is anchored to SaaS Capital's published median R&D
spend (24% of ARR), so a company spending what the median company spends buys
half the achievable rate. That deliberately does *not* preserve the old
calibration point: $100,000 of half-saturation at the $50k-MRR company the engine
was tuned for means spending twice your revenue on R&D to get half the available
improvement, which is not a defensible anchor at any size. Replacing it is a
change of belief, not a refactor, which is why it sits behind `scale_aware_rnd`.
`compute_expansion_mrr`'s $50,000 constant *is* 1.0x the calibration company's
revenue, so that one was rescaled rather than replaced and reproduces exactly.

Measured, $2,500 MRR over 12 months, R&D as a share of revenue:

| R&D | $/mo | quality after | effective churn | vs. spending nothing |
|---|---|---|---|---|
| 0% | $0 | 0.500 | 2.590% | - |
| 4% | $100 | 0.521 | 2.554% | -1.4% |
| 24% (median) | $600 | 0.570 | 2.469% | **-4.7%** |
| 100% | $2,500 | 0.608 | 2.403% | -7.2% |

Against 0.500000 at every spend before. It saturates, so there is no free lunch.

### Step 4 — charts end at death, and death is a first-class result &nbsp;· DONE

`_rollout` currently pads a dead company forward to the horizon, which makes
"died in month 2" and "stagnated for a year" render identically. Replace the
padding with real structure:

- `_rollout` returns `died_month: int | None`.
- Each policy's series gains `alive_fraction: list[float]` — the share of seeds
  still solvent in each month.
- `median` is computed over survivors only, and is `null` for months where
  `alive_fraction == 0`.
- Summary gains `median_death_month` and `death_month_p25/p75`.

Survivorship bias is introduced deliberately by the third bullet, so
`alive_fraction` must be rendered, not merely returned: the chart draws the
median solid while `alive_fraction == 1`, dashed as it falls, and stops at zero
with an explicit marker. The table's "Survives 0%" becomes **"ran out of cash in
month 1 in all 50 runs"**.

Files: `backend/whatif_service.py`, `frontend/src/whatif.jsx`.
Effort: **M** code, **S** validation.
Done when: a plan that dies in month 1 and a plan that stagnates for 12 months
are visually distinguishable at a glance. **They are** - solid line, dashed line,
death marker, verified in the running app (3 solid segments, 3 dashed, 1 marker
on the sample company).

The survivorship bias this introduces is worth stating plainly, because it is
visible in the data: the `rule_based` arm's median MRR runs 6,817 -> 6,446 ->
5,738 -> 4,267 -> 8,511 across its last five months. That last jump is not
recovery, it is two survivors out of twenty. The chart is dashed for exactly that
stretch, and the panel says why underneath.

### Step 5 — `founder_view.py`, the translation layer &nbsp;· DONE

One module at the API boundary converting `EnvState` + oracle output into founder
vocabulary. Raw engine fields stay in the response for debugging; **no component
renders a raw env field.** This is also the cleanest capstone artefact in the
plan — a defensible translation layer between simulation state and user-facing
semantics, which is precisely what stops a research ontology leaking into a
product aimed at people who do not have one.

| Engine output | Founder-facing | Rule |
|---|---|---|
| `rule_of_40: −339` | "you spend $3.40 for every $1 of revenue" | R40 suppressed below ~$83k MRR ($1M ARR); it is a public-SaaS benchmark and means nothing below that |
| `churn 4.37%` (sim-internal, decayed) | "about 1 in 20 customers a month" | never share the word "churn" with the founder's *entered* rate; they are different quantities |
| `runway: Infinity` | "at current costs you aren't burning cash" | never print ∞ |
| `survival_rate: 0.0` | "ran out of cash in month N in X of 50 runs" | always rendered adjacent to runway, so the two engines cannot contradict silently |
| `ltv/cac = 100×` → "Healthy" | "we can't measure this yet" | clamp the band; a ratio above ~20 is measurement noise, not health |
| `innovation_factor`, `valuation_multiple` | removed from UI | keep as sim state |

Also fix `ConfidenceStrip` (`components.jsx:299`), which renders
`confidenceBand(brief.confidence)` and `estimatedCount` from independent sources
and so prints "High confidence · 6 estimated inputs". Confidence must be a
function of the assumption count, not merely displayed beside it.

Effort: **M**. Done when: no `.jsx` file imports an engine field name. **None
does**, pinned by `test_no_component_renders_a_raw_engine_field`, which strips
comments first so that explaining a field is not mistaken for rendering it.

Thresholds live in `config/founder_view.json`, not in either implementation. The
client is local-first - Home computes runway and unit economics from numbers the
browser never sends - so a few rules necessarily exist on both sides; the numbers
do not, and a test asserts the JS reaches through the shared file rather than
inlining a copy.

Running it found one bug the unit tests could not: the LTV/CAC ceiling was a flat
20x, and the sample company sits at 20.3x off **44** measured customers. The
ceiling exists because a huge ratio usually means a tiny denominator; when the
denominator is visible and large enough to trust, that reason is gone, and
refusing to answer on good data is its own dishonesty. It now applies only to an
unverified sample, and a large ratio from a good one says "Healthy" with the
caveat that the payback assumes churn holds.

The `Assumed values (9)` panel is also split rather than listed. Interest rate,
consumer confidence, unemployment, valuation multiple and innovation factor are
`EnvState` internals; no founder has an opinion on any of them, and inviting one
to "enter anything here you actually know" invited an invented number into the
analysis. They collapse to one sentence about market conditions. What remains is
`Numbers we guessed (3)` - costs, acquisition cost, churn split - each of which a
founder could genuinely supply.

### Step 6 — the determinism test, and the contradiction test &nbsp;· DONE

Two tests, written before step 1 lands and kept red until it does.

1. **Determinism.** Fill onboarding with known values; assert every number on
   Home and Advice is a pure function of them. Python side asserts
   `build_env_state(payload)` against an explicit expected `EnvState`; JS side
   snapshots `derive.js` outputs.
2. **Non-contradiction — the important one.** Assert the three financial models
   in §1.1 cannot disagree about death:

   ```python
   # if the frontend formula says the company is not burning cash,
   # the simulator must not kill it
   if runway_months(cash, costs, mrr) == float("inf"):
       assert run_whatif(payload)["policies"]["hold"]["summary"]["survival_rate"] > 0
   ```

   This is the executable form of "two engines, one screen", and it fails today
   for every company below roughly $8,000 MRR.

Add a founder-scale fixture alongside them. `tests/test_whatif.py::FOUNDER` is
$12k MRR / $60k cash / headcount 2 — a comfortable company that survives on the
current constants. **No existing test exercises a company below $12k MRR**, which
is why the whole failure shipped green.

Effort: **S**. Done when both tests pass and the founder-scale fixture is in CI.
**Both pass.** The non-contradiction test is parameterised over four company
sizes, and asserts the converse too: a company that really is dying is reported
as dying by both engines. The contract is agreement, not optimism.

Total: 187 tests, `RuntimeWarning` promoted to an error, `advice_audit.py` at 0
violations across 6 profiles, and the research fingerprint unchanged.

---

## 6. Onboarding and information architecture

Real, and worth doing — after steps 1–4. Reordering the form while the engine
still kills every company changes which numbers are wrong, not whether they are.
Two items are cheap enough to fold into step 1:

- **Send `whatYouSell` and a "who buys it" segment** (B2B / SMB / consumer /
  mixed) to the server, and put both in `build_prompt`. This is the mechanism
  behind generic advice; it costs one field in `FounderConfig` and three lines in
  the prompt builder. The segment answer is also what should populate
  `churn_enterprise/smb/b2c`, instead of three separate fields in a drawer no
  founder opens.
- **Add a "start over" control** that dispatches `RESET_ALL`, and clear
  `onboardingDraft` on mount if it is older than a day. Until then every stale
  session is a bug report nobody can reproduce.

The rest — required-qualitative / four-numbers / nothing-else, moving residual
fields into contextual post-analysis prompts, collapsing Home / Advice / History
into one payload serving three jobs, and cutting the `Assumed values` panel down
to the one honest sentence about market conditions — belongs in its own pass,
once the numbers on screen are trustworthy.

---

## Appendix A — reproducing the measurements

All probes run against `venv/Scripts/python.exe` from the repository root. Five
scripts: a single instrumented rollout (§0), R&D and churn (§3), end-to-end
`run_whatif` (§0), the four configurations (§2.1), and marketing response by
channel and company size (§2).

The four-configuration comparison simulates both candidate fixes by patching
rather than editing, so the direction and size of each was measured before any
source change was made: `StartupEnv.step` is wrapped to refund the fake $8k slot
and charge the real $500, re-evaluating `terminated` against the corrected cash;
`_make_env` is replaced with one that sets `scale_aware_marketing: True`.

## Appendix B — guardrails inherited from `founder_roadmap.md`

- Re-run `advice_audit.py` after every change; it is the regression test for advice.
- Re-run `pytest tests/ -q`.
- `neo4j_backup.py dump` before anything that could write to the graph; `verify` after.
- Every engine change stays behind a default-off flag. `monthly_burn=None` and
  `scale_aware_marketing=False` must reproduce recorded research runs
  byte-identically; step 2 deliberately turns the second on for the product path
  only, and step 3's flag must do the same.
