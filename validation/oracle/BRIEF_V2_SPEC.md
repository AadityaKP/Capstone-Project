# Oracle brief v2 — specification and pre-registered gates

Branch `round2`. Committed before any brief-v2 code or run. Addresses the A4
FAIL (level-blind brief) and its consequence at real scale (oracle |error|
37.7pp vs boardroom 17.1pp on 8 HOLDOUT companies).

## What the current prompt gets wrong (from `oracle/prompt_builder.py`)

* Shows raw levels (`Cash: 1,200,000`, `Confidence: 45.0`) with no reference:
  no runway, no LTV:CAC, no benchmark band, no threshold. The model has nothing
  to judge a level *against*, so it reacts to the only relative information it
  has — the trend deltas and the shock alert line. That is exactly what A4 found.
* `Boardroom._estimate_runway_months` still uses `headcount × $8,000`; at real
  scale this is off by orders of magnitude.
* Memory documents and queries embed absolute MRR (`MRR 1,234,000`), so at real
  scale retrieval similarity is dominated by numbers that never recur.
* `ActionModifier` multiplies before `_apply_sanity_bounds`; LOW risk ×
  ACCELERATING × LOW efficiency = 1.2 × 1.3 × 1.05 = **1.638×**, which at real
  scale lifts marketing past the CMO corridor tiers (40/20/4% of MRR).

## Changes (each behind a flag; defaults = current behaviour)

### 1. `brief_version="v2"` — level block in the prompt

Insert after `Current State`, before `MARKET CONDITIONS`:

```
--- LEVEL ASSESSMENT (computed deterministically; use these bands) ---
Runway: {runway_months:.1f} months  [band: {runway_band}]
  bands: <6 CRITICAL | 6-12 HIGH | 12-24 MEDIUM | >24 LOW
Churn vs benchmark: {avg_churn:.3f} vs band median {band_median:.3f} → {churn_ratio:.2f}× [{churn_band}]
  bands: >1.5× HIGH | 1.15-1.5× ELEVATED | 0.85-1.15× NORMAL | <0.85× LOW
LTV:CAC: {ltv_cac:.2f}  [{unit_econ_band}]
  bands: <1.0 CRITICAL | 1.0-3.0 PRESSURED | ≥3.0 HEALTHY
Macro regime: {macro_regime}
  rule: RECESSION if confidence < 80 or unemployment > 7.0 or months_in_depression ≥ 3;
        EXPANSION if confidence > 110 and unemployment < 5.0; else NEUTRAL
Competitive pressure: {competitors} competitors [{comp_band}]
  bands: ≥10 SEVERE | 4-9 ELEVATED | <4 LOW      (these are the thresholds the market uses)
Cash burn this month: {net_burn:,.0f}  ({burn_pct_mrr:.0f}% of MRR)
```

Runway uses `business_logic.monthly_burn(state)` (the real burn the v2 corridor
already uses), never the headcount proxy. Band thresholds mirror constants that
already exist in the physics (`competitors ≥ 4 / ≥ 10`, `confidence < 80`,
`LTV:CAC ≥ 3` hiring gate, 2.7% churn band median) — no new tunables.

Add one instruction line to the Task block:

```
risk_level must be at least as severe as the Runway band and the LTV:CAC band.
macro_condition must agree with the computed Macro regime.
```

### 2. `brief_guardrails=True` — deterministic floors after parsing (variant v2b)

After the JSON is parsed, apply floors (the LLM may be *more* severe, never less):

```
severity = {LOW:0, MEDIUM:1, HIGH:2, CRITICAL:3}
risk_level  = max(risk_level, runway_band, unit_econ_band)   # by severity
macro_condition = macro_regime if macro_regime == "RECESSION" else macro_condition
```

Each override is logged in the decision trace (`brief_floor_applied: [...]`) so
the share of briefs touched by the floors is reportable.

### 3. `memory_query="normalized"`

Memory documents and queries drop absolute MRR and use scale-free fields:
`mrr_rel_start` (MRR / episode-start MRR, 1 decimal), `mrr_mom_pct`,
`avg_churn`, `churn_ratio`, `innovation`, trend labels, runway band. New Chroma
collection name (`..._norm`) so existing collections are untouched.
`get_mrr_tier` becomes tiers of `mrr_rel_start` (< 0.8, 0.8–1.2, 1.2–2, 2–4, > 4)
instead of absolute dollars.

### 4. `modifier_bound="tier"`

After `ActionModifier.modify`, clamp: marketing spend ≤ max(pre-modifier
proposal, CMO top tier = 0.40 × MRR); R&D ≤ max(pre-modifier proposal, CPO top
tier = 0.30 × MRR). The modifier can move a proposal *within* the corridor but
cannot lift it past the corridor's own top tier. (Only meaningful with
`corridor="scale_aware"`; with legacy corridor it is a no-op and the unit test
asserts that.)

### 5. Runway fix (`runway_estimator="burn"`)

`Boardroom._estimate_runway_months` uses `business_logic.monthly_burn(state)`.
Affects the oracle refresh trigger (`runway < 12`), so it is flagged too.

## Pre-registered gates

Model llama3.1:8b, temperature 0, unless stated. Variants: **v2a** = change 1
only (+3, 4, 5); **v2b** = v2a + change 2.

| Gate | Test | PASS | Runtime |
|---|---|---|---|
| B1 (A4-v2) | One-variable *level* sweeps from the A4 script — runway ↓ (cash), churn ↑, confidence ↓, competitors ↑ — plus a fifth new sweep LTV:CAC ↓. Brief changes in the expected direction. | ≥ 3 of the original 4 sweeps move (5th reported) | minutes |
| B2 (A3-bv2) | Research scale, legacy physics, `deterministic_rng`, freq 10, **same 20 seeds** as `a3_oracle_value.csv`. One new arm `oracle_v3_bv2`, paired against the *existing* boardroom and oracle_v3 arms in that file. | oracle_v3_bv2 > boardroom on final MRR in ≥ 16/20 seeds. Non-inferiority to oracle_v3 (paired diff bv2 − v1) is *reported* with CI, not gated. | ~600 LLM calls, ~45 min |
| B3 (real scale) | The same 8 HOLDOUT companies (APPF BAND KLTR NET PCTY QLYS RNG WK) × the same 10 matched seeds, v2 physics, brief v2, paired against the existing no-oracle boardroom rerun. Exploratory in round 1, exploratory here; disclosed reuse of companies (physics unchanged). | median \|growth − actual\| for oracle_bv2 ≤ 17.1pp (the boardroom's value on these companies). Oracle − boardroom paired diff *reported*, no criterion on its sign. | ~560 LLM calls, ~1.5 h |

**Variant choice rule (frozen):** v2a is the candidate if it passes B1. If v2a
fails B1 and v2b passes, v2b is the candidate and the floor-share is reported
prominently. If neither passes B1, brief v2 is recorded as FAIL, B2/B3 are not
run, and the paper keeps brief v1 with the A4 FAIL as a limitation. B2 must
pass before B3 is run.

## Second-LLM sensitivity (reported, no gate)

Run B1 with one additional Ollama model (whichever of `qwen2.5:7b-instruct`,
`mistral:7b-instruct`, `gemma2:9b` is already pulled; otherwise pull
`qwen2.5:7b-instruct`) for brief v1 *and* the chosen v2 variant. Output
`a4_level_sweeps_models.csv`. Purpose: is level-blindness a property of
llama3.1:8b or of the prompt?

## Outputs

`validation/results/a4_level_sweeps_bv2.csv`, `a3_oracle_value_bv2.csv`,
`oracle_v3_real_scale_bv2.csv` + `_summary.csv`, `a4_level_sweeps_models.csv`.
Nothing existing is edited.
