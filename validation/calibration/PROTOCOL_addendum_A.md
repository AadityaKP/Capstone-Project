# Protocol addendum A — oracle-layer decomposition under recalibrated physics

Branch `round2`. Committed before any run in this addendum is queued. Motivation:
§5d records that oracle_v3 > boardroom in only 8/20 seeds under v2 physics. The
recorded diagnosis ("the brief-driven spend-up stops paying once acquisition
saturates") is a hypothesis about *one* component — the ActionModifier's fixed
spend multipliers — not about the brief or the memory. This addendum tests that
hypothesis with pre-declared arms and interpretation rules. No constant is
re-tuned; no arm is added after any result is read.

## Common configuration

v2 physics (`marketing_curve="v2"`, `financing_enabled=True`,
`corridor="scale_aware"`, `competitive_entry="scale_neutral"`),
`deterministic_rng`, oracle frequency 10, the same 20 seeds as
`a3_oracle_value_v2phys.csv`, llama3.1:8b temp 0 unless stated. Existing arms
are reused as pairs; nothing recorded is re-run or edited. Outputs go to
`validation/results/a3_decomp_*.csv`.

## Arms

| ID | Arm | Paired against | Question |
|---|---|---|---|
| D-a | `oracle_v3_no_modifier` (brief feeds the weight adapter only; ActionModifier off) | boardroom in `a3_oracle_value_v2phys.csv` | Does the brief add value without the spend multipliers? |
| D-b | `oracle_v3` + `modifier_bound="tier"` (multipliers cannot lift spend past the CMO/CPO top tier) | same | Is the loss caused by spend-up past the corridor? |
| D-c | `oracle_v3` + `shock_recovery="mean_revert"` | **new** `boardroom` arm with the same flags (no LLM; mean-revert changes the world) | Does oracle value under fitted acquisition depend on shock recoverability? |
| D-d | `oracle_v3`, brief v1, model **qwen2.5:7b-instruct** | boardroom in `a3_oracle_value_v2phys.csv` | Is the null LLM-specific? (qwen reads churn/confidence levels; llama does not — §7 of round-2 report) |
| L-1 | `oracle_v3`, brief v1, qwen2.5:7b-instruct, **legacy** physics | boardroom in `a3_oracle_value.csv` (same 20 seeds) | Does the headline result depend on the LLM? |
| RS-2x | `oracle_v3` and `oracle_v3_no_memory`, `shock_schedule="random"`, legacy physics, **seeds 21–40** | each other | Single pre-declared extension of RS-2 to pooled n=40. |

## Criteria (frozen)

* D-a, D-b, D-c, D-d, L-1: PASS if the oracle arm beats its paired boardroom on
  final MRR in ≥ 15/20 seeds. Also reported: paired mean and median diff,
  bootstrap 95% CI, Wilcoxon p, survival, post-shock Rule-of-40 recovery rate.
* RS-2x: pooled n=40 paired diff (v3 − v3_no_memory) with bootstrap CI. Both
  the n=20 result already recorded and the pooled result are reported; the
  extension happens once, and this document is the record that it was decided
  before the extension seeds were run.

## Interpretation rules (frozen)

* D-a **or** D-b PASS while the recorded oracle_v3 (8/20) FAILs → paper states:
  "the brief mechanism's value survives recalibration; the legacy-tuned
  ActionModifier spend multipliers are the calibration-sensitive component."
* D-c PASS → "under fitted acquisition, oracle value depends on shock
  recoverability."
* D-d PASS while llama FAILs → "the null under v2 physics is LLM-specific."
* L-1 FAIL → the headline oracle claim carries an "llama3.1:8b" qualifier.
* All of D-a..D-d FAIL → the null under v2 physics stands; §5d/§8 language
  unchanged.
* RS-2x pooled CI excludes 0 → "small but detectable at n=40"; otherwise
  "not detectable at n=40".

## Non-LLM companions (same night, any order)

* A2 policy comparison, 50 seeds, v2 physics + `shock_recovery="mean_revert"`
  → `policy_comparison_v2phys_mr.csv` (completes the robustness panel for the
  most realistic configuration).
* E-battery (E1/E2/E3/E4/E5) under `shock_recovery="mean_revert"` for legacy
  and v2 physics → `environment_scorecard` rows suffixed `_mr`. Reported;
  no new criteria.

## Excluded on purpose

Brief v2 variants (B1 recorded FAIL; using them here would be selection after
the fact). Re-tuned modifier tables. Any further calibration round.
