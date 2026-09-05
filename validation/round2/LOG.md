# Round-2 log

Branch `round2` from `physics-v2`. Pre-registration committed before any code:
`validation/calibration/PROTOCOL_round2.md`, `validation/oracle/BRIEF_V2_SPEC.md`,
`validation/round2/ROUND2_PLAN.md` (verbatim). Tag `r2-preregistered`.

## Decisions (day 1, recorded before any round-2 run)

| # | Decision | Rationale |
|---|---|---|
| D1 | Legacy physics stays the paper's primary for research-scale results. v2-physics re-runs of A2 and A3 are appended as robustness rows. | All 75-seed runs are legacy; re-running everything is not possible in 4 days. Robustness rows convert the calibration into a strength. |
| D2 | Calibration round 2 runs under `PROTOCOL_round2.md`, time-boxed to day 2. EVAL2 = HOLDOUT-19 at q0+8; exactly two changes (company-specific CAC; opportunistic financing); no re-fit of `s`. | Both FAIL diagnoses point at mapping/financing, not physics. Same-company later-quarter states are the only fresh evaluation set available; disclosed. |
| D3 | Brief v2 under `BRIEF_V2_SPEC.md`. Variant chosen by the frozen rule. If both fail B1, brief v1 stays and A4 FAIL is a limitation. | Level-blindness is the one agent defect that hurts at real scale and is fixable by prompt design. |
| D4 | Memory narrative: the value is the brief mechanism (trend/shock reactor); episodic retrieval is a small, significant modulator (≈3%). The random-shock ablation decides whether even that survives. | Matches the recorded evidence; avoids over-claiming. |
| D5 | Outcome metrics in the paper: final MRR, survival, post-shock Rule-of-40 recovery, brief accuracy. "Global Reward" is dropped (reward pre-declared excluded; oracle has *worse* reward while winning every headline metric). | Claim audit. |
| D6 | E6 (permanent drawdowns) stays a limitation unless day 4 is free; then a `shock_recovery="mean_revert"` flag + 20-seed A3 rerun is a stretch item. | Structural; changes the headline world. |

## Session log

- **S1** (2026-09-05): pre-registration committed, tag `r2-preregistered`. No code.
- **S2** (2026-09-05): brief v2 implemented behind 5 flags (defaults legacy);
  13 unit tests; full suite 216 green; legacy regression gate exact. One code
  fix before any LLM read (enum coercion in the guardrail floors, crashed the
  first B1 attempt — no sweep result had been read). **B1 gate: FAIL by the
  frozen rule.** v1 0/4 sweeps move (replicates A4); v2a 1/4 (runway now
  ρ=0.97, was 0.00 — the level block fixes the headline defect); v2b 1/4
  gated + the new LTV:CAC sweep ρ=0.65 via floors (floor share 13%). Churn /
  confidence / competitors sweeps still flat: the ActionModifier consumes
  risk/growth/efficiency/innovation only — macro_condition never reaches the
  arithmetic — and the model holds innovation_urgency flat. Per the frozen
  variant rule: **brief v2 = FAIL; B2/B3 not run; brief v1 stays primary;
  A4 FAIL remains a limitation** (with the runway-responsiveness improvement
  reported as a finding, not adopted). No prompt was modified after reading.
- **S4** (2026-09-05): R2-1 data committed before any simulation (offset
  q0+8, 39/39 eligible, CAC clamp [926, 4233] @ $250 with 4 clamped, CAL-only
  hazard table, no bin inheritance). R2-2 flags + 9 tests. R2-3 gates, ZERO
  DEV2 fix iterations: R2-VAR PASS (LTV:CAC IQR 1.07), DEV2 sanity PASS
  (213 raises, finite), R2-REG EXACT (round-1 HOLDOUT reproduction and legacy
  A2/E1 both byte-identical on this branch).
- **S3** (2026-09-05): A2 v2phys done — boardroom > noop/random/heuristic on
  final MRR (paired g 0.78–0.82, Holm p≈1.6e-14, 50 seeds); post-shock R40 is
  WORSE for boardroom than noop/heuristic under v2 physics (heavy spend hurts
  the margin term) — robustness caveat for the paper. A3 v2phys queued on
  Ollama after B1.
- **S3/S5 harvest** (2026-09-05): **A3 v2phys robustness gate FAIL.**
  oracle_v3 > boardroom in only 8/20 seeds under v2 physics (mean paired diff
  −$110k, median −$3.4k, Wilcoxon p=0.45; both arms 100% survival, medians
  $650k vs $697k; 606 LLM calls). The recorded oracle advantage (20/20,
  +$1.15M) does not survive the fitted marketing curve at research scale: the
  ActionModifier's spend-up behaviour stops paying once acquisition
  saturates at the fitted rate. Pre-declared consequence applies: reported
  as-is; the paper must state the oracle-layer result is
  **calibration-sensitive** (in-model value under legacy physics; null under
  v2 physics). A2 half of the gate PASSes (boardroom > all baselines).
  R2-4 EVAL2 verdicts recorded (see r2_criteria_verdicts.json): R2-C1
  PARTIAL 14.8pp, R2-SIGN PASS 100%, R2-CORR FAIL (inverted: variance passes
  at 1.09, hold Spearman 0.18 — saturation flattens hold at q0+8 states),
  R2-FIN-a PASS 100%, R2-FIN-b N/A (premise empty). Random-shock ablation
  queued.
- **S7/S6/S9 harvest** (2026-09-05): RS-1 PASS — oracle_v3 > boardroom
  20/20 seeds under random shock timing (mean +$1.30M, p=1.9e-06): the
  advantage does not rest on the learnable fixed timetable. RS-2 FAIL —
  retrieval increment not detectable under random timing (mean +$11.6k,
  CI [−$0.6k, +$25.0k], 10/20); the +$37.9k recorded increment carries a
  fixed-timetable qualifier. Recovery rates 68/78/73%. Second LLM
  (qwen2.5:7b-instruct): v1 2/4 sweeps move (churn 0.65, confidence 0.62) vs
  llama 0/4 — level-blindness is substantially a model property; v2a runway
  0.97 on both models; no combination reaches 3/4. Scorecards appended
  (8 agent rows, 5 environment rows, full flag columns). §5d + §8/§9 claim
  language updated. **Deferred item:** case_study_replay.py (Ollama, ~40 min)
  to fill the decision-level quotes in case_study.md — figure and recorded
  numbers are in; re-run later with
  `python validation/round2/case_study_replay.py` then
  `python validation/round2/case_study_report.py`.
- **S10** (2026-09-05): full test suite + regression gate on `round2`;
  fast-forward merge round2 → physics-v2 → main (main was a strict ancestor,
  0 behind); tag `thesis-v2-final`; post-merge gate re-run on main.
- **S8 completion + S11 stretch** (2026-09-05, resumed session): case-study
  replay done — no_memory arm byte-identical (rel diff 0.0), memory arm
  0.14% drift over 120 months (fresh Chroma collection tie-breaks,
  disclosed); the month-60 decision reproduces the recorded divergence
  exactly and case_study.md now quotes the three retrieved GROWTH memories,
  both briefs, and the pre/post-modifier actions. S11:
  shock_recovery="mean_revert" (3-month half-life on hard-shock price/churn
  damage; default byte-identical, 3 tests, gate exact). A3-mr: oracle
  advantage does NOT flip — strengthens to 20/20 seeds, +$2.23M mean
  (p=1.9e-06). E6-mr: median drawdown depth 61–63% → 16–17% (EDGAR 11%);
  episodes rare (0.4–0.5/100q); recovery cell unestimable (3–4 censored).
  Scorecard rows appended; merged to physics-v2 and main.
