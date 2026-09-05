# Round-2 plan — 4 days (Sun 6 Sep → Wed 9 Sep 2026)

Inputs: `validation_report.md` (2026-08-30, §5c updated 09-05), `calibration_report.md`
(physics_v2, 2026-09-05), `PROTOCOL.md` (round 1). Companion drafts, to be committed
verbatim as pre-registration before any code: `PROTOCOL_round2.md`, `BRIEF_V2_SPEC.md`.

## 0. Standing rules for every session

* New branch `round2` from `physics-v2`. Never touch `main` results or any file under
  `validation/results/`, `validation/calibration/*.csv|json|png|md` from round 1.
  New outputs only, with suffixes: `_v2phys` (research-scale robustness), `_bv2`
  (brief v2), `_r2` (calibration round 2), `_rs` (random-shock ablation),
  `_models` (second LLM).
* Every behaviour change is a config flag with default = current behaviour. Legacy
  flags must reproduce recorded numbers exactly; round-1 v2 flags must reproduce
  `real_company_backtest_v2.csv` exactly. Run the regression gate after every merge
  into `round2`.
* Pre-registration commits (protocol, spec) land **before** the code that
  implements them. Verdicts are computed by scripts, appended to reports; nothing
  is iterated against EVAL2 or against any gate that has already been read.
* Ollama is the bottleneck: one LLM job at a time, queued, in the background
  (`nohup … &`, logs to `validation/logs/`). Non-LLM jobs run in parallel with it.
* Claude Code session prompts are in §6. Each session ends with a commit and a
  one-paragraph note in `validation/round2/LOG.md` (what ran, what was read, gate
  result).

## 1. Decisions (made now, recorded in LOG.md on day 1)

| # | Decision | Rationale |
|---|---|---|
| D1 | Legacy physics stays the paper's primary for research-scale results. v2-physics re-runs of A2 and A3 are appended as robustness rows. | All 75-seed runs are legacy; re-running everything is not possible in 4 days. Robustness rows convert the calibration into a strength. |
| D2 | Calibration round 2 runs under `PROTOCOL_round2.md`, time-boxed to day 2. EVAL2 = HOLDOUT-19 at q0+8; exactly two changes (company-specific CAC; opportunistic financing); no re-fit of `s`. | Both FAIL diagnoses point at mapping/financing, not physics. Same-company later-quarter states are the only fresh evaluation set available; disclosed. |
| D3 | Brief v2 under `BRIEF_V2_SPEC.md`. Variant chosen by the frozen rule. If both fail B1, brief v1 stays and A4 FAIL is a limitation. | Level-blindness is the one agent defect that hurts at real scale and is fixable by prompt design. |
| D4 | Memory narrative: the value is the brief mechanism (trend/shock reactor); episodic retrieval is a small, significant modulator (≈3%). The random-shock ablation decides whether even that survives. | Matches the recorded evidence; avoids over-claiming. |
| D5 | Outcome metrics in the paper: final MRR, survival, post-shock Rule-of-40 recovery, brief accuracy. "Global Reward" is dropped (reward pre-declared excluded; oracle has *worse* reward while winning every headline metric). | Claim audit. |
| D6 | E6 (permanent drawdowns) stays a limitation unless day 4 is free; then a `shock_recovery="mean_revert"` flag + 20-seed A3 rerun is a stretch item. | Structural; changes the headline world. |

## 2. Schedule

Times are approximate wall-clock; LLM jobs (🟠) are queued on Ollama in this order.

### Day 1 — Sun 6 Sep: pre-register, brief v2, robustness rows

| Slot | Task | Session |
|---|---|---|
| 09:00 | Branch `round2`; commit `PROTOCOL_round2.md`, `BRIEF_V2_SPEC.md`, empty `validation/round2/LOG.md` with the decision table. Tag `r2-preregistered`. | S1 |
| 09:30 | Implement brief v2 (flags 1–5 of the spec) + `tests/test_brief_v2.py`. | S2 |
| 12:00 | 🟠 B1: A4 level sweeps for v1, v2a, v2b (`a4_level_sweeps_bv2.csv`). Read → apply variant rule → LOG. | S2 |
| 13:00 | A2 under v2 physics: 50 seeds × 120 mo × {noop, random, heuristic, boardroom}, `deterministic_rng` (`policy_comparison_v2phys.csv`, paired tests). No LLM. | S3 |
| 14:00 | 🟠 queue: A3 under v2 physics — boardroom + oracle_v3, 20 seeds, freq 10 (`a3_oracle_value_v2phys.csv`) ~1.5 h. | S3 |
| 14:00 | In parallel: R2-1 data + R2-2 code (`make_eval2.py`, flags, `tests/test_round2.py`), R2-REG gate. **Do not run EVAL2.** | S4 |
| 18:00 | 🟠 queue (if B1 passed): B2 `oracle_v3_bv2`, 20 seeds (~45 min); then B3 real-scale brief v2, 8 companies × 10 seeds (~1.5 h). Overnight. | S5 |

### Day 2 — Mon 7 Sep: calibration round 2, random-shock ablation

| Slot | Task | Session |
|---|---|---|
| 09:00 | Read overnight: A3 v2phys, B2, B3. Compute gates by script. LOG. | S5 |
| 10:00 | R2-3 DEV2 gates (R2-VAR precondition, DEV2 hold-arm sanity run, financing fires). Fix on DEV2 evidence only; count iterations. | S4 |
| 12:00 | **R2-4 EVAL2 — once.** 4 arms × 30 seeds × 12 mo × {fin on/off} × {$250, $50} (~40 min, no LLM). Verdict script → `calibration_report_round2.md` draft. | S4 |
| 13:00 | 🟠 Second-LLM A4 sweeps, v1 + chosen v2 (~15 min). | S6 |
| 13:30 | 🟠 queue: random-shock ablation — boardroom, oracle_v3, oracle_v3_no_memory, 20 seeds, legacy physics, `shock_schedule="random"` (~2 h). | S7 |
| 14:00 | In parallel: case-study selector script on `a3_retrieval_decision_delta.csv`; identify candidate (seed, month); rerun that seed for both arms with trace logging if traces are missing. | S8 |
| 17:00 | Scorecard appends: `_v2phys`, `_bv2`, `_r2` rows. | S9 |

### Day 3 — Tue 8 Sep: reporting, merge

| Slot | Task | Session |
|---|---|---|
| 09:00 | Read random-shock results; RS gates by script; `_rs` scorecard rows. | S7 |
| 10:00 | Reports: `calibration_report_round2.md` final; `validation_report.md` §5d (round 2), §4.3 addendum (brief v2, second LLM), §4.6 addendum (random shocks), §8 claim language revised, §9 limitations updated. Case-study figure + write-up. | S9 |
| 14:00 | README reproduction commands; final full regression gate on `round2`; merge `round2` → `physics-v2` → `main` (all flags default legacy); tag `thesis-v2-final`. | S10 |
| 16:00 | Stretch (only if all above are done): `shock_recovery="mean_revert"` flag + tests; 🟠 A3 20-seed rerun overnight. | S11 |

### Day 4 — Wed 9 Sep: buffer and paper

Paper integration (results tables from scorecards, required language, limitations),
HITL dashboard check (shows retrieved memory as the "why", runs with v2 flags and
brief v2), presentation figures. Read stretch result if it ran; append or drop.

## 3. Gates and kill switches (all pre-declared; read once)

| Gate | Pass condition | If it fails |
|---|---|---|
| B1 | ≥3/4 level sweeps move for v2a (else v2b) | Brief v2 = FAIL; skip B2/B3; limitation stands |
| B2 | oracle_v3_bv2 > boardroom in ≥16/20 seeds | Do not run B3; report B2 FAIL; brief v1 stays primary |
| B3 | oracle_bv2 median \|error\| ≤ 17.1pp on the 8 companies | Report; oracle-at-real-scale stays "further from reality"; v2 still reported for A3 if B2 passed |
| Robustness (v2phys) | boardroom > each baseline, paired Holm p<0.05; oracle_v3 > boardroom ≥15/20 | Report as-is; paper says results are calibration-sensitive |
| R2-* | per `PROTOCOL_round2.md` | Recorded verdicts; no round 3 |
| RS-1 | oracle_v3 > boardroom in ≥15/20 under random timing | If fail: oracle value depends on the fixed timetable → major limitation |
| RS-2 | oracle_v3 − oracle_v3_no_memory paired CI excludes 0 | If fail: retrieval increment "not detectable under random shock timing" |

## 4. Paper deliverables checklist (day 3–4)

- [ ] Required plot 1 → replace "Global Reward" with final MRR / survival / post-shock R40 (D5), legacy primary + v2phys robustness panel.
- [ ] Required plot 2 "time to adapt" → A8 R40 recovery curves (`f7_post_shock_r40_recovery.png`) + note that recovery-*rate* differs, not speed; revenue-peak recovery (E6) shown separately, never conflated.
- [ ] Required plot 3 ablation → v3 vs v3_no_memory, fixed schedule (recorded) and random schedule (new).
- [ ] Required plot 4 case study → from S8: retrieved memory text, brief, pre/post-modifier action, paired 12-month KPI paths.
- [ ] E4 payroll-definition caveat wherever spend ratios appear.
- [ ] Remove any figure sourced from `*_screenshot_*` summaries; cite `primary_summary.csv`.
- [ ] §8 language: simulator sentence updated with C1-v2 (8.1pp) and round-2 verdicts; agents sentence unchanged unless B2 passed (then add brief v2 as the reported oracle variant with v1 as recorded).
- [ ] Limitations: single-LLM (now with second-LLM sensitivity), churn tenure-decay bias, under-financing (round 1) and whatever round 2 leaves, oracle memory tiers (fixed by `memory_query="normalized"` if adopted), same-company later-quarter EVAL2, D4 panel-wide in round 1, memory learns across episodes within a run.
- [ ] Do-not-claim list unchanged: v4≠v3, reward improvements, retrieval as main value source, real-world agent value, real-company forecast.

## 5. Runtime budget

| Job | LLM calls | Wall time |
|---|---|---|
| B1 sweeps (3 variants) | ~60 | 10 min |
| A3 v2phys (2 arms × 20) | ~900 | 1.5 h |
| B2 (1 arm × 20) | ~600 | 45 min |
| B3 (8 × 10, v2 physics) | ~560 | 1.5 h |
| Second-LLM A4 (2 variants) | ~40 | 15 min (+ model pull) |
| Random-shock ablation (3 arms × 20) | ~1,200 | 2 h |
| Stretch A3 mean-revert (2 arms × 20) | ~900 | 1.5 h |
| A2 v2phys, EVAL2, DEV2, gates, tests | 0 | ~2 h total CPU |

Total LLM-bound ≈ 8 h, all in background queue. Fits in two overnights plus day-2
afternoon.

## 6. Claude Code session prompts

Paste each block as the opening message of a Claude Code session in the repo root.
Adjust file paths if they moved on `physics-v2`; the prompts name the functions
as they exist in the last snapshot I saw (`business_logic.py`, `boardroom.py`,
`oracle/prompt_builder.py`, `oracle/memory.py`, `action_modifier.py`,
`simulation_runner.py`).

### S1 — branch and pre-registration

```
Create branch `round2` from `physics-v2`. Add validation/calibration/PROTOCOL_round2.md
and validation/oracle/BRIEF_V2_SPEC.md with exactly the contents I paste below (do not
edit them). Create validation/round2/LOG.md containing the decision table I paste.
Commit with message "round2: pre-register protocol and brief-v2 spec (no code)" and
tag `r2-preregistered`. Confirm the tree has no other changes. Do not implement
anything yet.
```

### S2 — brief v2 implementation + B1 gate

```
Read validation/oracle/BRIEF_V2_SPEC.md first. Implement, each behind a config flag
defaulting to current behaviour:
1. brief_version="v2": in oracle/prompt_builder.build_prompt, insert the LEVEL
   ASSESSMENT block exactly as specified. Runway must use business_logic.monthly_burn
   (the same burn the scale_aware corridor uses), not headcount*8000. Band thresholds
   are the constants already in the physics; do not add tunables.
2. brief_guardrails=True: after JSON parse, apply the severity floors and log
   `brief_floor_applied` in the decision trace.
3. memory_query="normalized": scale-free memory documents/queries per the spec; new
   Chroma collection suffix "_norm"; get_mrr_tier on mrr_rel_start. Existing
   collections must be untouched.
4. modifier_bound="tier": clamp post-modifier marketing to max(pre-modifier proposal,
   0.40*MRR) and R&D to max(pre-modifier proposal, 0.30*MRR); no-op under the legacy
   corridor.
5. runway_estimator="burn" for Boardroom._estimate_runway_months.
Write tests/test_brief_v2.py: flags default off → prompt byte-identical to today;
runway uses real burn; floors only ever raise severity; modifier_bound is a no-op
under legacy corridor and binds under scale_aware; normalized memory doc contains no
absolute dollar figure. Run the full test suite and the Phase 3 regression gate
(validation/calibration/p3_regression_gate.py) — both must be exact.
Then run the A4 level-sweep script three times (brief v1, v2a = flags 1,3,4,5; v2b =
v2a + guardrails), llama3.1:8b temp 0, adding a fifth sweep on LTV:CAC ↓. Write
validation/results/a4_level_sweeps_bv2.csv. Apply the variant choice rule from the
spec and write the result to validation/round2/LOG.md. Do not tune the prompt after
reading sweep results. Commit.
```

### S3 — research-scale robustness rows under v2 physics

```
Goal: append robustness rows, do not modify any existing result.
1. Run the A2 policy comparison with v2 physics flags (marketing_curve="v2",
   financing_enabled=True, corridor="scale_aware", competitive_entry="scale_neutral"),
   deterministic_rng, 50 matched seeds × 120 months, arms noop/random/heuristic/
   boardroom. Output validation/results/policy_comparison_v2phys.csv and
   statistical_tests_policy_baselines_v2phys.csv using the same paired
   Wilcoxon/Hedges-g/Holm code path as the recorded A2.
2. Queue (nohup, background, log to validation/logs/a3_v2phys.log) the A3 run under
   the same v2 flags: boardroom and oracle_v3, 20 seeds matching a3_oracle_value.csv,
   oracle frequency 10, llama3.1:8b. Output validation/results/a3_oracle_value_v2phys.csv.
   Write a small script validation/round2/gates_v2phys.py that computes: boardroom >
   each baseline (paired, Holm), oracle_v3 > boardroom seed count, and prints
   PASS/FAIL against: Holm p<0.05 for all three baselines; oracle wins ≥15/20.
Do not read the A3 output until the job has finished. Commit code; results are
committed when the run completes.
```

### S4 — calibration round 2: data, code, DEV2 gates (no EVAL2)

```
Read validation/calibration/PROTOCOL_round2.md first and follow it literally.
R2-1: write validation/calibration/make_eval2.py. From the EDGAR panel and
panel_split.csv: choose the initialization offset by the frozen rule (+8 if ≥12
HOLDOUT companies have ≥13 complete quarters from q0, else +4); print the count and
the offset. Build eval2_states.csv (columns as in mapped_states.csv plus split ∈
{DEV2, EVAL2}, init_quarter, cac_v2, cac_clamped) using only quarters ≤ init quarter.
Compute cac_v2 per the protocol formula; clamp to [p5,p95] over CAL companies'
quarters; write cac_mapping_r2.csv. Compute the financing hazard table on CAL
companies' burning quarters only with the D4 raise definition; write
financing_hazard_r2.json (bins, n, q_b, h_b, K_b, inheritance flags) and print it
next to the round-1 D4 numbers for comparison. Commit these before any simulation.
R2-2: add flags mapping_version ∈ {"v1","v2"} and financing_model ∈
{"rescue","opportunistic"}, defaults "v1"/"rescue". The opportunistic draw must
use the environment RNG stream so matched seeds stay matched. Write
tests/test_round2.py: hazard math (hand-computed), CAC on a worked example, a
no-look-ahead assertion (the mapping function only ever receives rows ≤ init
quarter), defaults reproduce round-1 behaviour.
R2-3: run p3_regression_gate.py (legacy exact) AND re-run the round-1 HOLDOUT
backtest with round-1 flags on this branch and diff against
real_company_backtest_v2.csv — must be identical. Then, on DEV2 rows only: compute
LTV:CAC spread at init (R2-VAR precondition; need IQR ≥ 0.5) and run the hold arm,
10 seeds, financing on, to confirm raises fire and nothing is NaN. You may iterate
on DEV2 to fix bugs; record every iteration in validation/round2/LOG.md.
STOP here. Do not run EVAL2 in this session.
```

### S5 — read overnight gates, queue B2/B3

```
Read validation/round2/LOG.md for the chosen brief variant. If B1 passed: queue in
order, one at a time, background with logs: (a) B2 — oracle_v3 with the chosen v2
flags, legacy physics, deterministic_rng, freq 10, the same 20 seeds as
a3_oracle_value.csv → validation/results/a3_oracle_value_bv2.csv; (b) B3 — the
p4_oracle_holdout.py configuration (same 8 companies, same 10 seeds, v2 physics)
with the chosen v2 flags → oracle_v3_real_scale_bv2.csv + _summary.csv.
Write validation/round2/gates_brief.py that pairs B2 against the existing boardroom
and oracle_v3 arms in a3_oracle_value.csv (win count, paired diff bv2−v1 with
bootstrap CI) and pairs B3 against the existing no-oracle boardroom rerun (median
|growth−actual| vs 17.1pp; paired oracle−boardroom diff). Only run B3 if B2 passed
(≥16/20). Next morning: run the gates script once, paste its output into LOG.md,
commit. Do not modify prompts or flags after reading.
```

### S6 — second-LLM A4 sensitivity

```
Check `ollama list`. Use the first of qwen2.5:7b-instruct, mistral:7b-instruct,
gemma2:9b that is present; otherwise pull qwen2.5:7b-instruct. Re-run the A4 level
sweeps (the five from S2) for brief v1 and for the chosen v2 variant with that
model, temp 0. Write validation/results/a4_level_sweeps_models.csv with a
`model` column. No gate; report the moved-count per (model, brief_version) in
LOG.md. Commit.
```

### S7 — random-shock-timing ablation

```
Add flag shock_schedule ∈ {"fixed","random"}, default "fixed". Under "random",
draw 3 shock months per episode uniformly from [12,108] with minimum spacing 12,
from the episode world RNG (so equal seeds give equal schedules across arms and
non-drawing policies still share the world); the shock type cycle is unchanged;
log the schedule per episode. Unit test: fixed reproduces {24,48,72}; random is
deterministic per seed and identical across policies. Queue (background, logged):
boardroom, oracle_v3, oracle_v3_no_memory, 20 matched seeds, legacy physics,
deterministic_rng, freq 10 → validation/results/a3_oracle_value_rs.csv with the
schedule columns. Write validation/round2/gates_rs.py: RS-1 oracle_v3 > boardroom
seed count (pass ≥15/20); RS-2 oracle_v3 − oracle_v3_no_memory paired diff on final
MRR with bootstrap CI (pass if CI excludes 0); also post-shock R40 recovery-rate
using the per-episode schedule instead of {24,48,72}. Read once when finished,
paste into LOG.md, commit.
```

### S8 — case study selection

```
From validation/results/a3_retrieval_decision_delta.csv (oracle_v3 vs
oracle_v3_no_memory, 20 matched seeds), write validation/round2/case_study_select.py
that ranks (seed, month) candidates where: the brief label differed, marketing or
R&D spend differed by >20%, and the oracle_v3 arm's MRR is higher than the
no-memory arm's 6 months later. Print the top 5 with the divergence. For the top
candidate, rerun that single seed for both arms with full decision-trace logging
(retrieved memory documents with weights, brief JSON, pre- and post-modifier
action, sanity-bounded action) if traces are not already stored. Produce
validation/figures/review/f8_case_study_seed<N>.png (paired 12-month MRR/cash/R40
paths with the decision month marked) and validation/round2/case_study.md quoting
the retrieved memory text and the brief verbatim. Do not cherry-pick beyond the
ranking rule; state the rule in the write-up.
```

### S9 — scorecards and reports

```
Append rows (never edit existing ones) to environment_scorecard.csv and
agent_scorecard.csv for: A2/A3 v2phys, A4-v2 (B1), A3-bv2 (B2), oracle real-scale
bv2 (B3), A4 second-LLM, RS-1/RS-2, and every R2-* criterion. Each row carries
physics_version, brief_version, mapping_version, financing_model, shock_schedule.
Write validation/calibration/calibration_report_round2.md in the same structure as
calibration_report.md (verdict table first, every criterion including failures,
DEV2 iteration count, clamp counts, hazard table vs round-1 D4). Add §5d to
validation_report.md summarising round 2, brief v2, second LLM, random shocks and
the case study, with the same status legend. Update §8 claim language and §9
limitations per the checklist in validation/round2/ROUND2_PLAN.md §4. Re-derive
every number from CSVs; no hand-typed numbers.
```

### S10 — merge and tag

```
On `round2`: run the full test suite, p3_regression_gate.py, and the round-1
HOLDOUT reproduction diff. All exact. Update validation/README.md with the round-2
and brief-v2 reproduction commands. Merge round2 → physics-v2 → main with all new
flags defaulting to legacy/round-1 behaviour; re-run the regression gate on main;
tag `thesis-v2-final`. Confirm with `git diff main~1 main --stat` that no file
under validation/results or validation/calibration from before round 2 changed.
```

### S11 — stretch: recoverable shocks (only if everything above is done)

```
Add flag shock_recovery ∈ {"none","mean_revert"}, default "none". Under
mean_revert, after inject_hard_shock, record pre-shock price and churn_* and in
apply_recovery move them back toward pre-shock values with a half-life of 3 months
(so ~87% recovered after 9 months, matching EDGAR's median 3-quarter recovery in
E6). Unit test: default leaves trajectories byte-identical. Queue A3 (boardroom,
oracle_v3, 20 seeds, legacy physics otherwise) → a3_oracle_value_mr.csv, and an
E6 drawdown/recovery recomputation on those episodes → e6_drawdown_recovery_mr.csv.
Read once; append rows; if the oracle advantage flips, that is the result.
```

## 7. What "done" looks like on Wed evening

* `main` tagged `thesis-v2-final`; legacy and round-1 numbers reproduce exactly.
* Scorecards carry every new row; two calibration reports; validation report §5c+§5d.
* Verdicts recorded for: brief v2 (B1–B3), robustness under v2 physics, round-2
  calibration (R2-*), random-shock ablation (RS-1/2), second-LLM sensitivity.
* Case-study figure and write-up; the four required plots exist with the metrics
  in D5.
* The paper's claim language and limitations updated from the checklist.
