# physics_v2 calibration report — ROUND 2

Date: 2026-09-05 · Branch `round2` (from `physics-v2`) · Protocol frozen
before any result in [PROTOCOL_round2.md](PROTOCOL_round2.md) and
[BRIEF_V2_SPEC.md](../oracle/BRIEF_V2_SPEC.md), tag `r2-preregistered`.
Round 1 is closed; its HOLDOUT was never re-used for fitting or selection.
Every number below is computed by script from CSVs on disk; failures are
results.

## 0. Verdict summary

| Criterion (frozen bar) | Result | Verdict |
|---|---|---|
| R2-C1 hold-arm median \|4q error\|, EVAL2 = HOLDOUT-19 at q0+8, round-1 `s` frozen (≤10 PASS / ≤20 PARTIAL) | **14.8pp** (signed +8.4pp; DEV2 15.7pp; $50 mapping 13.8pp) | **PARTIAL** |
| R2-SIGN growth-sign agreement ≥ 70% | **100%** (19/19) | **PASS** |
| R2-CORR: boardroom growth std ≥ actual/3 **AND** Spearman(hold, actual) > 0.3 | std ratio **1.09** ✓ but Spearman **0.18** (p=0.45) ✗ | **FAIL** |
| R2-FIN-a: ≥ 90% of EVAL2 survive >50% seeds under hold, opportunistic financing | **100%** | **PASS** |
| R2-FIN-b: ≥ 80% of zero-survival (financing off) companies rescued | premise empty — **no** EVAL2 company has 0 surviving seeds at q0+8 with financing off | **N/A** |
| R2-REG: round-1 flags reproduce round-1 HOLDOUT + legacy A2/E1 exactly | exact on all columns | **PASS** |
| R2-VAR precondition (DEV2): LTV:CAC IQR ≥ 0.5 under mapping v2 | IQR **1.07**, range [2.19, 10.0] | **PASS** |
| Brief-v2 B1 (≥3/4 level sweeps move; frozen variant rule) | v1 0/4, v2a 1/4, v2b 1/4 (+ new LTV:CAC sweep ρ=0.65 in v2b) | **FAIL** — brief v1 stays |
| Robustness (v2 physics): A2 boardroom > baselines AND oracle ≥15/20 | A2 ✓ (g 0.78–0.82, Holm p≈2e-14) but oracle_v3 > boardroom only **8/20** (p=0.45) | **FAIL** — oracle value is calibration-sensitive |
| RS-1 / RS-2 (random shock timing) | _harvested below (§6)_ | — |

DEV2 fix iterations: **0** (all gates passed first run). CAC clamp count: 4
of 39 companies at $250. EVAL2 was run exactly once.

## 1. What round 2 changed (two things, both behind flags, defaults = round 1)

1. **`mapping_version="v2"`** — company-specific CAC from each company's own
   trailing S&M efficiency (protocol formula; no look-ahead, asserted in
   tests), clamped to CAL-quarter [p5, p95] = [$926, $4,233] at $250 ARPA.
   LTV unchanged, so LTV:CAC varies across companies (DEV2 median 3.46,
   IQR 1.07).
2. **`financing_model="opportunistic"`** — runway-binned raise hazard from
   CAL burning quarters only (D4 raise definition): h = 0.295 / 0.084 /
   0.141 / 0.100 per month, K = 26.4 / 20.0 / 9.8 / 26.5 × monthly net burn
   for runway bins [0,12) / [12,24) / [24,48) / [48,∞); all bins n ≥ 10, no
   inheritance. Active from month 1; same single unconditional draw per step
   as the rescue rule. Round-1 `SATURATION_ACQUISITION_RATE_V2 = 0.0727`
   frozen — not re-fit.

## 2. R2-C1 as an out-of-time test of the round-1 fit

The fit was made on CAL at q0; EVAL2 initializes the *held-out* companies 8
quarters later. Median |error| degrades 8.1pp → 14.8pp (PARTIAL) with the
sign flipping to over-projection (+8.4pp signed): the model built for the
earlier, smaller states over-projects the later, more mature ones. DEV2
(15.7pp) sits at the same level — the degradation is time-shift, not
split-specific. v1 physics on these states would sit near +50pp; the fitted
curve carries most of its value out of time.

## 3. R2-CORR: the corridor-artifact criterion fails differently each round

Round 1: correlation without variance (hold ρ=0.73, boardroom std ratio
0.16). Round 2: variance without correlation (std ratio 1.09 via
CAC-differentiated corridor tiers; hold ρ=0.18, p=0.45). Diagnosis: at q0+8
every EVAL2 company's real S&M spend sits deep in the fitted curve's
saturation region, so the hold arm projects ≈43% growth for everyone
(std 0.032) while actual growth still varies (std 0.179) — the residual
dispersion is exactly the hypergrowth tail (CRWD −26pp, DDOG −28pp) and the
slow growers (KLTR +41pp) that a single global saturation parameter cannot
separate. Both rounds FAIL the conjunctive criterion; there is no round 3
inside this protocol.

## 4. Financing: R2-FIN-a PASS, R2-FIN-b N/A

With the opportunistic hazard, 100% of EVAL2 companies survive under hold
(ground truth: all did). The pre-registered ablation R2-FIN-b turned out to
have an empty premise: at q0+8 no EVAL2 company dies on all seeds even with
financing off (later-quarter states are better capitalized relative to
burn). Recorded as N/A, not gamed. The round-1 financing FAIL (2/6 rescued
at q0 under the rescue rule) stands as the round-1 result; the opportunistic
rule was not re-run on the round-1 states because the round-1 HOLDOUT is
closed.

## 5. Brief v2 (B1) and robustness under v2 physics — both FAIL, both informative

**B1.** The level block fixes the headline A4 defect for runway (ρ 0.00 →
0.97: LOW→MEDIUM→HIGH→CRITICAL as cash falls) and v2b's floors move the new
LTV:CAC sweep (ρ=0.65, floor share 13%). But churn, confidence and
competitors sweeps stay flat — mechanically, the ActionModifier consumes
risk/growth/efficiency/innovation only (macro_condition never reaches the
spend arithmetic) and the model holds innovation_urgency flat. 1/4 < 3/4 →
brief v2 recorded FAIL; brief v1 stays primary; B2/B3 not run; no prompt
was modified after reading results.

**Robustness.** A2 ordering survives v2 physics (boardroom > noop / random /
heuristic, paired g 0.78–0.82, Holm p≈1.6e-14, 50 seeds; caveat: boardroom's
post-shock Rule-of-40 is *worse* than noop/heuristic under v2 — heavy spend
hurts the margin term). The oracle layer does **not**: oracle_v3 >
boardroom in 8/20 seeds (mean paired diff −$110k, median −$3.4k, Wilcoxon
p=0.45, both arms 100% survival). The recorded 20/20 / +$1.15M advantage is
a property of the legacy physics' over-generous marketing response; once
acquisition saturates at the fitted rate, the brief-driven spend-up stops
paying. **Paper language: every oracle-layer value claim must be labelled
calibration-sensitive.**

## 6. Random-shock ablation (RS-1 / RS-2)

3 shock months per episode drawn uniformly from [12,108] (min spacing 12)
from the episode world RNG — equal seeds share schedules across arms; legacy
physics; 20 matched seeds; freq 10; all arms 100% survival.

- **RS-1 PASS:** oracle_v3 > boardroom in **20/20 seeds** under random
  timing (mean paired diff **+$1.30M**, median +$946k, Wilcoxon p=1.9e-06).
  The oracle advantage does not depend on the learnable fixed {24,48,72}
  timetable.
- **RS-2 FAIL:** oracle_v3 − oracle_v3_no_memory mean **+$11.6k**, 95%
  bootstrap CI **[−$590, +$25,014]** (includes 0), positive in 10/20 seeds.
  The episodic-retrieval increment is **not detectable under random shock
  timing** — the recorded fixed-timetable increment (+$37.9k, ≈3% of the
  oracle gain) must be described with this qualifier.
- Post-shock Rule-of-40 recovery within 24 months (per-episode schedules):
  boardroom 68%, oracle_v3 78%, no-memory 73% — the recovery-rate advantage
  persists directionally under random timing.

## 7. Second-LLM sensitivity (reported, no gate)

_(filled on harvest by s6_second_llm.py; B1 failed, so v2a stands in as the
level-block variant — deviation noted)_

## 8. Case study (S8)

Frozen ranking rule over the recorded A3 live replication; 115 qualifying
points; top: **seed 15, month 60** — the memory arm read risk LOW where the
no-memory arm read MEDIUM, spent 48% more on marketing, and held a +10.4%
MRR advantage 6 months later. Figure
`validation/figures/review/f8_case_study_seed15.png`; decision-level quotes
(retrieved memories, brief, pre/post-modifier actions) from the
fidelity-checked replay in `validation/round2/case_study.md`.

## 9. Reproduction

```
python validation/calibration/make_eval2.py            # R2-1 (data; committed first)
python -m pytest tests/test_round2.py tests/test_brief_v2.py tests/test_shock_schedule.py
python validation/round2/r2_3_dev2_gates.py            # R2-3 gates (DEV2 only)
python validation/round2/r2_4_eval2_backtest.py        # R2-4 EVAL2 (once)
python validation/round2/a2_policy_baselines_v2phys.py # A2 robustness rows
python validation/round2/a3_oracle_value_v2phys.py     # A3 robustness (Ollama)
python validation/round2/gates_v2phys.py
python validation/round2/b1_level_sweeps.py            # B1 (Ollama)
python validation/round2/a3_random_shock.py            # RS ablation (Ollama)
python validation/round2/gates_rs.py
python validation/round2/s6_second_llm.py              # second LLM (Ollama)
python validation/round2/case_study_select.py
python validation/round2/case_study_replay.py          # (Ollama)
python validation/round2/case_study_report.py
python validation/round2/s9_scorecards.py
```
