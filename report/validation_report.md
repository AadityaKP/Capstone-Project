# Validation report — startup simulator and multi-agent decision layer

Date: 2026-08-30 · Branch `review2-sim-frontend` · Full methodology and pre-declared
acceptance criteria: `validation/validation_plan.md` (criteria fixed before any result
was computed; deviations are labelled where they occur).
Consolidated 2026-09-06 on branch `round2`: incorporates the physics_v2
recalibration (§5c), round 2 (§5d), and the pre-registered addendum-A
decomposition (§5d; `validation/calibration/PROTOCOL_addendum_A.md`).

**Status legend.** Every number in this report is VERIFIED/EXECUTED from data on disk.
No synthetic or placeholder values appear anywhere in this report. (The A3 live
replication, pending in an earlier draft, completed on 2026-08-30 and is incorporated.)

---

## 1. Summary verdicts

**SIMULATOR: B — VALIDATED COMPARATIVE TESTBED WITH NAMED STRUCTURAL GAPS.**
Grade raised from C after round 2 because the failures that set the C are now
either repaired or repaired-by-flag with the residuals named and measured.
After the pre-registered split-panel recalibration the simulator retrodicts
held-out real companies (median |4q error| 8.1 pp at the calibration horizon,
C1-v2 PASS; 14.8 pp out-of-time at q0+8, R2-C1 PARTIAL; 100% growth-sign
agreement — the original physics over-projected by ~45–50 pp), its policies
spend realistically under the v2 corridor (E4 PASS, discretionary spend inside
the EDGAR p10–p90 band, vs ≈8% of revenue under legacy), within-unit growth
volatility matches under v2 (E5 PASS), a financing mechanism exists, and the
catastrophic-drawdown depth is repairable by the `shock_recovery="mean_revert"`
flag (median depth 61–63% → 16–17%, EDGAR 11%). The remaining structural gaps:
growth persistence is too high (E2 — structural, per D5), cross-company
dispersion is under-produced (corridor criterion FAIL in both rounds, for
opposite reasons), and under mean-revert drawdowns become too rare
(0.4–0.5/100 quarters vs EDGAR's 3.0 — depth is repaired at the cost of
frequency). It is usable as a *controlled comparative testbed* for policy
experiments under identical dynamics; it is still not supported as a
*predictive model* of individual company trajectories.

**AGENTS: B — EVIDENCE OF POLICY VALUE WITH IMPORTANT LIMITATIONS.**
Actions causally move outcomes, and the boardroom's superiority over
no-action, random and heuristic controls is robust to recalibration (paired
g 0.60–0.82 legacy, 0.78–0.82 under v2 physics, 1.21–1.68 under v2 physics
with recoverable shocks; all at matched seeds). The oracle layer beats the
boardroom in 74–75 of 75 recorded seeds and 20/20 under clean RNG, and the
advantage survives randomized shock timing (20/20) and recoverable shocks
under legacy physics (20/20, +$2.23M) — but it is **calibration-sensitive**,
and the pre-registered addendum-A decomposition locates the sensitivity: the
brief's only causal channel into actions is the ActionModifier's spend
multipliers (with them off, oracle_v3 is outcome-identical to the boardroom,
0/20; tier-bounding them does not restore value, 8/20; nor do recoverable
shocks, 7/20), and the v2-physics null is **LLM-specific** — qwen2.5:7b-instruct
passes the same pre-registered criterion under the fitted curve (15/20; small
mean +$28.8k, CI including 0) and replicates the legacy headline 20/20
(+$564k, p=1.9e-06). Memory-conditioned briefs are genuinely predictive
(+17 pp over base rate); episodic retrieval changes ~60% of monthly spend
decisions with a small outcome increment (≈3% of the oracle gain under the
fixed timetable), not detectable at n=20 under random shock timing but
detectable at the pre-declared pooled n=40 (+$39.7k, CI [+$18.0k, +$65.9k]).
The limitations are material: all demonstrated value is *in-simulator at
research scale*; the primary LLM's brief channel is blind to state levels;
and the real-scale counterfactual does not support any claim of agent value
on real company states.

## 2. What was validated against what

- Internal validity: new matched-randomness experiments (`deterministic_rng`), 50–75
  seeds per arm, paired statistics (Wilcoxon, Hedges-g on per-seed differences, Holm
  correction, bootstrap CIs). Episodes are the experimental unit throughout.
- External validity: the repo's screened SEC EDGAR panel — 39 SaaS companies, 1,288
  complete quarters (2010Q2–2026Q2) with revenue, S&M, R&D, G&A, cost of revenue,
  cash, operating cash flow; inclusion criteria declared before screening
  (`data/coverage_report.md`). All comparisons scale-free; simulator aggregated to
  quarters; nothing interpolated.
- Not validated (declared, not attempted): churn, CAC, LTV, price, product quality,
  customer counts, monthly dynamics, quarterly hiring, macro-block realism —
  unobservable in XBRL (`validation/edgar_data_audit.md`).

## 3. Environment validity (scorecard: `validation/results/environment_scorecard.csv`)

| Test | Result | Verdict |
|---|---|---|
| E1 growth distribution | sim median QoQ growth 5.1–5.4% vs EDGAR 5.7%, inside EDGAR IQR; KS 0.14–0.17, W1 ≈ 0.04 | **PASS** (all 3 arms) |
| E2 growth persistence | sim lag-1 autocorr 0.92 vs EDGAR 0.46 — sim growth is far too smooth/persistent | PARTIAL |
| E3 growth deceleration | within-unit Spearman(growth, log revenue): sim −0.47…−0.53 vs EDGAR −0.64 — right sign, right order | **PASS** |
| E4 spend structure | sim discretionary spend 8.0–8.5% of revenue vs EDGAR S&M+R&D p10–p90 [37%, 93%] | **FAIL** |
| E5 growth volatility | sim within-unit std 0.13–0.15 vs EDGAR 0.046 (×2.8–3.2) | PARTIAL |
| E7a churn constants | default churn (mean 3%/mo) inside ChartMogul $25–100 ARPA band (median 3.4%) | PASS |
| E7b gross margin | research profile books revenue at 100% margin vs EDGAR median 73.5% | **FAIL** (by design, disclosed) |

Interpretation of the failures. E4 partly reflects a definitional mismatch — EDGAR
functional expenses include payroll, the simulator's marketing/R&D spend excludes it
(payroll is a separate headcount burn) — but even allowing for that, simulator policies
under-spend relative to any real SaaS cost structure. E2+E5 together say the simulator
produces trajectories that are simultaneously smoother in trend and noisier
quarter-to-quarter than real companies: real growth has moderate persistence with low
noise; simulated growth has near-unit persistence with high noise (the Hill-response
draws). E7b is a known, deliberate research-profile simplification.

## 4. Agent validity (scorecard: `validation/results/agent_scorecard.csv`)

### 4.1 Do actions matter? (A1 — PASS)
Same state, same seed, one dimension varied, 12-month horizon, 20 seeds
(`validation/agents/action_effects*.csv`): marketing is a strong monotone lever
(terminal-MRR spread 35–124% of the mid-rung outcome; ρ=+1.0); pricing weak-positive
monotone (+4–7%); R&D weak-positive (+8–10%, saturating — its product-quality channel
is inert in the research profile while `innovation_factor`=1.0, so it acts only through
expansion upsell); hiring has no revenue channel and is pure cost — repeated
over-hiring bankrupts the company. Actions demonstrably reach the state; magnitudes
differ hugely by lever.

### 4.2 Better than trivial policies? (A2 — PASS)
50 matched seeds × 120 months, deterministic RNG, paired
(`validation/results/policy_comparison.csv`, `statistical_tests_policy_baselines.csv`):

| Arm | Survival | Median final MRR |
|---|---|---|
| noop | 100% | $2.7k |
| random | 4% | $332k (survivors' path lengths differ) |
| heuristic | 86% | $212k |
| boardroom | 96% | $457k |

Boardroom > noop (paired g=0.82), > random (g=0.60), > heuristic (g=0.78) on final MRR,
all Holm p ≤ 0.003. Honest caveat: noop *never* goes bankrupt and has the least-bad
Rule-of-40 — in this environment spending trades survival risk for growth, and the
reward function (which favors noop-like behaviour on some components) is misaligned
with the headline metrics and was excluded as an outcome, as pre-declared.

### 4.3 Economically sensible decisions? (A4 — split verdict)
Rule layer: 10/10 documented thresholds reproduce exactly (hiring gated on runway>24mo
and LTV:CAC≥3; marketing steps with LTV:CAC; R&D steps with churn, halved when cash is
low). **PASS.**
LLM brief layer: **FAIL on the pre-declared criterion.** One-variable *level* sweeps
(runway ↓, churn ↑, confidence ↓, competitors ↑; llama3.1:8b, temp 0) moved the brief
0/4 times — a state with confidence 45, unemployment 9.5%, rates 8.5% and 5 months of
depression still received risk=LOW / growth=ACCELERATING / macro=EXPANSION.
An exploratory follow-up (designed after that failure, labelled as such) shows the
responsiveness lives elsewhere: briefs respond strongly to *trend deltas*
(MRR MoM +10%→−30% moves growth_outlook ACCELERATING→DECLINING, ρ=+0.96, and the
marketing multiplier 1.64×→0.51×) and to the *shock alert* line (risk LOW→MEDIUM;
marketing ×0.85, R&D ×1.10); churn deltas are ignored. This matches the recorded runs,
where risk labels shift sharply at shock months (MEDIUM share 0.44→0.90). The oracle
layer is best described as a trend-and-shock reactor, not a state assessor.

### 4.4 Does the oracle layer add value over the boardroom? (A3 — PASS, recorded and replicated)
The recorded FULL thesis run (75 seeds × {boardroom, oracle_v1, oracle_v3}) was
re-analyzed as a **paired** design — valid because the legacy shared-world property was
verified empirically this session (non-drawing policies experience identical macro
paths at equal seed, 10/10 seeds; `validation/results/shared_world_check.csv`):

| Comparison | Metric | Mean paired diff | 95% CI | g | p | Positive seeds |
|---|---|---|---|---|---|---|
| oracle_v1 − boardroom | final MRR | +$960k | [+739k, +1.19M] | 0.94 | 5.5e-14 | 74/75 |
| oracle_v3 − boardroom | final MRR | +$862k | [+660k, +1.07M] | 0.93 | 5.3e-14 | 75/75 |
| oracle_v1 − boardroom | post-shock R40 | +10.6 | [+9.7, +11.4] | 2.73 | 5.7e-14 | 74/75 |
| oracle_v3 − boardroom | post-shock R40 | +9.7 | [+8.8, +10.6] | 2.42 | 5.3e-14 | 75/75 |

The original analysis used unpaired Mann-Whitney tests (p≈0.012–0.02 on final MRR);
pairing sharpens this to near-uniform per-seed dominance.

**Live replication (completed).** A fresh 20-seed run under `deterministic_rng`
(boardroom / oracle_v1 / oracle_v3 / oracle_v3_no_memory; freq 10; 1,864 LLM calls;
`validation/results/a3_oracle_value.csv`) replicates the result under the clean RNG
regime: every oracle arm beats boardroom in **20/20 seeds** on both metrics
(final MRR mean diff +$1.15–1.19M, g ≈ 0.94–0.96; post-shock R40 +13.0–13.2,
g ≈ 3.1–3.4; Wilcoxon p = 2e-6, the minimum attainable at n=20). Survival:
boardroom 95%, all oracle arms 100%.

Where the value comes from: the effect is concentrated post-shock — exactly where
§4.3 shows the brief channel actually responds. The oracle advantage is real in-sim
but rests on the shock/trend channel, not on continuous state assessment.

### 4.5 Claim audit (`validation/results/claim_audit.csv`)
22 recorded claims re-computed from raw CSVs: 20 REPRODUCED exactly; 1 UNSUPPORTED —
**any claim that oracle_v4 differs from oracle_v3** (per-seed correlation 1.000000,
identical medians in the CONFIRMATION run); 1 LEAKAGE-RISK — the curated
`primary_summary_screenshot_no_oracle_v3.*` files (a policy row deleted post-hoc);
cite only the canonical `primary_summary.csv`. Also confirmed: oracle policies have
*worse* total reward than boardroom while beating it on every headline metric — the
reward function must not be cited as an outcome.

### 4.6 Retrieval/memory value (A6 — prediction PASS; outcome effect small but positive)
Recorded FULL run, 2,200 fresh oracle_v3 briefs across 75 episodes: expected_outcome
accuracy 64.5% vs 47.6% majority-class base rate; +17.0 pp, episode-clustered
bootstrap CI [+11.2, +22.5] (`validation/results/brief_accuracy.csv`). The
memory-conditioned brief carries real predictive signal.

The completed matched-seed ablation quantifies the *outcome* increment. Retrieval
changes decisions substantially — marketing spend differs in 62.6% of months, R&D in
60.2%, the brief labels themselves in 20.7%
(`validation/results/a3_retrieval_decision_delta.csv`) — and the changed decisions are
slightly better: v3 − v3_no_memory on final MRR is +$37.9k mean (median +$5.8k),
95% CI [+14.0k, +66.3k], g = 0.59, p = 0.0023, positive in 15/20 seeds; post-shock R40
+0.27, CI [+0.07, +0.53], p = 0.044. **Attribution finding:** episodic retrieval
contributes only ≈3% of the oracle layer's total gain over the boardroom
(+$37.9k of +$1.19M) — the bulk of the value comes from the brief mechanism itself
(trend/shock reactivity through the ActionModifier), not from memory. At n=20 the
retrieval increment is statistically supported but must be described as small; it is
not the headline of the architecture's value.

## 5. Real-company counterfactual (C1 — FAIL, and a diagnosis)

39 companies initialized from their earliest complete EDGAR quarter (mapping in
`validation/real_company_backtest/mapped_states.csv`; price/churn/CAC assumed and
labelled; scale-aware physics; no scheduled shocks; 30 matched seeds; results in
`validation/results/real_company_backtest.csv`, n=33 evaluable):

- **Environment retrodiction: FAIL.** The hold arm (each company's own S&M/R&D held
  constant) projects median ≈ +95% 4-quarter revenue growth vs actual +45%; median
  |error| 49.6 pp (criterion: ≤10 PASS / ≤20 PARTIAL). Growth-sign agreement 97%.
- **Diagnosis.** The one *assumed* free parameter of the scale-aware marketing curve —
  `SATURATION_ACQUISITION_RATE = 0.20`/month, flagged as assumption in
  `calibration/bands.json` and `business_logic.py` — is falsified as several times too
  high at real spend intensities. Separately, the model has **no financing mechanism**:
  6 high-burn companies (ASAN, CRWD, DOMO, ESTC, RPD, TENB) go bankrupt in-sim on
  every seed under their own real spend, where the real companies raised capital.
- **Agent increment: no supportable claim.** Boardroom − hold is negative for 100% of
  companies: in a world whose spend response is over-generous, the corridor-limited
  boardroom under-spends relative to real companies. The mandated decomposition did its
  job — without it, the boardroom's median +43.8% (numerically close to the panel's
  actual median +44.7%) could have been mistaken for retrodictive skill; it is in fact
  a nearly company-independent corridor artifact.

Counterfactual language: starting from historical company states at t, the simulation
projected higher outcomes under the hold policy than under the boardroom policy, and
over-projected both relative to observed trajectories. Nothing here estimates what any
agent *would have* achieved at any real company.

## 5b. Tier-2 analyses (completed 2026-08-30, after the Tier-1 package)

**A7 — Robustness across initial conditions: PASS.** 3×3 grid (initial MRR 25/50/100k ×
cash 0.5/1/2M), 20 matched seeds per cell (`validation/results/a7_robustness_grid.csv`):
boardroom ranks 1st on median final MRR in **9/9 cells**; the paired advantage holds in
every cell (g 0.64–0.84 vs every baseline, all Wilcoxon p < 0.05). Caveat recorded:
`random` ranks 2nd by median final MRR in most cells only because MRR-at-bankruptcy
inflates its median — its survival is 4–20%; read survival alongside.

**A5 — Candidate-action regret: exploratory.** 24 decision states (8 seeds × months
6/30/50) × a 48-bundle grid, one-step deviation with boardroom continuation, 10 matched
evaluation seeds (`validation/agents/candidate_regret.csv`). Median regret is **2.2% of
the best candidate's outcome** (max 3.1%); the policy is top-of-grid in 5/24 states
(the calm month-6 states) and mid-pack (rank ~25/49) at post-shock states. Two
findings: (i) single-month deviations are low-stakes — the policy's value accrues from
repeated decisions, consistent with A1; (ii) at every post-shock state the winning
candidate is the maximum-spend grid corner ($40k marketing + $40k R&D), so the residual
regret is "marketing under-spend" against a response curve C1 independently shows is
over-generous — and because the optimum sits on the grid edge, true regret is a lower
bound. This is candidate regret, not global optimality.

**E6 — Drawdown/recovery vs EDGAR: exploratory, and the sharpest structural mismatch
found.** Identical definition on both panels (≥5% decline from running revenue peak;
recovery = regaining the peak; `validation/results/e6_drawdown_recovery.csv`):

| Panel | Episodes/100 quarters | Median depth | Recovery rate | Median recovery |
|---|---|---|---|---|
| EDGAR (39 companies) | 3.0 | 11% | 85% | 3 quarters |
| Sim boardroom (75 episodes) | 1.6 | 61% | 0% | — |
| Sim oracle_v3 | 1.4 | 63% | 2% | 10 quarters |

Real SaaS drawdowns are shallow and quickly recovered; simulated drawdowns are
catastrophic and permanent (hard shocks cut price and raise churn with no mean
reversion). **Important for the paper:** the thesis's 76–80% "recovery" figures are
*Rule-of-40-based* (a flow metric regaining its pre-shock level), not revenue-peak
recovery — the two must never be conflated, and revenue-level recovery essentially
does not happen in the simulator.

**A8 — Post-shock Rule-of-40 recovery: COMPARATIVE.** Event-time analysis around the
scheduled shocks (months 24/48/72) across the deterministic-RNG arms
(`validation/results/a8_shock_r40_curves.csv`, `a8_shock_recovery.csv`; figure
`validation/figures/review/f7_post_shock_r40_recovery.png`): within 24 months of a
shock, the share of events regaining the *pre-shock Rule of 40* is 75–78% for the
oracle arms vs 62–63% for boardroom (n=20 seeds × 3 shocks each), 43% heuristic and
29% no-action (n=50 seeds × 3 shocks); median time-to-recover is 1–2 months for every
policy, so the oracle advantage is in recovery *rate*, not speed. This is Rule-of-40
recovery — a flow metric regaining its pre-shock level — **not** revenue-peak recovery,
which essentially does not occur in the simulator (E6). Consistent with the recorded
FULL-run recovery rates (76–80% vs 67–69%) and with §4.3's finding that the brief
channel responds at shock alerts.

**C2 — Allocation-direction consistency: NULL.** 97 stress company-quarters (growth
< panel p25, spend intensity > median; 25 companies): improved and lagged halves show
statistically indistinguishable allocation-intensity changes (p = 0.65 S&M, 0.86 R&D);
directionally, improvers *grew* absolute S&M and R&D (+6%) while lagged companies
held or cut — the opposite of the agent's cut-marketing stress response. Observational,
survivor-biased, mechanically coupled to growth, and unadjusted for company clustering
— so this is recorded as *absence of observational support*, not a refutation
(`validation/results/c2_allocation_consistency.csv`).

## 5c. physics_v2 recalibration (2026-09-05, branch `physics-v2`)

The C1 diagnosis (§5) was acted on under a pre-frozen split-panel protocol
(`validation/calibration/PROTOCOL.md`: 20 CAL / 19 HOLDOUT companies,
stratified, criteria frozen before any run; HOLDOUT touched exactly once).
Full report: `validation/calibration/calibration_report.md`. All changes sit
behind config flags (`marketing_curve="v2"`, `financing_enabled`,
`corridor="scale_aware"`, `competitive_entry="scale_neutral"`), default
legacy; the recorded v1 results reproduce exactly from the same commit
(Phase 3 gate: per-seed A2 episodes and E1 aggregation byte-identical).

What changed: the one assumed free parameter of the scale-aware marketing
curve was fitted on CAL only (0.20 → 0.0727, bootstrap CI [0.0475, 0.1113]);
a financing rule measured from the panel (R=18mo, K=24.4× monthly burn,
p=0.261/mo) was added as an environment mechanism; every dollar floor/cap in
the boardroom and heuristic corridor became a fraction of current MRR
anchored to EDGAR percentiles (legacy values preserved at the $50k
calibration point); and a D1-audit finding — the competitive-entry shock's
$50k attractiveness anchor saturating at real scale — was neutralized.

HOLDOUT verdicts against the frozen criteria:

| Criterion | Result | Verdict |
|---|---|---|
| C1-v2 median \|4q growth error\| (≤10pp) | **8.1pp** (v1: 49.6pp); sign agreement 100% | **PASS** |
| Corridor artifact (boardroom growth std ≥ actual/3 AND hold-vs-actual ρ>0.3) | ρ=0.73 ✓ but std ratio 0.16 ✗ | **FAIL** |
| Financing (≥80% of v1 all-seed bankrupts survive) | 2/6 survive | **FAIL** |
| Regression (legacy exact) | exact | **PASS** |
| E4-v2 (research-scale spend in EDGAR bands) | 67.6% in [37%, 93%] | **PASS** |

The two failures are reported as results (one-round-trip rule; no re-tuning
against HOLDOUT). Diagnoses: the boardroom variance compression traces to the
backtest *mapping* (identical assumed churn/price/LTV:CAC for every company),
not the corridor floors; the financing rule's rescue-regime parameters
under-finance companies that in reality raised at ~48 months of runway,
long before distress. Research-scale E-battery under v2: E4 FAIL→PASS,
E5 PARTIAL→PASS, E3 PASS, E1 PASS→PARTIAL (median growth 9.3% vs IQR upper
9.1%), E2 PARTIAL (structural — freezing all noise sources still leaves
2× EDGAR volatility and 0.99 persistence; `d5_volatility_attribution.png`).
The boardroom−hold increment, negative for 100% of companies in v1, is
+5.2pp median (positive for 84% of HOLDOUT companies) — still a model-based
counterfactual, not a claim about real companies.

First oracle-at-real-scale numbers (exploratory; 8 HOLDOUT companies × 10
matched seeds, llama3.1:8b): the in-model ordering survives — oracle_v3 beats
the matched-seed boardroom by +11.0pp median paired 4q growth (84% of pairs,
p=2.1e-08) — but the oracle arm sits *further from actual growth* (median
|error| 37.7pp vs boardroom 17.1pp vs hold 8.1pp): the level-blind brief
channel reads LOW/ACCELERATING at these states and scales marketing past the
corridor, and the physics rewards the aggression. Oracle value remains a
strictly simulator-internal claim (`oracle_v3_real_scale_v2_summary.csv`).

## 5d. Round-2 calibration, brief v2, robustness and ablations (2026-09-05, branch `round2`)

Pre-registered before any code (tag `r2-preregistered`;
`validation/calibration/PROTOCOL_round2.md`, `validation/oracle/BRIEF_V2_SPEC.md`);
full report `validation/calibration/calibration_report_round2.md`; session log
`validation/round2/LOG.md`. All flags default to recorded behaviour; round-1
HOLDOUT numbers and legacy research numbers reproduce exactly on this branch.

**Round-2 calibration (EVAL2 = the 19 round-1 HOLDOUT companies at q0+8,
touched once; two changes only — company-specific CAC mapping and
opportunistic financing from CAL-only hazard; the fitted `s` frozen).**
R2-C1 **PARTIAL** (median |4q error| 14.8pp out-of-time, signed +8.4pp;
DEV2 15.7pp; $50 mapping 13.8pp). R2-SIGN **PASS** (100%). R2-CORR **FAIL,
inverted vs round 1**: boardroom growth std ratio 1.09 now passes (the CAC
mapping produces real state variety, DEV2 LTV:CAC IQR 1.07) but hold-arm
Spearman falls to 0.18 — at q0+8 every company's real spend sits in the
fitted curve's saturation region, so the hold arm projects ≈43% for everyone
while actual growth still varies. R2-FIN-a **PASS** (100% survive under
hold); R2-FIN-b **N/A** (no zero-survival company exists at q0+8 with
financing off — premise empty, recorded). Zero DEV2 fix iterations; one
EVAL2 run; no round 3.

**Brief v2: B1 FAIL by the frozen rule; brief v1 stays.** The deterministic
level block fixes runway responsiveness outright (sweep ρ 0.00→0.97) and the
v2b floors move the new LTV:CAC sweep (ρ=0.65; floor share 13%), but
churn/confidence/competitors sweeps stay flat (1/4 < 3/4): the
ActionModifier's arithmetic never consumes macro_condition, and
innovation_urgency stays flat. B2/B3 were not run; the A4 level-blindness
FAIL remains a limitation; no prompt was modified after reading results.

**Robustness under v2 physics (research scale).** A2 ordering survives:
boardroom > noop/random/heuristic on final MRR (paired g 0.78–0.82, Holm
p≈2e-14, 50 seeds; caveat: boardroom post-shock Rule-of-40 is worse than
noop/heuristic under v2 — heavy corridor spend hurts the margin term). The
**oracle layer's advantage does not survive**: oracle_v3 > boardroom in only
8/20 seeds (mean paired diff −$110k, Wilcoxon p=0.45, both arms 100%
survival). **Every oracle-layer value claim is therefore
calibration-sensitive** — real under the legacy physics the thesis recorded,
null under the CAL-fitted marketing curve, where saturation removes the
payoff of the brief-driven spend-up (decomposed component-by-component in
the addendum-A paragraph below).

**Random-shock-timing ablation (legacy physics, 20 matched seeds).**
RS-1 **PASS**: oracle_v3 > boardroom in 20/20 seeds with per-episode random
schedules (mean +$1.30M, p=1.9e-06) — the oracle advantage does not rest on
the learnable fixed {24,48,72} timetable. RS-2 **FAIL**: the retrieval
increment is not detectable under random timing (v3 − v3_no_memory mean
+$11.6k, 95% CI [−$0.6k, +$25.0k], 10/20 seeds) — the recorded +$37.9k
(≈3%) increment carries a fixed-timetable qualifier at n=20 (superseded by
the pre-declared RS-2x pooled n=40 result below). Post-shock R40
recovery within 24 months: boardroom 68%, oracle 78%, no-memory 73%.

**Case study** (frozen ranking rule over the recorded A3 replication;
`validation/round2/case_study.md`, figure `f8_case_study_seed15.png`):
seed 15, month 60 — the memory arm read LOW risk vs MEDIUM, spent 48% more
on marketing, +10.4% MRR six months later; illustrates the
retrieval→brief→modifier mechanism, quantified only by the paired A6
ablation. Second-LLM A4 sensitivity: `a4_level_sweeps_models.csv`.

**Recoverable shocks (S11 stretch; `shock_recovery="mean_revert"`, legacy
physics).** Hard-shock price/churn damage gets a 3-month half-life (~87.5%
recovered after 9 months, matching EDGAR's median 3-quarter drawdown
recovery); defaults byte-identical. The oracle advantage does not flip — it
**strengthens** (20/20 seeds, mean paired diff +$2.23M, p=1.9e-06; both arms
100% survival). E6 recomputed on these episodes
(`e6_drawdown_recovery_mr.csv`): median drawdown depth falls from the
recorded ~61–63% to **16–17%** (EDGAR 11%) — the depth half of the E6
structural mismatch is repaired — while drawdowns become rare (0.4–0.5 per
100 quarters vs EDGAR's 3.0; the sim lacks ordinary demand-side dips) and
the recovery-rate cell is unestimable from 3–4 censored episodes. E-battery
rows under mean-revert for both physics: scorecard tests `E1-mr`–`E5-mr`
(legacy: E1/E3 PASS, E2/E5 PARTIAL, E4 FAIL at 8.0% spend — the known
legacy failure mode; v2: E3/E4/E5 PASS, E1/E2 PARTIAL).

**Oracle-layer decomposition (addendum A, pre-registered 2026-09-05 and run
overnight into 2026-09-06; tag
`addendum-a-preregistered`, `validation/calibration/PROTOCOL_addendum_A.md`;
gates `validation/round2/gates_decomp.py`, results
`validation/results/a3_decomp_*.csv`).** Six arms, frozen criteria
(oracle arm > paired boardroom on final MRR in ≥15/20 seeds) and frozen
interpretation rules, committed before any run; all v2 arms match the
recorded a3_v2phys config (financing off — deviation from the addendum's
common-config line recorded in LOG.md before any run). Results:
* **D-a** `oracle_v3_no_modifier` **FAIL 0/20** — with the ActionModifier
  off, the outcome path is *identical* to the boardroom on every seed. The
  brief-adjusted proposal weights only rescale proposal confidences; the
  assembled action never reads them, so **the modifier's spend multipliers
  are the brief's only causal channel into actions**.
* **D-b** `modifier_bound="tier"` **FAIL 8/20** (mean +$20.4k, 95% CI
  [−$53.8k, +$103.5k], p=0.78) — capping the modifier's spend-up at the
  corridor's top tier does not restore value under the fitted curve.
* **D-c** v2 + mean-revert (new boardroom pair) **FAIL 7/20** (mean −$284k,
  CI [−$721k, +$23k]) — shock recoverability does not restore it either.
* **D-d** qwen2.5:7b-instruct, v2 physics **PASS 15/20** (mean +$28.8k, CI
  [−$41.9k, +$98.1k], p=0.12 — win-count criterion met; magnitude small).
  Frozen rule fires: **the null under v2 physics is LLM-specific**.
* **L-1** qwen2.5:7b-instruct, legacy physics **PASS 20/20** (mean +$564k,
  CI [+$306k, +$875k], p=1.9e-06) — the headline oracle claim carries no
  llama3.1:8b qualifier.
* **RS-2x** (pre-declared extension to seeds 21–40, legacy, random
  schedules): pooled n=40 retrieval increment (v3 − v3_no_memory) mean
  **+$39.7k, 95% CI [+$18.0k, +$65.9k]**, positive in 23/40, p=0.0014;
  extension cohort alone +$67.8k, CI [+$27.5k, +$115.7k]. Frozen rule
  fires: **small but detectable at n=40** — the n=20 RS-2 null reflected
  power, not absence.

The A2 robustness panel also passes in the most realistic configuration —
v2 physics + mean-revert, 50 seeds: boardroom > noop/random/heuristic on
final MRR, paired g 1.21–1.68, Holm p≈1.6e-14
(`statistical_tests_policy_baselines_v2phys_mr.csv`; run summaries in
`policy_comparison_v2phys_mr.csv`).

## 6. Leakage and implementation problems found

(Full audit: `validation/system_audit.md` §5.)
1. No look-ahead into prompts; memory maturation waits for realized outcomes; Chroma
   retrieval is run_id-scoped. Within-run cross-episode learning is by design and must
   be described (episodes are sequential, not i.i.d., for memory arms).
2. Fixed shock timetable (months 24/48/72) is in principle learnable by memory —
   bounded threat, disclosed as a limitation of the shock-handling claim.
3. The `random` policy is not comparable to anything in legacy-RNG runs (it perturbs
   the world); every new comparison used `deterministic_rng`.
4. Curated "screenshot" summary files; uncommitted-only v4-hetero results; an aborted
   empty confirmation dir — recorded in the claim audit.
5. oracle_v4 ≡ oracle_v3 absent shock-time graph context; v4_causal_hetero (the only
   true LLM-composed-action policy) *lost* its only recorded comparison (0.75 vs 1.00
   survival) — treated as out of scope for headline claims.

## 7. Answers to the eleven questions

1. **Realistic enough as a controlled environment?** Qualified yes at research scale
   for *comparative* policy experiments (E1/E3 pass; E2/E5/E4 documented); no as a
   predictive model (C1).
2. **Do agent actions affect outcomes?** Yes — causally, with per-lever magnitudes
   quantified (A1).
3. **Are decisions economically sensible?** Rule layer yes (10/10); LLM layer
   partially — trend/shock-sensitive, level-blind (A4). The level-blindness is
   substantially a model property, not only a prompt property: qwen2.5:7b-instruct
   reads churn and confidence levels unaided (2/4 sweeps) where llama3.1:8b
   reads none (0/4); the brief-v2 level block fixes the runway dimension on
   both models (`a4_level_sweeps_models.csv`).
4. **Do agents beat simple controls?** Yes, in-sim, paired, medium-to-large effects
   (A2; A3 recorded-paired).
5. **Does retrieval/context improve decisions?** Briefs are predictive (+17 pp,
   A6i); retrieval changes ~60% of monthly spend decisions and yields a small,
   significant outcome gain (+$37.9k, CI [+14.0k, +66.3k], 15/20 seeds) — ≈3% of the
   oracle layer's total value; the brief mechanism, not memory, carries the bulk.
   Qualifier: under random shock timing the increment is not detectable at n=20
   (RS-2 FAIL) but is detectable at the pre-declared pooled n=40 (RS-2x:
   +$39.7k, CI [+$18.0k, +$65.9k], p=0.0014) — small, real, and an order of
   magnitude below the brief channel.
6. **Robust across seeds/states?** Yes: 74–75/75 recorded seeds, 20/20 replicated
   under clean RNG, and the boardroom ranking holds in 9/9 initial-condition cells
   (A7) with paired g 0.64–0.84 throughout. Not under v2 physics with the primary
   LLM: the oracle increment is null under the fitted marketing curve (8/20), and
   the addendum-A decomposition shows the entire effect flows through the
   ActionModifier (the no-modifier arm is outcome-identical to the boardroom,
   0/20), with neither tier-bounding (8/20) nor shock recoverability (7/20)
   restoring it — while qwen2.5:7b-instruct passes the same pre-registered
   criterion (15/20), making the v2-physics null LLM-specific; the legacy
   headline replicates on qwen 20/20.
7. **Real-state initialization — agent beats controls?** No. Increment negative
   everywhere; see §5.
8. **Simulated vs observed real trajectories?** After the pre-registered
   recalibration, the simulator retrodicts held-out companies to a median |4q
   error| of 8.1 pp at the calibration horizon (C1-v2 PASS) and 14.8 pp
   out-of-time at q0+8 (R2-C1 PARTIAL), with 100% growth-sign agreement; the
   original physics over-projected by ~45–50 pp per 4 quarters.
9. **Which conclusions are empirical?** The EDGAR panel statistics; the claim audit;
   brief-label distributions; all paired in-sim comparisons *as statements about the
   simulator*.
10. **Which depend on simulator-based counterfactuals?** Every "agent adds value"
    statement (A2, A3, C1 arms) — value is defined inside the model's dynamics.
11. **What remains unvalidated?** Churn/CAC/price/product-quality realism, macro
    block, monthly dynamics, hiring, v4 causal-graph contribution; drawdown
    *frequency* under mean-revert (0.4–0.5/100q vs EDGAR 3.0) — the new
    unvalidated shock item now that drawdown *depth* is addressed by the
    `shock_recovery="mean_revert"` flag (61–63% → 16–17%, S11). (Moved to
    validated since the first draft: retrieval outcome-increment — small positive,
    detectable at pooled n=40 under random timing; robustness grid — PASS;
    candidate regret — exploratory results in §5b; allocation-direction
    consistency — null.)

## 8. Strongest defensible claims, and required paper language

**Simulator:** "At its calibration scale, the environment reproduces the quarterly
revenue-growth distribution (median within the EDGAR panel's IQR) and the
growth-deceleration-with-scale regularity of 39 public SaaS companies (1,288 quarters),
while exhibiting higher persistence than the panel and, by default, far deeper
non-recovering revenue drawdowns under its injected shocks — a pre-registered
mean-revert recovery flag repairs the drawdown depth (61–63% → 16–17%, EDGAR
11%) at the cost of making drawdowns rarer than the panel's (0.4–0.5 vs 3.0
per 100 quarters). Under the original physics it over-projected four-quarter
growth from real company states by ≈+50 pp;
after a pre-registered split-panel recalibration of its one assumed marketing
parameter it retrodicts held-out companies to a median |error| of 8.1 pp at the
calibration horizon (PASS) and 14.8 pp out-of-time eight quarters later
(PARTIAL), with 100% growth-sign agreement, while cross-company dispersion
remains under-produced (corridor criterion FAIL in both rounds, for opposite
reasons); it serves as a controlled comparative testbed rather than a
forecasting model."

**Agents:** "Under matched random worlds with the recorded (legacy) physics, the
boardroom policy improves final MRR over no-action, random and heuristic
controls (paired Hedges g 0.60–0.82, n=50; g 0.78–0.82 under the recalibrated
physics, 1.21–1.68 with recoverable shocks added), and the LLM-oracle layer
improves it further in 74–75 of 75 paired
episodes (g ≈ 0.93, p < 1e-13) — a result that survives randomized shock
timing (20/20 seeds), recoverable shocks (20/20, +$2.23M), and replication on
a second LLM (qwen2.5:7b-instruct, 20/20), but is **calibration-sensitive**:
under the recalibrated marketing curve the oracle increment is null with
llama3.1:8b (8/20 seeds, p=0.45). A pre-registered decomposition locates the
sensitivity in the ActionModifier's spend multipliers — the brief's only
causal channel into actions (without them the oracle arm is outcome-identical
to the boardroom on all 20 seeds; capping them at the corridor's top tier
does not restore value, 8/20; nor do recoverable shocks, 7/20) — and shows
the recalibrated-physics null is LLM-specific: qwen2.5:7b-instruct meets the
same pre-registered criterion under the fitted curve (15/20 seeds, mean
+$28.8k with a CI including 0). All effects
are simulator-internal and are reported as model-based counterfactuals."

Do not claim: v4 ≠ v3; reward improvements; retrieval as the main source of value
(it contributes ≈3% of the oracle gain; under random shock timing it is
detectable only at the pre-declared pooled n=40, +$39.7k, not at n=20);
oracle value under the recalibrated (v2) physics with llama3.1:8b — and the
qwen 15/20 win-count PASS is reported as LLM-sensitivity of the null, not as
established v2-physics value (its CI includes 0); any
real-world agent value; any real-company forecast.

## 9. Before submission / limitations

Must complete: state the E4 payroll-definition caveat wherever spend ratios appear;
replace any figure sourced from the screenshot-variant summaries. Should state as
limitations: single primary LLM (llama3.1:8b) and its level-blindness (A4 FAIL;
brief-v2 level block fixes the runway dimension only — B1 FAIL; second-LLM
sensitivity in `a4_level_sweeps_models.csv`); oracle value is
calibration-sensitive (§5d); the retrieval increment carries a fixed-timetable
qualifier (RS-2) at n=20, resolved as small-but-detectable at the
pre-declared pooled n=40 (RS-2x); churn tenure-decay bias in the backtest
mapping (~20pp
mechanism, absorbed by the fit); round-1 under-financing (2/6 rescued) and the
round-2 rescue of it evaluated only at q0+8; assumed price/churn in the
backtest mapping (round-2 CAC is company-specific, clamped, no look-ahead);
EVAL2 reuses the round-1 HOLDOUT companies at later quarters (same-company
correlation disclosed); the round-1 D4 financing parameters were panel-wide
(round-2 hazard is CAL-only); oracle memory tiers saturate at real scale;
memory arms learn across episodes within a run. From the addendum-A
decomposition: the D-a/D-b/D-c arms are llama-only (the modifier-channel
decomposition was not repeated on qwen); the second-LLM evidence is a single
7B-class model, and D-d's PASS is a win-count result whose CI includes 0;
drawdown frequency under mean-revert (0.4–0.5/100q vs EDGAR 3.0) replaces
drawdown depth as the open shock-realism item; E4 remains FAIL under legacy
physics even with mean-revert (spend share ≈8.0% — the flag does not touch
spend). Reproduction commands:
`validation/README.md`.
