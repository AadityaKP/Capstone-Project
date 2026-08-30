# Validation report — startup simulator and multi-agent decision layer

Date: 2026-08-30 · Branch `review2-sim-frontend` · Full methodology and pre-declared
acceptance criteria: `validation/validation_plan.md` (criteria fixed before any result
was computed; deviations are labelled where they occur).

**Status legend.** Every number in this report is VERIFIED/EXECUTED from data on disk.
No synthetic or placeholder values appear anywhere in this report. (The A3 live
replication, pending in an earlier draft, completed on 2026-08-30 and is incorporated.)

---

## 1. Summary verdicts

**SIMULATOR: C — IMPORTANT VALIDATION GAPS.**
At its calibration scale the environment reproduces several scale-free regularities of
real SaaS (growth-rate distribution, growth deceleration with scale) but is too smooth
and too volatile at once (persistence and volatility both off), its policies spend far
less of revenue than real SaaS companies do, its shocks produce catastrophic permanent
revenue drawdowns where real SaaS drawdowns are shallow and quickly recovered (§5b E6),
and when initialized from real company states it over-projects growth by ~45–50
percentage points over four quarters. It is
usable as a *controlled research environment* for comparing policies under identical
dynamics; it is not currently supported as a *predictive model* of real company
trajectories.

**AGENTS: B — EVIDENCE OF POLICY VALUE WITH IMPORTANT LIMITATIONS.**
Actions causally move outcomes; the boardroom beats no-action, random and plain
heuristic controls at matched seeds with medium-to-large paired effects; the
oracle layer beats the boardroom in 74–75 of 75 matched seeds in the recorded thesis
run (paired g ≈ 0.93 on final MRR) and in 20/20 seeds in a fresh replication under the
clean RNG regime; memory-conditioned briefs are genuinely predictive (+17 pp over base
rate), though episodic retrieval itself contributes only ≈3% of the oracle layer's
outcome gain. The limitations are material: all demonstrated value is
*in-simulator at research scale*; the LLM brief channel responds to trends and shock
alerts but is blind to state levels; and the real-scale counterfactual does not support
any claim of agent value on real company states.

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

### 4.4 Does the oracle layer add value over the boardroom? (A3 — PASS, recorded; ⏳ live replication running)
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

**C2 — Allocation-direction consistency: NULL.** 97 stress company-quarters (growth
< panel p25, spend intensity > median; 25 companies): improved and lagged halves show
statistically indistinguishable allocation-intensity changes (p = 0.65 S&M, 0.86 R&D);
directionally, improvers *grew* absolute S&M and R&D (+6%) while lagged companies
held or cut — the opposite of the agent's cut-marketing stress response. Observational,
survivor-biased, mechanically coupled to growth, and unadjusted for company clustering
— so this is recorded as *absence of observational support*, not a refutation
(`validation/results/c2_allocation_consistency.csv`).

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
   partially — trend/shock-sensitive, level-blind (A4).
4. **Do agents beat simple controls?** Yes, in-sim, paired, medium-to-large effects
   (A2; A3 recorded-paired).
5. **Does retrieval/context improve decisions?** Briefs are predictive (+17 pp,
   A6i); retrieval changes ~60% of monthly spend decisions and yields a small,
   significant outcome gain (+$37.9k, CI [+14.0k, +66.3k], 15/20 seeds) — ≈3% of the
   oracle layer's total value; the brief mechanism, not memory, carries the bulk.
6. **Robust across seeds/states?** Yes: 74–75/75 recorded seeds, 20/20 replicated
   under clean RNG, and the boardroom ranking holds in 9/9 initial-condition cells
   (A7) with paired g 0.64–0.84 throughout.
7. **Real-state initialization — agent beats controls?** No. Increment negative
   everywhere; see §5.
8. **Simulated vs observed real trajectories?** Simulation over-projects growth by
   ~45–50 pp per 4 quarters at real scale.
9. **Which conclusions are empirical?** The EDGAR panel statistics; the claim audit;
   brief-label distributions; all paired in-sim comparisons *as statements about the
   simulator*.
10. **Which depend on simulator-based counterfactuals?** Every "agent adds value"
    statement (A2, A3, C1 arms) — value is defined inside the model's dynamics.
11. **What remains unvalidated?** Churn/CAC/price/product-quality realism, macro
    block, monthly dynamics, hiring, v4 causal-graph contribution. (Moved to
    validated since the first draft: retrieval outcome-increment — small positive;
    robustness grid — PASS; candidate regret and drawdown/recovery — exploratory
    results in §5b; allocation-direction consistency — null.)

## 8. Strongest defensible claims, and required paper language

**Simulator:** "At its calibration scale, the environment reproduces the quarterly
revenue-growth distribution (median within the EDGAR panel's IQR) and the
growth-deceleration-with-scale regularity of 39 public SaaS companies (1,288 quarters),
while exhibiting higher persistence and volatility than the panel and far deeper,
non-recovering revenue drawdowns under its injected shocks; initialized from real
company states it over-projects four-quarter growth (median error ≈ +50 pp), so it
serves as a controlled comparative testbed rather than a forecasting model."

**Agents:** "Under matched random worlds, the boardroom policy improves final MRR over
no-action, random and heuristic controls (paired Hedges g 0.60–0.82, n=50), and the
LLM-oracle layer improves it further in 74–75 of 75 paired episodes (g ≈ 0.93,
p < 1e-13), with the gain concentrated in post-shock periods — all effects are
simulator-internal and are reported as model-based counterfactuals."

Do not claim: v4 ≠ v3; reward improvements; retrieval as the main source of value
(it contributes ≈3% of the oracle gain); any real-world agent value; any
real-company forecast.

## 9. Before submission / limitations

Must complete: state the E4 payroll-definition caveat wherever spend ratios appear;
replace any figure sourced from the screenshot-variant summaries. Should state as limitations: single LLM (llama3.1:8b) and its level-blindness;
no financing mechanism; assumed price/churn/CAC in the backtest mapping; benchmark
calibration post-dates backtest initialization states; memory arms learn across
episodes within a run. Reproduction commands: `validation/README.md`.
