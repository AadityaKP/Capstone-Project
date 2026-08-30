# Validation plan — restructured for what actually exists

Written 2026-08-30 **after** `system_audit.md` and `edgar_data_audit.md`, and before any
new result was computed. Acceptance criteria in §4 were fixed at writing time.
Hard budget: 2 days on this machine (non-LLM episodes ≈ 0.02 s; LLM briefs via local
llama3.1:8b, ~10–40 s each; Neo4j up; Ollama installed).

---

## 1. Critique of the existing (de facto) validation plan

There is no `validation_plan.md` in the repo. The de facto plan = the thesis experiment
suite (`experiments/thesis_analysis.py` scenarios + Mann-Whitney tests), the July
`evidence_audit.md`, and the generic toolbox in the project owner's brief. Item by item:

| Existing item | Verdict | Why |
|---|---|---|
| boardroom vs oracle_v1 vs oracle_v3, n=75, Mann-Whitney | **KEEP (as recorded evidence) + MODIFY** | The data exist and internally reproduce. But (a) unpaired tests on a paired design waste power; (b) no trivial baselines (no-action/random/heuristic) were ever run, so "the agent adds value" currently means "adds value over one specific rule-based board", not over doing nothing; (c) re-analyse paired, add trivial baselines under `deterministic_rng`. |
| CONFIRMATION run (5 policies, n=50) | **KEEP with caveat** | Reproduction, not replication (same seeds 0–49 ⊂ 0–74). oracle_v4 row must never be cited as distinct from v3 (identical to 9 digits). |
| oracle_v4 / v4_causal comparisons | **LOWER PRIORITY** | v4 ≡ v3 absent shock-time graph context; v4_causal_hetero lost its only recorded run. Not the claim to defend this week. |
| `primary_summary_screenshot_no_oracle_v3.*` | **REMOVE from any use** | Curated copy with a policy row deleted; use canonical `primary_summary.csv`. |
| Memory ablation scenarios (`MEMORY_ABLATION_SCENARIOS`) | **MODIFY** | Correctly designed (4 cells, deterministic_rng) but never executed. Full grid needs Neo4j arms → Tier 2/3. The episodic cell (v3 vs v3_no_memory) is the part the thesis needs. |
| Reward as outcome metric | **REMOVE** | Sign-disagrees with every headline metric across policies (system_audit §3.4). Diagnostic only. |
| Monthly rows as independent samples | **REMOVE** | All new stats use episodes (or companies) as units; paired where seed-matched. |
| RAGAS-style retrieval metrics | **NOT POSSIBLE / SKIP** | No ground-truth corpus; and the retrieval claim is about *decisions*, not text similarity. Replaced by brief-accuracy and decision-delta tests. |
| "Validate against 50 EDGAR companies" (brief's default) | **REPLACE** | The repo already has a screened 39-company panel with declared inclusion criteria and 1,288 complete quarters. Adopt it; do not pad to a round number. |
| Absolute-dollar comparison to EDGAR | **NOT POSSIBLE** | Two orders of magnitude of scale mismatch; use scale-free quantities only (edgar_data_audit §4). |
| Churn/CAC/price validation vs EDGAR | **NOT POSSIBLE** | Not in XBRL. Churn constants get a *range check* vs ChartMogul bands already in `calibration/bands.json`; CAC/price declared unvalidatable. |
| Macro block realism (rates, unemployment, confidence) | **SKIP (declared limitation)** | Out of EDGAR scope; validating it would not strengthen the two claims being defended. |

The two claims the package must defend:
**(S)** the environment reproduces the scale-free regularities of real SaaS well enough
to be a credible testbed; **(P)** the agent layer makes economically sensible decisions
that causally improve simulated outcomes over trivial and rule-based controls.

## 2. The plan

### A. Environment validation (vs the 39-company EDGAR panel; sim aggregated to quarters)

| ID | Test | Data | Method |
|---|---|---|---|
| E1 | Revenue-growth distribution | sim monthly traces (recorded FULL boardroom arm + new no-LLM runs) → QoQ; EDGAR 1,288 quarters | median/IQR/p10–p90, KS, Wasserstein |
| E2 | Growth persistence | same | within-unit lag-1 autocorrelation of QoQ growth |
| E3 | Growth deceleration with scale | same | within-unit corr(growth, log revenue); direction + magnitude |
| E4 | Expense-structure validity | sim action traces (marketing+R&D)/MRR; EDGAR S&M%+R&D% | distribution comparison per policy |
| E5 | Volatility | same as E1 | within-unit std of QoQ growth |
| E6 | Drawdown/recovery (exploratory) | EDGAR revenue-decline episodes vs sim shock-recovery events | frequency, depth, duration — comparative only, mechanisms differ |
| E7 | Range/constraint validity | sim churn/margin constants vs ChartMogul bands & EDGAR GM | inside/outside published range |

Justification: these are exactly the quantities EDGAR actually provides (audit §2), they
are scale-free (audit §4), and each one corresponds to a mechanism the simulator claims
to have (growth engine, spend response, deceleration via CAC/competition, shocks).

### B. Agent/policy validation

| ID | Test | Method |
|---|---|---|
| A1 | Action-effect (do actions matter?) | Same state, same seed (`deterministic_rng`), ladder over each action dimension separately; 12-month outcome deltas; also full-horizon from t=0 |
| A2 | Policy baselines (no-LLM) | noop / random / heuristic / boardroom, 50 seeds × 120 months, seed-matched; paired Wilcoxon + Hedges-g on final MRR, survival, post-shock R40 |
| A3 | Oracle value, matched seeds (LLM) | boardroom vs oracle_v3 (+ oracle_v3_no_memory if time), 12 seeds × 120 months, freq 10, deterministic_rng, temperature 0; paired analysis; positioned as a matched-seed spot-replication of the recorded n=75 result |
| A4 | State responsiveness / decision sensibility | (i) rule agents: direct evaluation of the documented thresholds (deterministic); (ii) LLM brief pathway: one-variable sweeps (runway ↓, churn ↑, competitors ↑, confidence ↓) + 6 adversarial states → brief severity and resulting modifier multipliers; monotonicity via Spearman |
| A5 | Candidate-action regret (Tier 2) | ~20 sampled decision states × candidate grid (~54 bundles) × 10 seeds × 12 months; regret of policy action vs best candidate; explicitly *candidate* regret, not global optimality |
| A6 | Retrieval value | (i) recorded-data: oracle_v3 brief `expected_outcome` vs realized 6-month outcome (FULL run, n=75 episodes of traces) — does memory-conditioned prediction beat base rates; (ii) decision deltas + outcomes v3 vs v3_no_memory from A3 arms (underpowered; labelled) |
| A7 | Robustness (Tier 2) | A2 arms × {initial_mrr 25/50/100k, initial_cash 0.5/1/2M} × 20 seeds; ranking stability |
| A8 | Shock handling | from A2/A3 traces + recorded recovery events; recovery rate/time by policy |

Justification: A1 establishes causal traction (the precondition for everything);
A2 supplies the missing trivial-baseline evidence; A3 tests the thesis's actual claim
under the clean RNG regime the repo itself says is required for ablations; A4 tests the
*decision mapping* directly and cheaply where the LLM's entire influence is a
deterministic function of the brief (system_audit §3.2–3.3).

### C. Real-company counterfactual (EDGAR backtest)

C1. For each panel company: initialization quarter = earliest complete-core quarter with
cash and ≥4 subsequent actual quarters. Map: mrr = revenue/3; cash = cash+STI;
monthly_burn = G&A/3 (fixed opex); hold-arm discretionary spend = actual S&M/3 and
R&D/3; gross_margin = company's own GM; churn = ChartMogul band median for the assumed
ARPA band; price, CAC, product_quality **assumed and labelled** (sensitivity variant
shifts price band ±1). Physics: scale-aware flags on, scheduled research shocks off,
`deterministic_rng` on, 12 months × 30 seeds.

Arms: `hold` (company's own spend, held) · `noop` (no discretionary spend) ·
`heuristic` (scaled) · `boardroom` (scaled). Decomposition, mandated:

- **Environment retrodiction error** = sim `hold` 4-quarter growth − actual 4-quarter growth.
- **Agent incremental effect** = Outcome(boardroom) − Outcome(hold), same seeds, in-sim.
- Separately: sim agent outcome vs actually observed company outcome, reported in
  counterfactual language only ("starting from the historical state at t, the simulation
  projected …"), never as "the agent would have beaten company X".

C2 (Tier 2, observational): direction-of-allocation consistency — among EDGAR
company-quarters resembling agent stress states (low growth + high spend), did companies
that subsequently improved shift S&M/R&D intensity in the direction the agent moves them?
Supporting evidence only, no causal claim.

## 3. Prioritization

Scale: value/effort H/M/L. "Def/hr" = defensibility per hour, the ranking criterion.

| Analysis | Research value | External validity | Agent validity | Effort | Time | Def/hr | Tier |
|---|---|---|---|---|---|---|---|
| Claim audit (recompute recorded headline numbers) | H | — | H | L | 1h | H | **1** |
| E1–E5, E7 environment battery | H | H | — | M | 3h | H | **1** |
| A1 action-effect | H | — | H | L | 2h | H | **1** |
| A2 trivial baselines, matched seeds | H | — | H | L | 2h | H | **1** |
| A4 state responsiveness (incl. LLM sweeps) | M | — | H | M | 3h | H | **1** |
| C1 EDGAR backtest (39 companies) | H | H | M | M | 4h | H | **1** |
| A6(i) brief-accuracy from recorded traces | M | — | M | L | 1h | M | **1** |
| Scorecards + report | H | H | H | M | 3h | H | **1** |
| A3 matched-seed LLM replication (n=12) | H | — | H | M | 3–5h wall (mostly unattended) | M | **2 (start early, runs in background)** |
| A5 candidate regret | M | — | M | M | 3h | M | 2 |
| A7 robustness grid | M | — | M | L | 1h | M | 2 |
| E6 drawdown/recovery comparison | M | M | — | M | 2h | M | 2 |
| C2 allocation-direction consistency | L | M | L | M | 3h | L | 2 |
| Full 4-cell memory ablation (Neo4j arms) | M | — | M | H | 8h+ | L | **3 — skip** |
| oracle_v4_causal_hetero validation | L | — | L | H | 10h+ | L | **3 — skip** |
| Macro-block realism | L | L | — | H | — | L | **3 — skip** |
| Matched-state causal inference beyond C1/C2 | L | M | L | H | — | L | **3 — skip** |

## 4. Acceptance criteria (fixed before computing results)

Where a published reference exists, thresholds use it; otherwise the test is declared
**comparative** or **exploratory** and reports magnitudes with uncertainty, not verdicts.

| ID | Criterion |
|---|---|
| E1 | PASS: sim median QoQ growth ∈ EDGAR [p25, p75] and the two IQRs overlap. PARTIAL: median ∈ [p10, p90]. Else FAIL. KS/Wasserstein reported descriptively (no universal threshold — populations differ by construction). |
| E2 | PASS: same sign and \|Δautocorr\| ≤ 0.25. PARTIAL: same sign. FAIL: sign flip. |
| E3 | PASS: negative growth–scale relationship in both. PARTIAL: negative in EDGAR, ~0 in sim. FAIL: positive in sim. |
| E4 | PASS: sim median discretionary-spend ratio ∈ EDGAR [p10, p90] of (S&M+R&D)%. PARTIAL: within [min, max]. Else FAIL (per policy). |
| E5 | PASS: sim median within-unit growth volatility within ×2 of EDGAR median. PARTIAL: ×4. Else FAIL. |
| E6 | Exploratory — no verdict. |
| E7 | PASS: each calibrated constant inside its cited band. |
| A1 | PASS: marketing and pricing ladders produce monotone outcome ordering with terminal-MRR spread ≥ 10% of the mid-rung outcome; every lever's effect (incl. ~0 for legacy R&D) reported. FAIL only if *no* lever moves outcomes ≥ 5%. |
| A2 | PASS: boardroom > noop AND boardroom > random on final MRR, paired Wilcoxon Holm-adjusted p < 0.05, Hedges g ≥ 0.3, plus no survival disadvantage. Heuristic-vs-boardroom: comparative. |
| A3 | PASS (as replication): oracle_v3 − boardroom paired difference on post-shock R40 and final MRR has the same sign as the recorded n=75 result and its 95% bootstrap CI excludes the opposite sign on ≥1 of the 2 metrics. n=12 is disclosed as small; this arm confirms direction under clean RNG, it does not re-estimate magnitude. |
| A4 | PASS (rules): each documented threshold reproduces exactly. PASS (LLM): Spearman monotonicity in the expected direction, \|ρ\| ≥ 0.5, for ≥ 3 of 4 sweeps (runway→risk, churn→R&D scale, confidence→marketing scale, competitors→risk). PARTIAL: 2 of 4. |
| A5 | Exploratory: report median candidate regret as % of best-candidate outcome. |
| A6(i) | PASS: v3 brief expected_outcome accuracy > majority-class base rate with 95% CI excluding the base rate (episode-clustered bootstrap). |
| C1-env | PASS: median \|4-quarter cumulative growth error\| ≤ 10 pp AND growth-sign agreement ≥ 70% of companies. PARTIAL: ≤ 20 pp. Else FAIL. |
| C1-agent | Claim "in-model value at real scale" only if boardroom − hold increment > 0 for ≥ 2/3 of companies with paired (per-company, across-seed) significance; otherwise report distribution without the claim. |

## 5. Execution order and fallbacks

Day 1: claim audit → E-battery → A1 → A2 → start A3 in background (Ollama) → A4 sweeps
→ C1 mapping + runs. Day 2: A3 harvest, A6, scorecards, report; Tier 2 as time allows.

Fallbacks, pre-committed:
- If Ollama cannot serve reliably: A3 downgrades to the recorded-run reanalysis (paired
  where possible) + A4(ii) is dropped; the retrieval claim rests on A6(i) and is
  labelled weaker. No LLM number is simulated by a stand-in.
- If C1 mapping proves unstable at EDGAR scale (e.g., scale-aware physics still
  diverges), the backtest reports the failure as a **negative result** with diagnosis —
  it is not tuned until it passes. (No re-calibration against the validation panel.)
- Anything not finished lands in `results/provisional_placeholders.csv` with
  status `SYNTHETIC_NOT_FOR_PUBLICATION`; the report separates VERIFIED / EXECUTED
  from SYNTHETIC / PROVISIONAL.

Output layout (respecting existing repo structure — EDGAR stays in `data/`):

```
validation/
  system_audit.md  agent_audit.md  edgar_data_audit.md  validation_plan.md  README.md
  analysis/        environment + agent + backtest scripts (each writes results/)
  agents/          decision_log.csv, action_effects, policy comparison outputs
  real_company_backtest/  mapped_states.csv, backtest outputs
  results/         environment_scorecard.csv, agent_scorecard.csv, policy_comparison.csv,
                   statistical_tests.csv, real_company_backtest.csv, claim_audit.csv,
                   validation_summary.csv, provisional_placeholders.csv
  figures/
report/validation_report.md
```
