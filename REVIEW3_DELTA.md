# REVIEW3_DELTA — what changed from Review 2 to Review 3

Baseline: `review-2` tip = merge-base `3280103` ("Restore n=75 prioritized thesis results", 2026-04-05).
Head: `review-3` @ `8185d61`. **47 commits (43 non-merge), 2026-04-13 → 2026-09-01.**
Everything below is traceable to `git diff 3280103..HEAD` or the commit log; uncertainty is flagged inline.

---

## 1. Executive summary (slide-ready)

- **The system is now validated, not just built:** a pre-declared validation plan graded the simulator **C (controlled comparative testbed, not a forecaster)** and the agent stack **B** — the oracle layer beats the boardroom in **74–75 of 75 recorded matched seeds (paired g ≈ 0.93)** and **20/20 in a fresh clean-RNG replication** (`validation_package/01_validation_report.md`).
- **Real-world benchmark added:** a screened **SEC EDGAR panel of 39 public SaaS companies, 1,288 complete quarters (2010–2026)** with pre-declared inclusion criteria; simulated growth sits inside the real growth distribution (E1/E3 PASS), and the failures (spend structure E4, real-company retrodiction C1) are documented with diagnoses, not hidden.
- **The memory question is answered with attribution:** episodic retrieval changes ~60% of monthly spend decisions and yields a significant but **small** outcome gain (+$37.9k of the oracle layer's +$1.19M ≈ **3%**); the brief mechanism (trend/shock reactivity) carries the bulk — reported honestly.
- **New causal layer (oracle v4):** a Neo4j causal graph store records shock → decision → 6-month outcome, feeds role-specific causal chains into CFO/CMO/CPO proposals, and a 4-arm memory ablation (none / episodic / semantic / full) is implemented in the runner.
- **Research became a product without contaminating the research:** a founder-facing FastAPI + React app (advice, seed-matched what-if projection, live review demo) behind a `SIM_PROFILE` switch that reproduces Review-2 behaviour byte-identically by default; **158 pytest tests vs 0 at review-2**.

---

## 2. Before → After

**At review-2 (`3280103`)** the repo was 11 top-level items: `env/`, `agents/`, `boardroom/`, `oracle/`, `config/`, `experiments/` (3 scripts), `results/`, `simulation_runner.py`, `seed_dbs.py`. No tests, no docs, no README, no external data, no UI.

```
Review 2:
  StartupEnv (global RNG, fixed $50k start, 120-month cap)
    → rule-based CFO/CMO/CPO proposals → Boardroom scoring/conflicts
    → Oracle v1/v3 brief (Chroma memory in v3) → WeightAdapter + ActionModifier
    → thesis runner (75 seeds, boardroom vs oracle_v1 vs oracle_v3) → CSVs

Review 3 adds:
  StartupEnv(initial_config: any start state, private deterministic RNG,
             scale-aware curves, real burn, gross margin — all opt-in flags)
    → proposals: rule-based ×scale | LLM rationale | batched causal-LLM (v4)
    → Oracle v1/v3/v4/v4_causal + Neo4j causal graph (shock→outcome learning loop)
    → policies: + noop, v4, v4_causal, v4_causal_no_memory, v4_causal_hetero, v3_hetero
    → evaluation: paired stats (Hedges g, Welch CI, Holm), EDGAR benchmark,
      real-company backtest, claim audit, F1–F11 figure set, review site
    → product: FastAPI backend + React founder app + review demo (SIM_PROFILE-gated)
```

---

## 3. Changes by bucket

### 3.1 Simulation env / business logic (~460 LOC changed)
- **New capability — parameterized environment:** `StartupEnv(initial_config)` accepts any starting state (MRR, cash, churn, headcount, real `monthly_burn`, `max_months`…), replacing the hardcoded $50k/120-month fixture. Enables real-company backtests and founder projections. (introduced `95d1ac5`, extended `0eafa08`→`f1457f4`; [env/startup_env.py](env/startup_env.py))
- **Bug fix — shared-world determinism:** physics drew from the *global* `random` module with **state-dependent draw counts** (`apply_recession_cascade` draws only under high unemployment+rates), so policies at equal seeds desynchronised — measured 7 vs 8 draws/step, hit in 20/20 episodes by ~month 33. Fixed by an opt-in private RNG + always-draw (`deterministic_rng`), off by default so recorded runs reproduce exactly. This underwrites every paired comparison. (`fc035af`; [env/business_logic.py](env/business_logic.py), [env/startup_env.py](env/startup_env.py))
- **Bug fix — founder-scale physics:** (a) burn was `headcount × $8,000` re-implemented in ~8 places — a $500/month founder was charged $8k and died at month 0 of every projection; now `business_logic.monthly_burn()` with `EnvState.monthly_burn` (`5c2aa66`). (b) Marketing Hill-curve constants were dimensionally wrong (gamma a bare dollar draw); reparameterised to the company's own customer base and CAC (`0eafa08`). (c) R&D was a lever that did **literally nothing** (`gain *= 1 − innovation_factor` with factor=1.0 → gain 0); scale-aware form gives product quality its own headroom, half-saturation at the published 24%-of-revenue median (`f1457f4`). (d) CAC runaway feedback (1.4e17 → 9.0e43 in one step) fixed via `stable_cac`.
- **Refactor with a rule:** every fix is behind an opt-in flag (`scale_aware_marketing`, `scale_aware_rnd`, `gross_margin`, `scheduled_shocks`, `deterministic_rng`) — research-profile behaviour is bit-identical by default. Assumed parameters (`SATURATION_ACQUISITION_RATE=0.20`) are labelled ASSUMED in code and surfaced as such downstream.

### 3.2 Agents & boardroom (~1,070 LOC changed)
- **New capability — causal LLM proposals:** `agents/causal_proposal_agents.py` (new, 483 LOC) generates all three C-suite proposals in one causal-context LLM call, with per-role causal chains, stress-persistence escalation, and graceful fallback to scaled rule agents (`c21998f`, `5d1c46c`).
- **New capability — boardroom v4 integration:** injectable `proposal_generator`, proposal cache keyed alongside the oracle brief cache, causal-confidence score boost, stress-node-aware conflict resolution (R&D protected when `Cash_Shortage`), v4 R&D cash cap; decision trace now serialises proposals, causal contexts, stress node and pre/post-modifier actions — the substrate for causal explanation and the HITL advice UI. ([boardroom/boardroom.py](boardroom/boardroom.py))
- **New capability — learning loop:** in v4_causal mode the boardroom writes each shock event and, 6 months later, its realized outcome back into the causal graph.
- **New capability — G11 scaling:** `scale` on all agents and `scale_absolutes` on the boardroom (mrr/50k) so dollar tiers calibrated for a $50k-MRR company aren't strawmen at other scales; 1.0 default preserves research runs (introduced `9f0e218`, profile-gated `17cd983`/`aec4c5e`).
- **Bug fixes (audit-found, `109254a`, `11b866b`):** boardroom burn used `cost_per_employee` (a one-time recruiting cost) as monthly salary — a $500-burn founder read as $10k/month and had whole plans zeroed; hire-cut rounding could never remove an unaffordable hire (floor→ceil on the real-burn path); over-cut marketing budget is refunded; hiring guard now uses **net** runway (revenue-aware) via `_net_runway_months`.
- **Refactor:** `Proposal` gains `rationale` and `causal_confidence`; optional LLM rationale mixin on proposal agents with state-bucketed caching ([agents/proposal_agents.py](agents/proposal_agents.py)).

### 3.3 Memory & Oracle (~1,360 LOC changed, incl. new graph store)
- **New capability — Neo4j causal graph store:** `oracle/graph_store.py` (new, 741 LOC): shock events, outcomes, action→KPI evidence edges (`CONFIRMED_CAUSE`/`MAY_CAUSE` with confidence), seeded priors, role-scoped causal-chain queries (`b8ac80e`, `c21998f`, `164e990`).
- **New capability — modes v4 / v4_causal:** memory-aware mode set extended everywhere (oracle, weight adapter, prompt); graph context (historical recovery stats for the active shock type + similar past shocks) enters the brief prompt; stress-node identification heuristic (`Cash_Shortage`/`Churn_Spike`/`CAC_Pressure`/`Growth_Stall`).
- **Refactor — retrieval quality:** memory docs and queries are tier-prefixed (SEED/EARLY/GROWTH/SCALE × churn/innovation tiers); recency is now **episode-relative** (decay 30 months, was 120 global); outcome-alignment bonus favours DECLINE memories when MRR is falling; months <3 not stored (early noise). ([oracle/memory.py](oracle/memory.py), [oracle/context.py](oracle/context.py))
- **Bug fix — cache-key regression (`fbd9401`):** `b8ac80e` had dropped the retrieved-memory signature from `Oracle.build_cache_key`, so cached briefs ignored differing memories — restored; this protects the integrity of the memory-vs-no-memory ablation.
- **New capability — honest failure (spec G3):** `OracleBrief.parse_ok` flags the neutral fallback so no consumer can present a parse failure as analysis (`9f0e218`); `URGENCY_MAPPING` rescaled to [0,1] (`b8ac80e`; [oracle/schemas.py](oracle/schemas.py)).

### 3.4 LLM layer (~375 LOC changed)
- **New capability — provider factory / heterogeneous routing:** `create_llm_client("ollama"|"openai"|"anthropic"|"dummy")`; per-role clients wired for CFO/CMO/CPO (`d3e7278`, `1217446`). **Caveat: openai/anthropic are placeholders that route to Ollama** — say "routing architecture", not "multi-provider results".
- **Refactor — determinism & robustness:** Ollama calls use JSON mode + `temperature=0` with text-mode fallback (`65e7aa3`).
- **Bug-fix class — parser hardening:** first-JSON-object extraction, enum whitelist normalization, list coercion, confidence clamping, `parse_ok=False` on fallback ([oracle/parser.py](oracle/parser.py)).
- **Refactor — prompt:** opt-in burn/runway clause (fixed "cash-flow positive months of cash" bug and the $8k-burn lie at founder scale) and published churn benchmark line; research prompt byte-identical by default ([oracle/prompt_builder.py](oracle/prompt_builder.py)).

### 3.5 Experiments / evaluation (~5,000 LOC of new analysis code + runner changes)
- **New capability — statistical rigor** (`ecc514d`): `thesis_analysis.py` adds Hedges-corrected Cohen's d with magnitude bands, Welch 95% CIs on mean differences, and **Holm-Bonferroni** family-wise correction on every pairwise table — ablations now report effect *size*, not just p-values.
- **New capability — memory-architecture ablation:** `MEMORY_ABLATION_SCENARIOS` (none / episodic-Chroma / semantic-graph / full) + `include_memory_ablation` in the thesis runner; new `oracle_v4_causal_no_memory` policy is the semantic-only arm the attribution question was missing. **Code is present; no committed CSVs from a 4-arm run yet** — see §7.
- **New capability — runner instrumentation:** per-episode shock/recovery metrics (recovery time, post-shock Rule-of-40 window 25–60), retrieval trace export, `noop` baseline agent, v4/v4-hetero policy registry, `environment_config` passthrough, causal outcome writes per step ([simulation_runner.py](simulation_runner.py)).
- **New capability — EDGAR pipeline** (`109261c`): `data/ingest_edgar.py` (888 LOC) builds `data/edgar.db` from SEC `companyfacts` with Q4 recovered by exact YTD differencing (never imputation); pre-declared inclusion criteria → **39/41 companies included, 1,288 complete quarters** (`data/coverage_report.md`).
- **New capability — validation suite** (`ecc514d`, `e205da7`): 20 analysis scripts under `validation/analysis/` (environment battery E1–E7, action ladders A1, policy baselines A2, oracle value A3, brief sweeps A4/A6, regret A5, robustness grid A7, shock recovery A8, drawdown E6, claim audit, shared-world verifier, real-company backtest C1, scorecards, F1–F11 figure builder, review-site builder).
- **New capability — sourced calibration store** (`0eafa08`, `21f6677`): `calibration/bands.json` + accessor where every value is either printed in a cited source or `None` — no silent defaults; one unverifiable table was **withdrawn** after PDF verification (schema v2 note in the file).

### 3.6 Founder product & review demo *(bucket added — doesn't fit the given list and is too large to omit: ~8,660 LOC across 31 files)*
- **New capability — backend:** FastAPI app (`backend/`, 9 modules): `/api/advise` (boardroom advice on the founder's real company, with assumed-value disclosure and spend ceiling), `/api/whatif` (seed-matched multi-policy rollouts with uncertainty bands and a shared-shock-tape verifier), `/api/review/*` (dataset card, F-figures, live policy compare, real-company backtest), SQLite persistence (`95d1ac5`, `9f0e218`, `109261c`, `8185d61`).
- **New capability — reproducibility switch** (`17cd983`, `848941b`, `aec4c5e`): `backend/sim_profile.py` — `SIM_PROFILE=review2` (default) constructs engine objects exactly as the batch runner does; `founder` enables scale-aware physics, real burn, isolated `chroma_db_founder`, `oracle_v4_causal`, G11 scaling, hiring guard. Single point of truth for every profile-dependent kwarg.
- **New capability — frontend:** React/Vite app (25 files): founder flow (Onboarding → Home → Advice with causal explanation → History → Company → Settings) plus Review-3 demo tabs (Dataset, Run on real companies, policy Compare) (`469d64d`, `67bb71c`, `8185d61`).
- **New capability — memory isolation** (`1a7c20c`, `9a4f3de`): founder memory store is injected, never inherits `CHROMA_PATH` — product usage cannot contaminate the research corpus.

### 3.7 Tests, tooling, docs (~3,250 LOC tests; docs & launchers new)
- **New tests: 158 pytest functions** in 10 files + conftest (0 at review-2): determinism & stats (20), founder guards (30), what-if (31), v4-causal-hetero (27), founder view (18), calibration (15), founder contract (8), integration (7), API (2); plus 5 script-style LLM smoke/eval harnesses (`75fb153`, `fc035af`, `c21998f`).
- **New docs:** README, `docs/` (data provenance, calibration plan, founder spec/roadmap/scale-fix plan, testing runbook), `validation_package/00_START_HERE.md` reviewer bundle, reproduction README (`b625069`, `90d8cce`, `2f862bb`).
- **New tooling:** `start.ps1` one-command launcher (two bug-fix commits: port-conflict diagnosis `de8f1a9`, orphaned-server fix `56e6635`), `run_app.py`, `.claude/launch.json`, `.env.example`, `neo4j_backup.py`, `advice_audit.py` (the audit harness that found the §3.2 defects).

---

## 4. Results & evidence

| Quantity | Value |
|---|---|
| Commits (non-merge / merges) | 43 / 4 |
| Files changed | 282 (+243,541 / −303 incl. generated artifacts) |
| Source code only (env, agents, oracle, backend, frontend, experiments, tests) | 76 files, +18,652 / −301 |
| New modules | `oracle/graph_store.py`, `agents/causal_proposal_agents.py`, `calibration/`, `backend/` (9), `frontend/src` (25), `data/` pipeline (2), `validation/analysis/` (20), 2 launchers |
| Deleted modules | 0 |
| New tests | 17 test files; 158 pytest functions (0 → 158) |

**New results artifacts and what they actually contain** (all committed; sources in `validation/results/`, `validation/agents/`, `validation_package/`):

- `validation_summary.csv` — 44 verdict rows: E1/E3/E7a PASS, E2/E5 PARTIAL, E4/E7b FAIL, E6 exploratory; A1/A2/A3/A6i/A7 PASS, A4ii FAIL, C1 FAIL, C2 NULL, A5/A8 exploratory/comparative.
- `a3_recorded_paired.csv` + `statistical_tests.csv` — recorded 75-seed run re-analysed **paired**: oracle_v3 − boardroom final MRR **+$862k [CI +660k, +1.07M], g=0.93, 75/75 seeds positive**; oracle_v1 +$960k, 74/75.
- `a3_oracle_value.csv` — fresh 20-seed replication under `deterministic_rng`: every oracle arm beats boardroom **20/20**; +$1.15–1.19M final MRR; survival 100% vs 95%.
- `a3_retrieval_decision_delta.csv` — memory ablation (v3 vs v3_no_memory): retrieval changes **62.6% of marketing / 60.2% of R&D months**, outcome gain **+$37.9k [CI +14.0k, +66.3k], g=0.59, p=0.0023** ≈ 3% of the oracle gain.
- `brief_accuracy.csv` — memory-conditioned `expected_outcome` accuracy **64.5% vs 47.6% base rate (+17pp, CI [+11.2, +22.5])**.
- `policy_comparison.csv` — 50 matched seeds: survival noop 100% / random 4% / heuristic 86% / boardroom 96%; boardroom beats all, paired g 0.60–0.82, Holm p ≤ 0.003.
- `a7_robustness_grid.csv` — boardroom ranks 1st on median final MRR in **9/9** initial-condition cells.
- `a8_shock_recovery.csv` — post-shock Rule-of-40 recovery within 24 months: **75–78% oracle vs 62–63% boardroom** (43% heuristic, 29% noop).
- `real_company_backtest.csv` — C1 **FAIL**: hold-arm projects ~+95% median 4-quarter growth vs actual +45% (median |error| 49.6pp); growth-sign agreement 97%; 6 high-burn companies go bankrupt in-sim because the model has **no financing mechanism**.
- `e6_drawdown_recovery.csv` — sharpest structural mismatch: real drawdowns median 11% deep, 85% recovered; simulated 61–63% deep, ~0–2% recovered.
- `claim_audit.csv` — 22 prior claims re-computed: 20 reproduce; **"v4 ≠ v3" UNSUPPORTED** (per-seed correlation 1.000000 in `results/confirmation_runs/…`); curated "screenshot" summary files flagged LEAKAGE-RISK.
- `environment_scorecard.csv` / `environment_stats.csv` — E-battery numbers (e.g. sim median QoQ growth 5.1–5.4% vs EDGAR 5.7%, inside IQR).
- `validation/figures/review/f1–f11` (PNG+SVG) + `validation/review_site/index.html` (self-contained presentation site) + `validation_package/` (7-doc reviewer bundle with its own figures/results copies).
- `outputs/oracle_v4_compare*/` — v4 debug/confirmation traces backing the v4≡v3 finding and the predicate fixes.

---

## 5. Talking points (10-minute order)

1. **"Since Review 2, I stopped asking 'does it work' and started asking 'is it true'"** — the delta's biggest artifact is a validation package with pre-declared criteria and graded verdicts. *So what: every claim on the following slides comes with its CSV.*
2. **Benchmark against reality: 39 public SaaS companies, 1,288 quarters from SEC EDGAR** (F1–F4). *So what: the simulator's growth distribution and deceleration-with-scale match real SaaS; I can now say which parts of the world it gets right — and wrong.*
3. **The comparisons are finally fair: matched random worlds** — I found and fixed a desynchronisation bug in the shared RNG stream (7 vs 8 draws/step) and re-ran everything paired. *So what: policy differences are attributable to the policy, not to different worlds.*
4. **Headline result: the oracle layer wins in essentially every matched world** — 74–75/75 recorded, 20/20 replicated, g≈0.93 (F6). *So what: the architecture's value is not a mean artifact; it's per-seed dominance.*
5. **Memory ablation with honest attribution** — retrieval changes ~60% of decisions but contributes only ≈3% of the gain; the brief's trend/shock reactivity carries the rest (F10). *So what: I know* where *the value comes from, and I don't oversell memory.*
6. **New causal layer (v4): the system now records what it did during shocks and what happened 6 months later** in a Neo4j graph, and grounds proposals in those chains. *So what: from correlation-shaped memory toward causal explanation — with the 4-arm ablation built to test it.*
7. **The claim audit caught my own errors** — "v4 beats v3" is retracted (identical trajectories absent shock context), and curated summary files are flagged. *So what: the negative results are in the deck because they're in the data.*
8. **From research to product without touching the research:** founder app (advice + seed-matched what-if) behind `SIM_PROFILE`; review-2 default is byte-identical to the batch runner. *So what: HITL demo runs live, and reproducibility survives.*
9. **Engineering maturity: 0 → 158 tests, docs, one-command launch, review site.** *So what: a reviewer can reproduce any number in this presentation from the repo.*
10. Close on limits (next section) before they're asked.

---

## 6. Likely reviewer questions

*(Note: I could not find Review-1 feedback recorded in the repo; the benchmark-comparison tie-in below is based on your prompt, not a committed document.)*

- **"You were asked about benchmark comparison — where is it?"** Two kinds now exist: (1) *external*: the EDGAR panel with 7 environment tests (E1/E3/E7a PASS; E2/E5 PARTIAL; E4/E7b FAIL — each with a diagnosis); (2) *internal*: paired baselines noop/random/heuristic/boardroom at 50 matched seeds (A2, g 0.60–0.82). Both use pre-declared criteria and the episode as the unit.
- **"Is the simulator realistic?"** As a *comparative testbed* at research scale, yes with caveats (growth distribution and deceleration match; persistence/volatility/spend-share don't). As a *forecaster*, no — C1 over-projects by ~50pp, and I can say why: the one assumed marketing parameter is falsified as too high, and there is no financing mechanism. That distinction is the report's headline verdict.
- **"Does memory actually help?"** Retrieval-conditioned briefs are genuinely predictive (+17pp over base rate) and the outcome increment is significant (p=0.0023, 15/20 seeds) but small (≈3% of the gain). The honest claim: the brief mechanism, not episodic retrieval, carries most of the value; the 4-arm ablation isolating episodic vs semantic memory is implemented and is the next run.
- **"What does the causal graph add over v3?"** In recorded data, nothing yet — v4 ≡ v3 without shock-time graph context, and I flagged my own earlier claim as unsupported. The graph's contribution is testable now (semantic-only arm exists); note the semantic arm partly encodes hand-seeded priors, which any claim must disclose.
- **"Fragmented vs centralized baseline comparison?"** The committed comparisons are noop/random/heuristic (fragmented C-suite) vs boardroom (centralized negotiation) vs oracle variants; boardroom > heuristic paired g=0.78 is the fragmented-vs-centralized datapoint (`policy_comparison.csv`).
- **"Why does the LLM matter if rules do the work?"** A4 splits it: the rule layer reproduces 10/10 documented thresholds; the LLM brief is a trend-and-shock reactor (moves the right way on MRR deltas, ρ=+0.96, and shock alerts) but is level-blind — it failed the pre-declared one-variable level sweeps 0/4. That's a limitation slide, and it's why the post-shock window is where the oracle's value concentrates (A8).
- **"Can a founder trust the product?"** The product path is exactly the research engine plus disclosed deltas: real burn, scale-aware curves, G11 scaling — each fix is flag-gated, and every assumed value is labelled (assumed-value disclosure in `/api/advise`, `parse_ok` honest-failure flag, calibration store with citations or `None`).

---

## 7. Known gaps visible in the diff (preempt these)

- **4-arm memory ablation not yet run:** `MEMORY_ABLATION_SCENARIOS` and the semantic-only policy exist in code; no `memory_ablation_*.csv` is committed. F10 attribution rests on the 2-arm v3 vs v3_no_memory data.
- **v4/v4-causal has no positive result:** v4 ≡ v3 in the confirmation run; `oracle_v4_causal_hetero` (the only LLM-composed-action policy) *lost* its only recorded comparison (0.75 vs 1.00 survival) and is excluded from headline claims (report §6.5).
- **C1 backtest FAIL stands:** ~+50pp over-projection, no financing mechanism, `SATURATION_ACQUISITION_RATE=0.20` falsified at real spend intensities. Framed as a diagnosis, but it is an open defect.
- **LLM level-blindness (A4ii FAIL)** and single-model dependence (llama3.1:8b); openai/anthropic providers are placeholders routing to Ollama.
- **E4 spend structure FAIL / E2+E5 PARTIAL / E6 drawdown mismatch** — simulated shocks cause permanent revenue collapses real SaaS doesn't show; never conflate Rule-of-40 recovery with revenue recovery.
- **Leakage-risk files are still in the repo** (`primary_summary_screenshot_no_oracle_v3.*` under `results/confirmation_runs/`) — flagged by the claim audit; cite only canonical `primary_summary.csv`.
- **Reward function misaligned with headline metrics** (oracle has *worse* total reward while winning every headline metric) — excluded as an outcome, but a reviewer may probe it.
- **Uncommitted work in the tree right now:** polish on the review demo (`backend/review_service.py`, `frontend/src/copy.js`, `Dataset.jsx`, `Review.jsx`, `Run.jsx`, +65/−31) — commit before the demo.
- **Fixed shock timetable (months 24/48/72) is in principle learnable by memory** — disclosed as a bounded threat; memory arms learn across episodes within a run (episodes are sequential, not i.i.d.).
- Minor: the validation report header names branch `review2-sim-frontend` while the work is committed on `review-3` — cosmetic, but fix before handing the PDF over.

---

## 8. Suggested next milestones

1. **Run the 4-arm memory ablation** (`include_memory_ablation=True`, deterministic RNG, ≥50 seeds) — it is one flag away and directly answers the architecture-attribution question with the new effect-size/Holm machinery.
2. **Give v4's graph a fair test:** extend graph context beyond shock months (or route it through the batched causal proposer in a controlled run) so v4 vs v3 is a real comparison, then re-audit the claim.
3. **Close the two falsified physics items:** fit `SATURATION_ACQUISITION_RATE` against the EDGAR panel instead of assuming 0.20, and add a minimal financing mechanism — both are named as the C1 blockers in the report.
4. **Address brief level-blindness:** include absolute-state anchors (runway, churn-vs-benchmark) in the research prompt path (the founder path already has them) and re-run A4 level sweeps.
5. **Commit or delete the flagged screenshot-variant summaries** and the uncommitted demo polish; regenerate the report header.
6. **One thesis-grade write-up pass:** the report's §8 "required paper language" is already drafted — lift it into the thesis with the F1–F11 figures.
