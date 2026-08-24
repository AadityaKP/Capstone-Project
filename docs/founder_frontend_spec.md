# Founder Frontend Specification — Startup Society of Minds

**Document type:** Founder-facing frontend product specification
**Prepared from:** Direct read-only inspection of the repository (all branches), 2026-08-24
**Working branch at time of review:** `frontend` (HEAD `67bb71c` "Frontend version1")
**Status of this document:** Product specification only. No product code was written or modified.

---

## How to read this document

Every major feature in this spec carries a capability classification:

| Tag | Meaning |
|---|---|
| **[C1]** | Already supported by the current system as-built |
| **[C2]** | Supported with relatively small product/backend changes (the exact change is stated) |
| **[C3]** | A frontend/product abstraction over an existing capability |
| **[C4]** | A genuinely new capability that must be built |
| **[C5]** | Not supported — must never be presented as though it exists |

A crucial fact discovered during repository inspection shapes every classification in this document: **the repository contains three materially different frontend/backend states on three branches.**

1. **`frontend` branch (checked out):** the Python research harness plus a **static, non-functional React mockup**. Every number in the UI is a hardcoded constant; there are zero `fetch` calls, no forms that submit, and the "Run new simulation" button has no handler ([App.jsx](frontend/src/App.jsx)).
2. **`backend` branch (unmerged):** a **working integrated application** — FastAPI + SQLite + a fully API-wired version of the same React app, including a founder-suppliable `ScenarioConfig`, custom initial environment state, background run execution with status polling, and persisted runs/traces. The compiled bytecode of this backend also sits untracked in the working tree (`backend/__pycache__/*.pyc`, June 2026), with source deleted locally.
3. **`startup-multi` branch (most recent commit, 2026-08-24):** further Oracle evolution (`BatchedCausalProposalGenerator`, causal stress routing) — partially reviewed; treated as UNVERIFIED beyond what is cited.

Unless stated otherwise, **[C1] refers to code on the checked-out `frontend` branch**; where a capability exists only on the `backend` branch, it is tagged **[C1-backend-branch]** and the practical product cost is "merge/port the branch," which is closer to [C2] than to building from scratch.

---

## 1. Executive Product Interpretation

### 1.1 What this system is, technically

A calibrated synthetic SaaS-company simulator (Gymnasium environment, monthly steps, up to 120 months, terminates on bankruptcy) governed by explicit "physics" — Hill-function marketing response, segment churn with tenure decay, macro shocks, hysteresis-scarred innovation ([env/business_logic.py](env/business_logic.py)). On top of it sits an **AI boardroom**: three deterministic heuristic C-suite agents (CFO/CMO/CPO) propose partial actions each month; a Boardroom module scores the situation, weights strategic priorities, merges the proposals, lets an optional LLM **Oracle** (local Ollama `llama3.1:8b`) reweight and multiplicatively modify the plan, then applies bounds, floors, and cash-conflict resolution to emit one final action bundle — marketing spend + channel, hires, R&D spend, price change ([boardroom/boardroom.py](boardroom/boardroom.py)). The Oracle's structured brief (risk level, growth outlook, efficiency pressure, innovation urgency, macro condition, key risks/opportunities/focus, confidence, and in v3+ a 6–12-month outcome expectation) is produced from the current state, trend context over a 5-snapshot history, up to 3 retrieved outcome-labelled memories of similar simulated situations (ChromaDB), and — in `oracle_v4_causal` — aggregate recovery statistics for the active shock type from a Neo4j causal graph ([oracle/](oracle/oracle.py)). Every month yields a complete `decision_trace` explaining what was decided and why.

Today the system only ever advises the simulator's synthetic company. On the `backend` branch, however, the entry point already accepts a founder-shaped initial state (`ScenarioConfig` → `StartupEnv(initial_config)`), which is the seed of the founder product.

### 1.2 What this system is, to a founder

**A monthly AI advisory board for an early-stage SaaS company.** The founder describes their company in about ten numbers they already know (revenue, cash, burn, price, churn, market crowdedness). The advisory board analyses the position, names the biggest risk in plain language, recommends this month's resource allocation across marketing, product/R&D, hiring, and pricing — with reasons — and can stress-test a decision by running it forward many times in a simulated market and showing the range of outcomes.

### 1.3 The honest mental model

The candidate framing given in the project brief — *"AI advisory board that analyses your current position, explains the biggest risk, recommends this month's resource allocation (marketing/R&D/hiring/pricing), and lets you stress-test decisions in simulation"* — **is the honest core**, verified element by element:

| Claim | Grounding | Verdict |
|---|---|---|
| "Analyses your current position" | `Oracle.generate_brief(state)` is a pure function of an `EnvState` + context; `Boardroom.decide(state)` needs no environment | Honest **[C2]** (needs a founder-state entry point; the state constructor exists on the `backend` branch) |
| "Explains the biggest risk" | `brief.risk_level`, `brief.key_risks[]`, plus the decision trace's weight/modifier deltas | Honest **[C1]**, with a caveat: `key_risks` strings are LLM-generated free text and need a display guardrail (§15) |
| "Recommends this month's resource allocation" | `final_action` covers exactly marketing/hiring/R&D/pricing | Honest **[C1]**, with a scale-calibration caveat (§5.6) |
| "Stress-test decisions in simulation" | `run_simulation(..., environment_config=...)` runs seeded episodes from a custom state (backend branch); an action-override wrapper is a small addition | Honest **[C2]** — and every surface must present results as *simulated scenario ranges, never forecasts* |

Mental models that would be **dishonest** and are rejected: "predicts your company's future" (it simulates a synthetic world, with randomized elasticities and shocks); "learns from thousands of real startups" (every memory is a simulated trajectory); "monitors your business" (nothing is integrated; data arrives only when the founder types it); "an AI co-founder" (it has no execution ability and no context outside the ten numbers).

The chosen positioning, used consistently in all product copy in this spec: **"Your AI advisory board — analysis, a monthly plan, and stress-tests in simulation."** Decision-support, not prophecy.

### 1.4 The three facts that shape the whole frontend

1. **Latency is tens of seconds per LLM call.** Measured: 16–24s per Ollama call ([tests/ollama_test_results.csv](tests/ollama_test_results.csv): 20.2s, 21.1s, 24.3s; provider smoke test 15.9s). One analysis = 1 brief call (~20s), plus optionally 3 per-agent narrative calls (~60–75s more). A full 120-month simulated episode with Oracle ≈ 28–30 LLM calls (measured in the n=75 thesis run) ≈ 8–12 minutes. Pure-physics simulation without the Oracle runs in milliseconds per month. Consequence: analysis is an **async job with honest staged progress**, scenario runs default to **no-LLM policies** for interactivity, and results are always persisted so they never have to be recomputed to be re-read.
2. **The reasoning engine is calibrated to a $50k-MRR-scale synthetic company.** Spend recommendations come from fixed tiers ($2k/$10k/$20k marketing; $3k/$8k/$15k R&D) with floors ($20k+ R&D under innovation deficit; max($5k, 2% of MRR) marketing) and salary assumed at $8k/employee/month. The product must translate magnitudes honestly (§5.6) and state its calibration envelope.
3. **Failure is currently silent.** A malformed or empty LLM response becomes a neutral MEDIUM/STABLE brief with confidence 0.5 ([oracle/parser.py:122](oracle/parser.py)). In a founder product, that would present degraded output as considered advice. The founder frontend requires the failure state to be surfaced (§15, §17) — a small backend change **[C2]**.

---

## 2. Current System Capability Map

### 2.1 Verified as-built flow (the canonical spine)

Every frontend element in this spec attaches to a node of this flow or is declared a gap in §21.

```
FOUNDER INPUT (backend branch: ScenarioConfig)                                [entry]
      │  initial_mrr, initial_cash, initial_headcount, average_price, cac, ltv,
      │  churn_{enterprise,smb,b2c}, competitors, product_quality, innovation_factor,
      │  interest_rate, consumer_confidence, unemployment, valuation_multiple, max_months
      ▼
EnvState  (env/schemas.py — 18 fields; months_elapsed NOT configurable today)
      ▼
┌──────────────────────────── MONTHLY DECISION (Boardroom.decide) ────────────────────────────┐
│ CFO/CMO/CPO heuristic proposals ──► situation ScoreVector (efficiency/growth/innovation/   │
│ (objective, actions, impact,        macro — computed from STATE ONLY; identical for all    │
│  risks[], confidence)               three proposals)                                        │
│        │                                                                                    │
│        ▼                                                                                    │
│ Oracle refresh decision: initial | cadence (every N months) | event                         │
│   (runway<12mo, active shock, MRR ≤85% of last refresh, churn +≥1.5pp, confidence −≥15,     │
│    unemployment +≥2) — otherwise REUSE last brief                                           │
│        │ refresh → LRU cache (key: mode, MRR/50k, runway/3, competitors, confidence/10,     │
│        │            MRR trend, shock flags; max 5000) → hit? reuse : LLM call (~20s)        │
│        ▼                                                                                    │
│ OracleBrief (enums + key_risks/opportunities/focus + confidence; v3+: expected_outcome)     │
│   context: TrendContext (needs ≥2 snapshots) · top-3 memories (Chroma, outcome-labelled,    │
│   run_id-scoped) · GraphContext (v4_causal + active shock only: recovery stats from Neo4j)  │
│        ▼                                                                                    │
│ WeightAdapter: base 0.30/0.20/0.40/0.10 → urgency-nudged (k=0.2, 0.3 memory-aware modes,    │
│   × brief confidence; +0.05 innov/eff if DECLINE predicted; 70/30 smoothing)                │
│        ▼                                                                                    │
│ Merge: CMO→marketing, CFO→hiring+pricing, CPO→R&D (aggressively scaled by innovation        │
│   deficit, up to 15% of MRR when innovation < 0.3)                                          │
│        ▼                                                                                    │
│ ActionModifier (multiplicative, from brief): e.g. risk CRITICAL → marketing ×0.3, R&D ×1.5, │
│   hiring cap 0; growth COLLAPSING → marketing ×0.3; efficiency CRITICAL → all spend ×0.65   │
│        ▼                                                                                    │
│ Sanity bounds (marketing ≤ max(30% cash, $20k); hires ≤ 10)                                 │
│ Dynamic minimums (R&D ≥ max(MRR × deficit × 0.10, $20k + deficit × $50k);                   │
│                   marketing ≥ max($5k, 2% MRR))                                             │
│ Cash-shortfall conflict resolution (cut marketing → hiring → R&D; R&D 80%-protected when    │
│   innovation score > 0.6)                                                                   │
│        ▼                                                                                    │
│ final_action  +  decision_trace (month, refresh_reason, brief_source, weights before/after, │
│   brief, memories, pre/post-modifier actions, per-domain % changes)                         │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
      ▼
env.step(action): stochastic shocks → hard shocks (months 24/48/72, cycled by seed) →
  cascade/hysteresis/recovery → marketing Hill response → churn → expansion MRR → pricing
  elasticity (random −0.9..−0.2) → burn (salaries $8k/head + spend + hires) → CAC/LTV update
  → reward → info {rule_of_40, full state dump, shock_label}
      ▼
PERSISTENCE (backend branch): SQLite — scenarios · simulation_runs (status/error/summary) ·
  episode_results · monthly_traces · action_traces;  ChromaDB oracle_live_memories (run_id-
  scoped, 6-month outcome maturation);  Neo4j shock→decision→outcome graph (optional)
      ▼
OPTIONAL NARRATIVE LAYER [exists, off by default]: per-agent LLM refinement rewrites each
  proposal's expected_impact into a 2-sentence rationale grounded in state numbers
  (agents/proposal_agents.py use_llm; exercised by tests and the oracle_v3_hetero policy)
```

### 2.2 Corrections and additions to the briefing (§1 of the project prompt), from source

The repository wins over the prompt in these places:

1. **Oracle v4 exists and the causal graph is live, not dead.** Modes `oracle_v4` and `oracle_v4_causal` are implemented and registered ([oracle/oracle.py](oracle/oracle.py), [simulation_runner.py](simulation_runner.py), [oracle_v4_breakdown.md](oracle_v4_breakdown.md)). `oracle_v4_causal` wires a real `CausalGraphStore` (Neo4j) that records Shock→Decision→Outcome chains during runs and injects aggregate recovery statistics ("mean recovery 4.6 months across 12 prior episodes of RATE_HIKE") into the prompt. It degrades gracefully to `oracle_v4` when Neo4j is absent.
2. **The separate `OracleAgent` (§1.5 of the prompt) is dead code.** `agents/oracle_agent.py` is imported nowhere. `seed_dbs.py` seeds that prototype's stores (Chroma collection `episodes`, Neo4j `Entity` triples), not the live system's (`oracle_live_memories`, Shock/Decision/Outcome graph). The free-text `analyze_situation(query)` loop therefore does **not** exist as a usable capability → any conversational advisor is [C4] (§12).
3. **Memory retrieval parameters differ from the briefing:** 6 candidates queried (not 10), 30-month recency decay (not 120), plus an **outcome-alignment re-ranking bonus** (declining MRR trend boosts DECLINE-outcome memories by +0.08, etc.) and suppression of trivial early months (`source_month < 3`) ([oracle/memory.py](oracle/memory.py)).
4. **The brief cache key contains no memory signature** (deliberately removed in v4: "cache key without memory-signature poisoning"). Key = (mode, MRR/$50k bracket, runway/3 bracket, competitor count, confidence/10 bracket, MRR trend, shock flags).
5. **R&D floor has an absolute component:** `max(MRR × deficit × 0.10, $20,000 + deficit × $50,000)` — so whenever the modifier path runs, recommended R&D never drops below $20k/month regardless of company size. Material for small-startup calibration (§5.6).
6. **Two different burn constants coexist:** runway estimates and env salary burn use $8,000/employee/month; proposal efficiency scoring uses $10,000; conflict resolution uses `cost_per_employee` (default $10,000) for base burn. The founder product must not inherit this ambiguity — §5.4 resolves it.
7. **Proposal "confidence" is not differentiated.** All three proposals receive the identical state-derived ScoreVector each month; per-agent static confidences (0.8/0.75/0.78) are constants. The UI must not present agent-vs-agent confidence as meaningful signal.
8. **`oracle_v3_hetero` (the LLM-refined-proposals policy) is broken on this branch** — it returns a raw `Boardroom`, which lacks the `get_action` the runner calls. The `backend` branch fixes it (wraps in `BoardroomAgent`, shares one client). The narrative-refinement capability itself is real and tested.
9. **Reset/initial values:** cash $1,000,000, product_quality 0.1, CAC $50 (`BASE_CAC`), price $50, LTV $7,000 — note the internal inconsistency: with price $50 and ~3% blended churn, the engine's own LTV formula (price/churn) would give ≈$1,667, not $7,000, and initial LTV:CAC = 140. First real update recomputes both. Founder inputs should therefore **derive** LTV rather than ask for it (§5).
10. **Hard shock selection is `seed % 3`** (0→competitor surge, 1→rate hike, 2→recession) at months 24/48/72 **of simulation time** — which interacts with founder company age if `months_elapsed` were ever mapped (§11.5).
11. **Hiring is affordability-capped** at `cash / 18 / cost_per_employee` inside the env, and modifier hiring caps are 0/0/1/2 by risk level — recommended hires are effectively ∈ {0, 1, 2}.
12. **The runner measures recovery**: per-shock pre-shock Rule-of-40 benchmark, recovery month, recovery time, plus post-shock window (months 25–60) aggregates — the empirical basis for any "recovery" language in the UI.
13. **Thesis-scale evidence exists** (n=75 seeds × 3 policies, [results/.../thesis_summary_report.md](results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background/thesis_summary_report.md)): boardroom-alone 97.3% survival / $1.39M mean final MRR; oracle_v1 98.7% / $2.35M; oracle_v3 98.7% / $2.25M; oracle improvements in final MRR and post-shock Rule-40 statistically significant (Mann-Whitney p ≤ 0.02). Usable in marketing copy **only** as "in our simulated benchmark" claims.
14. **`oracle_frequency` defaults disagree across layers** (Boardroom 3, backend API 5, thesis experiments 10). The founder product sets its own cadence semantics (§13) and should not expose the knob.
15. **The frontend↔backend contract exists only on the `backend` branch** (§2.4). The checked-out frontend triggers nothing.
16. **OracleBrief contains no narrative/rationale field.** The backend-branch UI references `brief.rationale || brief.summary` — neither exists; it silently falls back to placeholder copy. A founder-readable narrative is an add-on (existing per-agent refinement [C2], or a brief-level narrative field [C2]).
17. **`months_elapsed` is not settable** via `ScenarioConfig` — company age (which drives tenure churn decay, a up-to-3.3× churn effect) cannot currently be represented [C2 fix, §5.3].

### 2.3 Capability inventory

| Capability | Status | Where |
|---|---|---|
| Monthly decision from an arbitrary `EnvState` (no env needed) | **[C1]** pure functions | `Boardroom.decide(state)`, `Oracle.generate_brief(state)` |
| Founder-shaped initial state → simulation | **[C1-backend-branch]** | `ScenarioConfig`, `StartupEnv(initial_config)`, `run_simulation(environment_config=)` |
| Structured strategic brief with risk/outlook/urgency/focus/confidence | **[C1]** | `oracle/*` |
| Trend awareness (needs ≥2 observed snapshots) | **[C1]** | `oracle/context.py` |
| Outcome-labelled "similar situations" memory (simulated) | **[C1]** | `oracle/memory.py`, ChromaDB |
| Shock-type recovery statistics (causal graph) | **[C1]** optional infra | `oracle/graph_store.py`, Neo4j |
| Full decision trace (why/what changed) | **[C1]** | `boardroom.last_decision_trace` |
| Per-agent plain-language rationale (2 sentences, state-grounded) | **[C1]** exists, off by default; policy wiring fixed on backend branch | `agents/proposal_agents.py use_llm` |
| Seeded multi-episode simulation with monthly/action/retrieval traces | **[C1]** | `simulation_runner.py` |
| HTTP API, background runs, status polling, SQLite persistence of runs/traces/scenarios | **[C1-backend-branch]** | `backend/*` |
| Founder-tier vocabulary (SEED/EARLY/GROWTH/SCALE, churn & innovation tiers) | **[C1]** | `oracle/context.py` |
| Accounts/auth, multi-user | **[C5→C4]** nothing anywhere | — |
| Per-company (not per-run) history & memory | **[C4]** (memory is run_id-scoped in-process; nothing persists a company's month-over-month reality) | — |
| Free-text Q&A advisor | **[C5→C4]** (only prototype is dead code) | — |
| Real-world data ingestion/integrations/benchmarks | **[C5]** | — |

### 2.4 The existing frontend↔backend contract (backend branch)

Recorded verbatim from [backend/main.py] and the wired [App.jsx] on that branch — this is the reuse substrate:

- `GET /api/health` → `{status, database:"sqlite", simulation_engine:"ready"}`
- `GET /api/config` → `{policies[], default_policy:"boardroom", oracle_policies_require_llm[]}`
- `GET/POST /api/scenarios` → persisted `ScenarioConfig` under a name (no update/delete)
- `POST /api/runs` (202) → validates policy, inserts `simulation_runs` row (status `queued`), spawns a daemon thread, returns run; run executes `run_simulation(...)` with `return_action_trace + return_monthly_trace`, then bulk-inserts episode/monthly/action rows and a computed `summary_json` (avg final MRR/cash/LTV:CAC/Rule-40, survival %, shock_events[], first 10 memories, latest_brief); on exception sets status `failed` + `error` text
- `GET /api/runs?limit=` → recent runs with parsed summaries
- `GET /api/runs/{id}?include_trace=true` → run + `episodes_results[]` + full `monthly_trace[]`
- Frontend: single `api()` fetch helper, 1.5s `setInterval` polling while status ∈ {queued, running}, scenario form bound to state, empty/progress components, an SVG polyline chart computed from the real monthly trace with shock markers.

No auth, no pagination beyond `limit`, no run cancellation, no WebSockets/SSE — polling only. Single-process SQLite with WAL. These are acceptable foundations for a single-founder MVP and are inherited as constraints in §22.

### 2.5 Performance & cost realities (verified)

| Operation | Cost | Source |
|---|---|---|
| One Oracle brief (1 LLM call) | ~16–24 s | ollama/llm test CSVs |
| Per-agent narrative refinement (3 calls) | ~60–75 s total | ollama_test_results.csv (20.2/21.1/24.3 s) |
| Boardroom decision without LLM | milliseconds | pure Python |
| 120-month episode, physics only (heuristic/boardroom policy) | well under a second | pure Python |
| 120-month episode with Oracle (freq 10) | 28–30 LLM calls ≈ 8–12 min | thesis run episode stats |
| Cache effectiveness | ~1–2.7 cache hits/episode at freq 10 | thesis run stats |
| Brief reuse between refreshes | free (stored on Boardroom) | code |

Design rule derived: **founder-facing scenario simulation defaults to the `boardroom` (no-LLM) policy** — interactive speeds, deterministic per seed — and LLM commentary is layered on afterwards as a single optional call.

---

## 3. Founder Jobs-to-be-Done

Grounded in what the verified engine can answer. ✅ = supported now (per tags), ⚠️ = partial, ❌ = unsupported (must not be implied).

| # | Founder question | Verdict | Grounding |
|---|---|---|---|
| J1 | "What should I focus on right now?" | ✅ [C1/C2] | `recommended_focus[]`, applied weight mix, final_action priorities |
| J2 | "What's my biggest risk / what's going wrong?" | ✅ [C1] | `risk_level`, `key_risks[]`, event-trigger reasons (churn jump, MRR drop, runway) |
| J3 | "How should I allocate money this month (marketing vs product vs hiring vs pricing)?" | ✅ [C1] with §5.6 scale translation | `final_action` |
| J4 | "Why this recommendation?" | ✅ [C1→C3] | decision_trace (weights, modifier deltas, refresh reason, brief enums) + [C2] narrative refinement |
| J5 | "What happens if I make this decision?" (price change, spend change, hiring) | ⚠️ [C2] | seeded simulation from founder state + an action-override wrapper (§11); always labelled simulation |
| J6 | "How long is my runway?" | ✅ [C3] | arithmetic on founder inputs (display true burn-based runway; see §5.4) |
| J7 | "Has anything changed since last time? Am I improving?" | ⚠️ [C2/C4] | trend context exists but per-company month persistence must be built (§13, §21) |
| J8 | "What did companies like mine do in this situation?" | ⚠️ [C1] **simulated only** | memories + causal graph stats; must be labelled "in simulated scenarios" everywhere (§15) |
| J9 | "Should I raise prices?" | ⚠️ [C1] narrow | CFO rule: +5% only when LTV:CAC < 3; effectively binary advice — present honestly as "hold" vs "consider ~5% increase" |
| J10 | "Am I ready to hire?" | ⚠️ [C1] coarse | runway>24mo & LTV:CAC≥3 → 1 hire; risk caps 0/0/1/2 — present as readiness signal, not headcount plan |
| J11 | "How do I compare to other startups like mine?" | ❌ [C5] | no real-world benchmark data exists |
| J12 | "Which channel/campaign/feature should I build?" | ❌ [C5] | engine knows only ppc-vs-brand spend aggregates and an R&D dollar figure |
| J13 | "When should I fundraise / at what valuation?" | ❌ [C5] | valuation_multiple is synthetic physics, not market advice; also prohibited-adjacent (financial advice) |
| J14 | "Watch my metrics and alert me" | ❌ [C5→C4 Later] | no data feeds; possible only after integrations exist |

The product commits to J1–J10 and visibly refuses J11–J14 (§15.4 "What this advisor cannot see").

---

## 4. Founder Product Loop

The loop is derived from the real reasoning flow — including the two mechanisms that make the system *smarter over time within its own terms*: trend context activates at the **second** observed month, and memories mature into outcome-labelled evidence after a **6-month** horizon.

```
DESCRIBE (once, ~5 min)          → minimum inputs → company state established
        ▼
ANALYSE (async, ~20–90 s)        → brief + boardroom decision + decision trace
        ▼                          [1 LLM call; +3 optional narrative calls]
READ THE ADVICE (2 min)          → position → biggest risk → this month's plan → why → evidence
        ▼
(OPTIONAL) STRESS-TEST (~secs)   → "what if I raise price 10%?" → simulated range vs baseline
        ▼
ACT (in the real world)          → checklist derived from the plan; product is honest that it
        ▼                          cannot see execution
UPDATE (next month, ~2 min)      → 6 numbers re-entered → snapshot persisted [C4 store]
        ▼
WHAT CHANGED                     → metric deltas + trend directions (engine-computed) +
        ▼                          re-analysis (event triggers use real thresholds: MRR ≤85%,
        │                          churn +1.5pp, runway <12 → copy: "re-analysed because your
        │                          churn jumped")
        └────────► month after month: history accumulates → after 6 updates, the founder's own
                   months become outcome-labelled memories [C2: run_id → company_id scoping]
                   → advice quotes the founder's own past ("last time churn rose like this…")
```

Anti-pattern explicitly rejected: a passive dashboard the founder "checks." The product has exactly one heartbeat — **the monthly update ritual** — because the engine's intelligence (trends, event triggers, memory maturation) is keyed to fresh state, and stale state produces stale-labelled advice (§15.3).

---

## 5. Input Architecture

Derived from `EnvState` ([env/schemas.py](env/schemas.py)), the backend branch's `ScenarioConfig`, and what the reasoning code actually consumes. Two consumers matter: the **Oracle prompt** uses MRR, cash, CAC, LTV, avg churn, innovation, unemployment, confidence, competitors, interest rate, months-in-depression, plus trends and shock label; the **boardroom scoring/rules** use cash, headcount, CAC/LTV, churn segments, innovation, unemployment, depression months. Fields none of them read meaningfully for advice (e.g. `valuation_multiple` outside reward shaping) are deprioritised.

### 5.1 Master input table

Formats: `$` = currency, `%` = percentage, `#` = count. Effort/Complexity: L/M/H. "Where founder gets it" assumes a typical early-stage SaaS founder with Stripe/billing + a bank account + a spreadsheet.

| Input | Why needed (engine consumer) | Req? | Format | Effort | Cognitive | Where founder gets it | Validation (from Pydantic + physics) | Capability using it |
|---|---|---|---|---|---|---|---|---|
| Company name | labelling only | Req | text | L | L | — | 1–100 chars (ScenarioCreate) | workspace identity |
| **MRR** | anchor of everything: prompt, tiers, floors, attractiveness | Req | $ /mo | L | L | billing/Stripe | > 0 | brief, plan sizing, tiers |
| **Cash in bank** | runway, efficiency score, conflict resolution, bankruptcy | Req | $ | L | L | bank account | > 0 | brief risk, refresh triggers, plan bounds |
| **Monthly burn (total costs)** | runway & risk calibration — engine models burn as headcount×$8k; founder-supplied burn must override (§5.4) | Req | $ /mo | L | L | P&L / spreadsheet | ≥ $5k (MIN_BURN_RATE floor as sanity hint) | runway, CFO/hiring rules, event triggers |
| **Price / ARPU** | pricing physics, LTV derivation, new-user estimation | Req | $ /user/mo | L | M ("average per paying customer") | billing | > 0 | pricing advice, LTV, CAC |
| **Monthly churn (blended)** | churn physics, CPO tiers, innovation urgency, LTV | Req | % /mo | M | M — monthly vs annual confusion is the #1 expected error | billing/retention report | 0–100%, warn > 30% (MAX_CHURN), hard error > 100%; explicit "monthly, not annual" helper | churn tiers, brief, R&D advice |
| **Company age** | tenure churn decay (up to 3.3× effect); "months_elapsed" | Req | # months | L | L | known | 0–600; **[C2]** add to ScenarioConfig + env mapping | effective churn, scenario timing |
| **Market crowdedness** | competitor count drives acquisition damping (≥4/≥10), CAC scaling (>5/≥8), cache shock flag (>8) | Req | choice | L | L | founder judgment | 3 options → 2 / 5 / 9 competitors **[C3]** | brief, CAC realism |
| New customers last month | CAC derivation (CAC = marketing spend ÷ new customers) | Opt (either this+spend or CAC directly) | # | L | L | CRM/billing | ≥ 0 | CAC → LTV:CAC → marketing tiers |
| Marketing spend last month | CAC derivation + burn sanity | Opt | $ | L | L | P&L | ≥ 0 | CAC, onboarding sanity check |
| CAC (direct) | if founder already knows it | Opt | $ | L | M | own analytics | > 0 | growth score, CMO tiers |
| Per-segment churn (enterprise/SMB/B2C) | segment shock realism (SMB ×1.5 in competitor surge etc.) | Opt (enriched) | %×3 | M | H | cohort analysis | each 0–100% | shock realism in scenarios |
| Headcount (people) | display + payroll cross-check (engine unit ≠ people; §5.4) | Opt | # | L | L | known | ≥ 1 | display, hiring translation |
| Product maturity self-rating | proxy for `product_quality` (churn dampener 1−0.5q) | Opt | 3-choice | L | M | judgment | maps 0.2/0.5/0.8 **[C3]**, labelled *estimated* | churn physics |
| Valuation multiple | reward shaping only | Opt (advanced) | ×ARR | L | H | last round | > 0, default 10 | scenario reward only |
| Macro conditions (interest, confidence, unemployment) | brief's macro_condition, CAC scaling, demand damping | Auto-default | — | — | — | — | defaults 3% / 100 / 4% = "typical conditions", labelled system-set; optional "market mood" 3-way override **[C3]** | macro realism |
| `innovation_factor` | internal R&D-efficiency scar | Hidden | — | — | — | — | default 1.0 (unscarred) — a real company has no measurable analogue | R&D advice internals |
| `months_in_depression` | internal hysteresis counter | Hidden | — | — | — | — | 0 | internals |

**Explicitly never asked:** LTV (derived: price ÷ blended churn — matches `compute_ltv`; asking founders for LTV invites inconsistent methodology), seeds, policies, episode counts, oracle mode/frequency, Rule-of-40 inputs.

### 5.2 The churn mapping (blended → 3 segments)

The engine averages the three segment churns **unweighted** (`(e+s+b)/3` in scoring, prompt, and physics). Therefore:

- **Default mapping [C3]:** set all three segments to the founder's blended monthly churn. `mean(x,x,x) = x` reproduces their number exactly; segment-specific shock multipliers then apply uniformly — a reasonable approximation.
- **Do not ask for customer mix percentages.** The engine has no mix weighting; collecting a mix would be data theater (input with no consumer). This corrects the prompt's suggested design.
- **Enriched tier [C1]:** founders who genuinely know per-segment churn can enter all three (fields already exist in `ScenarioConfig`); this improves shock realism in scenarios (e.g. recessions double only B2C churn).

### 5.3 Company age (`months_elapsed`) — required backend change [C2]

Tenure decay multiplies effective churn by `max(0.3, exp(−0.15 × 0.4 × months))`: a brand-new company runs at ~0.86×, a 24-month-old company at the 0.30 floor. Ignoring age overstates churn pressure for older companies by up to ~2.9×, distorting R&D urgency. Change needed: add `company_age_months` to `ScenarioConfig` and map to `months_elapsed` in `StartupEnv.reset`. Side effect to manage: hard shocks trigger at absolute months 24/48/72, so an age-20 company would hit a scripted shock 4 months into a scenario — §11.5 resolves this by parameterising the shock schedule for founder scenarios.

### 5.4 The burn mapping (the $8k problem) — required backend change [C2]

The engine computes salary burn as `headcount × $8,000` and runway as `cash ÷ (headcount × $8,000)`, ignoring marketing/R&D and any other opex. Real founders know **total monthly burn**, and their per-head cost is rarely $8k. Resolution:

- **Engine mapping:** set *engine headcount* = `round(founder_non_marketing_non_R&D_burn / 8000)`, so the engine's internal burn ≈ reality and every runway-keyed rule (oracle event trigger at runway < 12, CFO hiring gate at > 24, efficiency scoring) operates on truthful numbers. Founder's real headcount is stored separately for display.
- **UI mapping:** "hires" in recommendations are translated as **payroll capacity** ("room to add ≈ $16k/month of payroll — roughly 1–2 hires at typical salaries"), never as literal people, because the engine's hiring unit is an $8k salary slot.
- **Founder-facing runway** is always computed from their true numbers: `cash ÷ total monthly net burn` (with MRR offsetting), displayed with the formula on tap. The engine's internal runway is never shown.
- Preferred cleaner fix (still small): add `monthly_burn_override` to `ScenarioConfig` and have env/boardroom read it when present. Either variant is [C2]; the virtual-headcount mapping needs zero backend change and can ship first [C3].

### 5.5 System-estimated fields disclosure

`product_quality`, `innovation_factor`, macro fields, and (if unset) valuation multiple are **system-estimated**. The UI marks every such value with an "Estimated" chip wherever it influences visible advice (§15.1), with one-line explanations ("We assume typical market conditions: ~3% rates, normal consumer confidence").

### 5.6 The calibration envelope (must-state limitation)

Spend advice comes from fixed tiers and floors tuned to a $50k-MRR simulated company (CMO $2k/$10k/$20k; CPO $3k/$8k/$15k halved under $200k cash; R&D floor ≥ $20k under deficit; marketing floor max($5k, 2% MRR); sanity cap max(30% cash, $20k)). Consequences the product must own:

- For a $5k-MRR company, floors can exceed sensible spend; for a $2M-MRR company, tiers are trivially small. **Product rule:** present allocations primarily as **percent-of-MRR mix and priority ranking**, with dollar figures as "suggested starting point," and display a calibration note outside roughly $10k–$500k MRR: "Your company is outside the range this advisor is best calibrated for; treat amounts as directional."
- [C2 candidate for V1]: scale the boardroom's absolute floors/tiers by `initial_mrr / 50k` — a small, surgical backend change that makes dollar advice honest across sizes. Until then, [C3] percent-based presentation carries the load.

---

## 6. Input Complexity Analysis

### 6.1 Per-input burden assessment

| Input | Complexity | Availability | Sensitivity | Error likelihood | Change freq | Auto-derivable? |
|---|---|---|---|---|---|---|
| MRR | Low | immediate | med (financial) | low | monthly | no |
| Cash | Low | immediate | high (financial) | low | monthly | no |
| Total burn | Low | 1 look at P&L | high | med (gross vs net confusion — ask for costs, compute net) | monthly | no |
| Price/ARPU | Low | immediate | low | med (multi-plan averaging) | rarely | from MRR ÷ customers if customer count given |
| Blended churn | Med | most billing tools | med | **high** (monthly vs annual; logo vs revenue) | monthly | no |
| Company age | Low | known | low | none | auto-increments | yes after first entry |
| Crowdedness | Low | judgment | low | low | rarely | no |
| New customers / mkt spend | Low | CRM/P&L | med | low | monthly | — |
| CAC direct | Med | needs own analytics | med | med | monthly | yes (spend ÷ new) |
| Segment churns | High | cohort tooling | med | high | monthly | no — enriched only |
| Product maturity | Low (judgment) | immediate | low | subjective by design | quarterly | no |
| Valuation multiple | High | fundraise docs | high | med | rarely | default |

### 6.2 Progressive data strategy

- **Minimum viable input (first analysis unlocks):** MRR, cash, total burn, price, blended churn, company age, crowdedness + company name. **8 fields, ~3–5 minutes.** Justification against code: these populate every variable the Oracle prompt and boardroom rules read, except CAC/LTV — for which the first analysis uses the engine-default CAC ($50) *flagged as estimated* and derived LTV, and the UI immediately invites the CAC upgrade. The brief, risk assessment, plan, and runway are all genuinely computable from this set.
- **Recommended enriched input (better first analysis):** + new customers & marketing spend last month (→ real CAC → honest LTV:CAC → correctly-tiered marketing advice), + product maturity rating, + real headcount.
- **Advanced/optional:** per-segment churn, valuation multiple, macro override, R&D spend last month (used only for burn sanity and onboarding narrative — the engine plans R&D forward; it does not read current R&D spend as state).
- **Collection mechanism: structured form** (grouped, 3 short steps), not conversational — the inputs are numeric, few, and monthly-repeated; a chat flow adds latency and transcription errors to a 2-minute task. No integrations are invented: Stripe/accounting sync is listed honestly in Later (§24).

---

## 7. Onboarding

Design constraints honored: minimum input before first value; the "we know enough" moment is explicit; every screen states why a number is needed; skippable enrichment; no giant questionnaire; honest expectation-setting about simulation grounding; first analysis latency (~20–90 s) is used, not hidden.

### Screen O1 — Welcome & honest framing
- **Objective:** set the mental model before any input.
- **Copy (verbatim spec):** "Meet your AI advisory board. Three AI advisors — finance, growth, product — plus a strategist analyse your numbers and give you a monthly plan. Their experience comes from thousands of simulated startup scenarios, not real company data — so treat advice as a structured second opinion, not a prophecy."
- **Controls:** [Get started] · [How it works] (3-panel explainer of the loop from §4).
- **Saved:** nothing.

### Screen O2 — Your company (identity + age + market)
- Fields: company name (req), what you sell (one-line, optional — display only), **company age in months** (req), **market crowdedness** (req; "Just us / A few / Crowded" → 2/5/9 competitors with helper "How many direct competitors do customers compare you to?").
- Validation inline; progress "1 of 3"; [Continue].

### Screen O3 — Money (the state anchors)
- Fields: **MRR** (req), **cash in bank** (req), **total monthly costs** (req; helper: "Payroll + tools + rent + marketing + everything. We'll compute your burn and runway."), marketing spend last month (opt).
- Live feedback as they type: "Runway ≈ 14 months" computed as cash ÷ max(costs − MRR, small ε), with "assuming revenue and costs stay flat."
- Sanity checks: costs < $5k warns "unusually low — include salaries?"; burn > cash errors.

### Screen O4 — Customers (pricing + retention)
- Fields: **average price per customer per month** (req), **monthly churn %** (req; helper text + a monthly/annual toggle that converts annual→monthly to kill the #1 error), new customers last month (opt → "we'll compute your acquisition cost"), product maturity (opt 3-choice: Early & rough / Solid / Polished).
- **The "enough" moment:** on completing O4's required fields, a banner: **"That's enough for your first analysis."** [Run my first analysis] (primary) · [Add detail first] (opens enrichment accordion: segment churn ×3, real headcount, CAC direct, valuation multiple).
- **Post-submit:** scenario persisted (`POST /api/scenarios` shape, extended per §5); analysis job created.

### Screen O5 — First analysis in progress (the latency screen)
- Honest staged progress tied to real pipeline stages, no fake percent bars: "Reading your numbers ✓ → Your advisory board is deliberating (≈ half a minute)… → Writing up recommendations…" Elapsed-time counter. If narrative refinement is enabled, stage 3 says "Your advisors are writing their reasoning (~1 min)".
- Failure path per §17 (LLM down → rules-based advice banner, never a dead end).
- **Post-completion:** auto-navigate to first Advice view with a one-time overlay: "How to read this page" (3 callouts).

**What is saved when:** partial onboarding persists locally after each screen (resume on return); the scenario row is only created at O4 submit; nothing external is called before that.

---

## 8. Information Architecture

Five founder destinations plus settings. Research surfaces from the existing UIs (policy pickers, seeds, episode counts, evaluation tables, ablation toggles, similarity scores) are deliberately absent — they live on in an owner-only debug mode if ever needed, never in founder navigation.

```
Home  (position + this month's plan + what changed)          ← default after onboarding
│
├── Advice            (full analysis: risk, plan, reasoning, evidence, confidence)
│     └── per-recommendation detail (expanded card)
├── Stress-test       (scenario builder + results vs baseline)          [V1; §11]
├── History           (monthly timeline: numbers → advice → deltas)
├── My company        (edit inputs; data-provenance view; monthly update entry point)
└── Settings          (account, advisor narrative on/off, data export, delete)
```

| Item | Purpose | Primary content | Expected behaviour | Relationships |
|---|---|---|---|---|
| Home | "Where am I, what should I do, what changed" in 30 seconds | position banner, 3–4 KPIs, plan summary, change feed | scan → click into Advice or update numbers | pulls latest analysis + latest snapshot |
| Advice | the full monthly analysis | layered output per §9–10 | read 2–10 min; expand reasoning/evidence | source: brief + decision_trace + narratives |
| Stress-test | try a decision safely | scenario form + ranged results | configure → run (secs) → compare → save | uses same scenario state; writes scenario runs |
| History | continuity and accountability | timeline of updates, advice, deltas, events | monthly ritual review | reads snapshots + past analyses |
| My company | data honesty center | all inputs with provenance chips; update flow | monthly 2-min update | writes snapshots; triggers re-analysis |
| Settings | boring by design | account, toggles, export | rare | — |

Fewer-pages rationale: the backend-branch UI's five pages (Dashboard/Setup/Oracle/Shocks/Evaluation) partition by *system internals*; this IA partitions by *founder intent* (act now / understand / try / remember / maintain). Oracle & memory content becomes the **Evidence layer inside Advice**; Shock tracker content becomes History events + scenario stress mode; Evaluation disappears from founder view entirely.

---

## 9. Main Dashboard (Home)

What a founder sees Monday morning, top to bottom. Every element names its concrete data source; nothing is shown without an associated decision or action.

| # | Section | Information & why it matters | Interaction | Click destination | Data source | Updates |
|---|---|---|---|---|---|---|
| 1 | **Position banner** | One sentence + risk chip: "⚠ Elevated risk — churn is rising and runway is under 12 months." The single most load-bearing line in the product. | click | Advice | `brief.risk_level` (4-level chip, founder palette) + top `key_risks[0]` filtered through guardrail §15.4; staleness suffix if data > 35 days | on analysis |
| 2 | **KPI row (4 cards)** | Runway (months, from true burn §5.4) · MRR + MoM delta · Churn + trend arrow · Growth efficiency (LTV:CAC as "healthy/watch/unhealthy" bands at ≥3 / 1–3 / <1, thresholds from CFO/CMO rules & reward code) | hover = formula + provenance chips | My company (per-metric history in V1) | founder snapshots + `TrendContext` (`mrr_delta_pct`, `churn_delta`); arrows only when ≥2 snapshots | on update |
| 3 | **This month's plan (3–4 action cards, compact)** | e.g. "Marketing: hold ~$6k, stay on performance channels" · "Product: increase to ~$12k (priority)" · "Hiring: wait" · "Pricing: hold". Amounts as $ + % of MRR per §5.6. | click card | Advice → that recommendation expanded | `decision_trace.final_action`, channel from marketing.channel, priority = applied weight ranking | on analysis |
| 4 | **What changed** | Since last update: metric deltas + *why we re-analysed* in plain language ("Re-analysed early because MRR dropped more than 15%") | click | History | snapshot diff + `refresh_reason`/event trigger mapped to copy (§26) | on update/analysis |
| 5 | **Evidence peek (1 line)** | "In simulated situations like yours, the most common 6-month outcome was stagnation — plan focuses on retention." | click | Advice → Evidence layer | `expected_outcome` + retrieved memory outcome mix, always with the *simulated* qualifier | on analysis |
| 6 | **Data freshness footer** | "Numbers from Aug 1 · Update takes ~2 minutes" with primary [Update my numbers] after 25+ days | click | My company update flow | snapshot timestamp | daily |

Empty variants per §17. Nothing on Home shows: weights, enum names, oracle mode, memory similarity, seeds, Rule-of-40 (see §26 for its fate).

---

## 10. Analysis & Recommendation UX

### 10.1 The seven-layer output architecture

Raw outputs (brief enums, identical-scored proposals, weight vectors, modifier multipliers, memory documents with similarity scores, cache metadata) are never shown as-is. Each founder-facing layer maps to concrete sources; deeper layers are behind progressive disclosure ("Why → Evidence → Details").

| Layer | Founder sees | Concrete source | Transformation | Disclosure |
|---|---|---|---|---|
| **L1 Immediate answer** | "Your position: Elevated risk. Growth is slowing; the board recommends shifting budget from acquisition to retention this month." | `risk_level` + `growth_outlook` + top of `recommended_focus[]` | enum→copy tables (§26); one generated sentence assembled from enums, not free LLM text | always visible |
| **L2 The plan** | 4 recommendation cards (marketing/product/hiring/pricing) with amount, direction vs current, one-line rationale | `final_action` (+ channel), direction = vs last month's plan or founder's reported spend | $ + % of MRR; hires → payroll capacity (§5.4); pricing → hold / ~+5% (§3 J9) | always visible |
| **L3 Reasoning** | "Why this plan": 2–4 bullets, e.g. "Risk is elevated, so we scaled marketing back ~40% and protected product investment"; per-advisor 2-sentence rationales when narrative layer is on | weight deltas (base→applied), `ActionModifier` deltas already precomputed in trace (`marketing_spend_change_pct`, `rd_spend_change_pct`, `hires_change`), `recommended_focus[]`, [C2] narrative refinement (`use_llm` rationale) | modifier multipliers → verbal scale ("scaled back sharply/somewhat/slightly"); weights → "the board's focus this month: Product 45% · Efficiency 30% · Growth 15% · Market 10%" as a mix bar, no decimals | 1 tap |
| **L4 Evidence** | "What this is based on": your trends (2+ updates), similar simulated situations (up to 3, rewritten from memory documents: phase/churn/innovation tiers + what happened after 6 months), and — during shock-like conditions with v4_causal — "across 12 simulated rate-hike scenarios, median recovery was ~5 months" | `TrendContext`, `retrieved_memories[].document` + `realized_outcome`, `GraphContext.causal_summary` | strip similarity/recency/weights; template: "A simulated company at a similar stage (early, high churn) saw **decline** over the following 6 months." Mandatory prefix: *From simulations, not real companies* | 1 tap |
| **L5 Expected impact** | "If you follow this plan, in simulation": median 6-month MRR path + range band vs. "hold everything flat" | [C2] mini-rollout: N seeded no-LLM episodes from founder state with plan vs. flat action (milliseconds–seconds) ; v3+ `expected_outcome` as the qualitative headline | fan/band chart, median bolded; labelled "simulated range, not a forecast" | 1 tap (V1; MVP shows only qualitative `expected_outcome`) |
| **L6 Confidence & freshness** | "How sure is the board": qualitative band (Low/Moderate/High from `confidence` <0.4 / 0.4–0.7 / >0.7), *why analysed now* ("scheduled monthly" / "your churn jumped"), data age, count of estimated inputs | `brief.confidence`, `refresh_reason`, `brief_source`, snapshot timestamp, provenance ledger | never a percentage; §15 rules | visible as a compact strip; expands |
| **L7 Next actions** | checklist: "Set marketing budget to ~$6k" · "Schedule pricing review" · "Update numbers on Oct 1 (reminder?)" | derived 1:1 from L2 + update ritual | imperative phrasing; checkable (§13) | always visible at bottom |

### 10.2 Recommendation card (exact structure)

```
┌────────────────────────────────────────────────────────────────┐
│ PRODUCT & RETENTION                                 PRIORITY ★ │  ← domain + priority (from applied
│                                                                │     weight ranking)
│ Invest ≈ $12,000 in product this month     (~16% of MRR)       │  ← final_action.product.r_and_d_spend,
│ ▲ up from ~$8,000 last month                                   │     % of MRR, delta vs previous plan
│                                                                │
│ Why: Your churn (5.2%/mo) is the board's top concern, and      │  ← visible rationale: 1 sentence from
│ retention improves when product investment rises.              │     narrative layer if on, else enum-
│                                                                │     assembled template
│ ⓘ Based on your churn trend and 2 similar simulated cases      │  ← evidence teaser (count only)
│                                                                │
│ [Why this number?]  [See evidence]  [I'm doing this ✓]         │
└────────────────────────────────────────────────────────────────┘
   expanded "Why this number?": the L3 chain for this domain only —
   base proposal (CPO tier by churn) → strategic adjustment (modifier ×1.5, risk HIGH)
   → guardrails ("kept above the board's minimum product investment") — all verbal.
   expanded "See evidence": the L4 items filtered to this domain.
   "I'm doing this ✓" feeds the action plan (§13); "Adjust…" (V1) opens the amount for
   founder override, recorded as a decision (§14).
```

Confidence is shown **once per analysis** (L6 strip), not per card — the engine produces one brief confidence, not per-recommendation confidences, and faking granularity would be dishonest.

### 10.3 Copy assembly rules

- Enum-to-copy tables live in one place (§26) and are the only path from engine vocabulary to screen.
- LLM free text appears **only** in: per-advisor rationales (narrative layer) and `key_risks`/`key_opportunities` bullets — each rendered under the guardrail in §15.4 (length clamp, numeric-claim check, fallback to enum template).
- Numbers rounded: currency to nearest $500 below $20k, nearest $1k above; percentages to 0.1pp; never render engine floats raw.

---

## 11. Decision / Scenario UX ("Stress-test") — supported, with stated changes [C2/C3]

### 11.1 Classification honesty

- Simulating forward from a founder-derived state: **[C1-backend-branch]** (`environment_config` end-to-end).
- Overriding the *first-month action* to the founder's hypothetical ("raise price 10%", "cut marketing to $2k", "add $10k payroll"): **[C2]** — a thin wrapper agent that plays the founder's fixed action in month 0 (or months 0–k) then delegates to the boardroom policy. ~30 lines by pattern of existing agent wrappers.
- Paired baseline comparison: **[C1]** by construction — run the same seeds with and without the override; seeded `random`/`np.random` make pairs legitimate.
- Speed: default policy `boardroom` (no LLM) → 20 seeds × up to 24 months ≈ instantly perceptible (<2–3s); an optional single Oracle commentary call afterwards (~20s) is opt-in ("Ask the board to comment on this result").

### 11.2 How a founder describes a decision

Structured controls only (no free text in MVP/V1 — there is no NL→action parser and building one is unjustified [C4 declined]):

- Decision type chips: **Change price** (±% slider, −50%..+100% — the adapter's real bounds) · **Change marketing** ($ or % of MRR; channel toggle "performance (ppc) / brand") · **Change product spend** ($) · **Add payroll** ($/mo, translated to engine hires) · **Combination** (up to one per domain).
- Assumptions panel (visible, editable): horizon (6/12/24 months; default 12), "market stress" toggle (§11.5), "keep my other numbers as entered".

### 11.3 Run behaviour & loading

[Run stress-test] → 20 paired seeded episodes (decision vs. baseline), progress bar honest but brief ("Simulating 20 scenario runs…"); results persist as a named scenario run (backend `scenarios`+`simulation_runs` tables reused; add `kind=stress_test` and the override payload to the run record **[C2]**).

### 11.4 Result structure (fixed template)

1. **Headline (comparative, ranged):** "Raising price 10%: in 20 simulated runs over 12 months, median MRR ended **+6%** vs. holding price (range −8%…+19%). Runway improved in 17 of 20 runs."
2. **Band chart:** MRR (and cash) median lines + interquartile band, decision vs. baseline, same axes.
3. **Trade-offs:** auto-extracted from episode metrics deltas (churn, LTV:CAC, runway at horizon).
4. **Risks & caveats (always rendered):** "This is a calibrated simulation, not your market. Price elasticity in the simulator varies randomly between runs (−0.9…−0.2) — your customers may respond differently. Results assume your inputs stay accurate."
5. **Actions:** [Save scenario] [Compare with another] [Send to my plan] (adopts the override into this month's checklist, recorded as founder decision §14).

### 11.5 Shock policy in founder scenarios (decision + required change)

In-sim hard shocks fire at absolute months 24/48/72 (seed-cycled), which are meaningless relative to a founder's timeline and would fire mid-scenario for a company mapped to age ≥ 12 with horizon ≥ 12. **Decision: founder stress-tests exclude scripted hard shocks by default** (stochastic month-to-month volatility stays — it is the physics), and offer an explicit **"Add market stress"** toggle that injects one chosen shock (competitor surge / rate spike / downturn — translated names per §26) at scenario month 3, so founders can ask "…and what if a downturn hits while I do this?" **[C2]:** parameterise the shock schedule (`shock_months`, `shock_type`) via `initial_config` instead of the hardcoded `{24,48,72}` / `seed % 3`.

### 11.6 Presentation red lines

Never a single trajectory presented as *the* outcome; never the word "forecast/prediction"; the phrase **"in simulation"** appears in the headline itself, not a footnote; ranges always shown; baseline always shown.

---

## 12. AI Advisor UX (conversational) — deliberately absent in MVP/V1

Decision: **no chatbot.** Justification against real capability: the only free-text reasoning path ever written (`OracleAgent.analyze_situation`) is dead code wired to seed-data stores the live system doesn't use; the live Oracle consumes a fixed state schema, not questions. A credible Q&A advisor is therefore [C4] (new prompt path, new grounding contract, new hallucination surface) with weak incremental value over the structured layers, at ~20s per exchange.

What replaces it:
- **Contextual "explain this" everywhere [C3]:** every metric, recommendation, and evidence item has a deterministic explainer assembled from the decision trace — instant, grounded, zero hallucination risk.
- **"Ask the board to comment" on stress-test results [C2, V1-optional]:** one templated LLM call whose prompt embeds the scenario deltas (reuses the refinement pattern in `proposal_agents`), rendered as a labelled advisor note with the §15.4 guardrail. Bounded scope, one call, clearly not a chat.
- **Later (§24):** if a conversational surface is ever built, it must answer only from the stored state/trace/history and visibly decline out-of-scope questions (J11–J14).

---

## 13. Actions & Follow-Through

The Insight → Decision → Action → Update → Outcome → New-Analysis loop, and the persistence it requires.

- **Plan checklist [C3+C4-persistence]:** L2 cards convert to checkable intentions ("Set marketing budget ~$6k"; "Hold pricing"). State: suggested → accepted ("I'm doing this") → done / skipped ("I did something else" + optional amount). Stored per month **[C4: `company_months.decisions` — see §21]**. No task-manager ambitions (no due dates/assignees) — it is a decision record, not project management.
- **Targets:** each accepted card sets this month's reference values; next update renders plan-vs-actual ("You planned ~$6k marketing; you reported $9k").
- **The monthly update ritual (the product's heartbeat):** entry from Home footer, History, or (Later) email nudge. Pre-filled with last month's values; founder edits ~6 numbers; diff preview ("MRR +7% · churn −0.4pp"); submit → snapshot persisted → **re-analysis** triggered honoring the engine's real event semantics: the product's monthly ritual *is* the cadence refresh, and larger-than-threshold changes surface the event copy ("re-analysed because your churn jumped +1.8pp") sourced from `refresh_reason` [C1 logic, C2 to route founder updates through Oracle.observe_state so trends/cache behave identically to in-sim].
- **Why the ritual pays off (engine-true):** update #2 activates trend context (`history_points ≥ 2` gates the trends block of the prompt); update #6 onward, each earlier month matures into an outcome-labelled memory of the founder's **own** company (horizon = 6 months, `classify_realized_outcome` ±10% MRR) — provided memory scoping is switched from per-run to per-company **[C2: run_id := company_id]**. From month 7, the Evidence layer can truthfully say "your own May situation resolved as GROWTH."
- **Outcome closure:** when a month matures, History annotates it: "Plan: shift to retention → 6 months later: churn −1.1pp, MRR +14% (simulator label: GROWTH)." Correlation, not credit — copy says "what happened next," never "because you followed the plan."

---

## 14. History & Learning

**Form chosen: a single vertical monthly timeline** (not a feed, not snapshot cards): one entry per update/analysis/decision/stress-test, newest first, because the founder's mental model is "months of my company," the data is inherently monthly, and a timeline makes the Decision→Outcome pairing legible. A feed would fragment one month into many items; snapshot grids hide causality.

Timeline entry anatomy: date · entered numbers (compact) · deltas vs previous · risk chip then · plan summary · founder's recorded decisions (accepted/skipped/custom) · matured outcome badge (from month +6) · events ("advisor flagged churn jump"). Filters: All / Decisions / Advice changes. Each entry expands to the full archived Advice view (analyses are persisted, never recomputed [C4 storage]).

Metric mini-charts (MRR, churn, runway over months) live at the top of History once ≥3 snapshots exist — the same data the engine's trend context consumes, so the chart and the advice never disagree.

---

## 15. Trust, Evidence & Explainability

### 15.1 The honesty contract (product-wide rules)

1. **Simulated vs. observed:** any evidence derived from simulation (memories, causal stats, stress-tests, expected outcomes) carries the "From simulations" label at the point of display — in the sentence, not a tooltip. The words *forecast, prediction, will* are banned in generated surfaces; *in simulation, tended to, median run* are the house style.
2. **Provenance chips on every number:** `You provided (Aug 1)` · `Estimated by the system` · `Derived (price ÷ churn)` · `Simulated`. My-company page shows the full ledger.
3. **Staleness:** advice header shows data age; > 35 days switches the position banner to "Based on numbers from {date} — update for current advice" and Home's primary CTA becomes the update.
4. **Refresh honesty:** `refresh_reason` → plain copy: initial → "your first analysis"; cadence → "scheduled monthly review"; event → the specific trigger ("runway fell below 12 months", "MRR dropped ≥15% since last analysis", "churn rose ≥1.5 points", "market conditions worsened"). `brief_source=cache_hit/reuse` → "consistent with your last analysis" (never pretend fresh reasoning happened).
5. **Confidence without fake precision:** the 0–1 scalar renders as Low/Moderate/High only. Enum levels never become percentages. No p-values, similarity scores, seeds, weights-as-numbers anywhere in founder view.
6. **Limitations page ("What this advisor can and can't see"):** static, linked from every Evidence layer: no real-company data; doesn't know your product, team, or market specifics; macro conditions assumed typical unless set; calibration envelope (§5.6); advice ≠ financial advice.

### 15.2 LLM-failure surfacing (replaces the silent neutral fallback) [C2 — required]

Backend change: `parse_llm_response` and the LLM client must return an explicit failure/fallback flag instead of a silent `default_neutral_brief()` (smallest viable change: `(brief, parse_ok)` or a `brief.meta_source` field; plus empty-LLM-output detection). Frontend contract: when the flag is set, the analysis renders in **rules-based mode** — the boardroom's heuristic plan is real and useful without the Oracle — with a banner: "The AI strategist couldn't be reached; this plan comes from the board's built-in rules. Analysis will retry automatically." Degraded-but-honest beats confident-and-fake; a neutral MEDIUM/STABLE brief presented as considered analysis is the one failure mode this product must never ship.

### 15.3 Evidence rendering (memories & causal stats)

Memory documents ("Phase: EARLY | Churn: HIGH … After 6 months the realized outcome was DECLINE") render as: "**A simulated company at a similar stage** — early revenue, high churn, product losing momentum — **declined over the following 6 months.** The plan below is shaped to avoid that path." Max 3; outcome dots (▲growth ▼decline ▬flat); no similarity numbers (the engine's own re-ranking already ordered them). Causal stats (v4_causal, shock active): "Across 12 simulated rate-hike scenarios at your stage, the median recovery took ~5 months; recoveries were faster when boards cut acquisition spend early." Only rendered when `total_occurrences > 0`.

### 15.4 `key_risks` / narrative guardrail [C3 rule + C2 nice-to-have]

LLM bullets are: clamped (≤ 140 chars, ≤ 3 shown); scanned for numeric claims not present in state/trace (a number that matches no input within tolerance → bullet replaced by enum-templated fallback); never merged into the position banner unless they pass the scan. Rationale: `llama3.1:8b` free text is fluent (see ollama_test_results.csv) but unverified.


---

## 16. Complete Screen Inventory

Twelve screens. Each row is concrete enough to visualise; wireframes for the six core screens follow in §17. "Backend dependency" names the §21/§28 endpoint.

| # | Screen | Purpose / primary founder question | Entry points | Layout top→bottom | Data required | Key interactions | Backend dependency | Empty/loading/error | Mobile behaviour | Next actions |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | Welcome | "What is this and can I trust it?" | first visit | value prop → honesty framing → CTA | none | Get started / How it works | none | n/a | single column | O2 |
| S2 | Sign-in / account | access | pre-onboarding or later | email+password or magic link | account | auth | **[C4] auth service** | error: bad credentials | native inputs | O2 or Home |
| S3 | Onboarding O2–O4 | "Describe my company" (§7) | S1/S2 | 3 grouped steps, progress, live runway feedback | §5 minimum set | typed inputs, unit toggle, enrichment accordion | `POST /api/companies` (+scenario shape) | inline validation; resume partial | one field-group per viewport | S4 |
| S4 | First-analysis progress | "Is it working?" | O4 submit; any re-analysis | staged honest progress (§7 O5) + elapsed | job status | cancel? no (jobs are short); leave-and-return safe | `POST /api/advise` (async job) + polling | LLM-down → §17.7 rules-based path | full-screen | S6 |
| S5 | Home | "Where am I, what do I do, what changed?" (§9) | default | banner → KPIs → plan → changes → evidence peek → freshness | latest snapshot+analysis | click-through everywhere | `GET /api/companies/{id}/latest` | §17.1–.3 variants | cards stack; banner sticky | S6/S8/S10 |
| S6 | Advice | "The full monthly analysis" (§10) | Home, History | L1 banner → L2 cards → L6 strip → L3/L4/L5 expandables → L7 checklist | analysis record (brief+trace+narratives) | expand cards, accept actions, evidence drill | `GET /api/analyses/{id}` | archived analyses always renderable; partial-narrative state | cards stack; expandables become sheets | S7/S8/S10 |
| S7 | Recommendation detail | "Why this number?" | S6 card | expanded card (§10.2): chain → evidence → controls | decision_trace slice | accept / adjust (V1) / evidence | same record | — | full-screen sheet | back to S6 |
| S8 | Stress-test builder | "What if I…?" (§11) | Home, S6 | decision chips → params → assumptions → run | current state | configure, run, stress toggle | **[C2] `POST /api/stress_tests`** | invalid combo inline; run-fail retry | steppers over sliders | S9 |
| S9 | Stress-test result | "Should I do it?" | S8, History | headline range → band chart → trade-offs → caveats → actions | paired run results | save, compare, send-to-plan, optional board comment (~20s labelled) | same + `GET` | running state secs; comment-call spinner separate | chart horizontally scrollable | S6/S10 |
| S10 | History | "What happened over months?" | nav | mini-charts (≥3 snapshots) → timeline (§14) | snapshots, analyses, decisions, stress-tests | expand months, filters | `GET /api/companies/{id}/months` | §17.10 no-history | timeline is naturally mobile | S6 archived |
| S11 | My company | "My data, its provenance; update it" | nav, Home footer | provenance ledger → grouped current values → [Update numbers] flow (pre-filled diff §13) | scenario + snapshots | edit, update ritual, unit helpers | `POST /api/companies/{id}/months` | stale banner; validation | grouped accordions | S4 (re-analysis) |
| S12 | Settings | account, narrative toggle ("richer explanations, ~1 min longer analysis"), export, delete | nav | simple list | account | toggles | **[C4] auth/export** | — | native | — |

Deliberately absent from founder surfaces: policy/seed/episode/oracle-frequency controls, ablation toggles, evaluation tables, memory-similarity views, raw trace browsers. If the owner keeps a research console, it lives behind a separate route/flag outside this IA.

---

## 17. Low-Fidelity Wireframes

Information hierarchy only; visual design inherits the existing `styles.css` system (§19).

### 17.1 Onboarding — Money step (O3)

```
┌──────────────────────────────────────────────────────────────┐
│  Startup Society          Step 2 of 3 · Money   [●●○]        │
├──────────────────────────────────────────────────────────────┤
│  Your money                                                  │
│  We use these to compute runway and size this month's plan.  │
│                                                              │
│  Monthly recurring revenue        [$  30,000      ] /month   │
│  Cash in the bank                 [$ 220,000      ]          │
│  Total monthly costs              [$  48,000      ] /month   │
│    payroll + tools + marketing + everything                  │
│  Marketing spend last month (opt) [$   6,000      ]          │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ≈ Runway: 12 months   (cash ÷ net burn, all else flat) │  │
│  └────────────────────────────────────────────────────────┘  │
│                                    [ Back ]  [ Continue → ]  │
└──────────────────────────────────────────────────────────────┘
```

### 17.2 Home (S5)

```
┌────────┬─────────────────────────────────────────────────────────────┐
│ SSoM   │  Acme Analytics                      numbers from Aug 1 ✓   │
│        │ ┌─────────────────────────────────────────────────────────┐ │
│ ▸Home  │ │ ⚠ ELEVATED RISK — churn is rising and runway is under   │ │
│  Advice│ │   12 months. This month: protect retention.   [Details] │ │
│  Stress│ └─────────────────────────────────────────────────────────┘ │
│  History│ ┌──────────┐┌──────────┐┌──────────┐┌──────────┐           │
│  Company│ │ RUNWAY   ││ MRR      ││ CHURN    ││ EFFICIENCY│          │
│  ⚙     │ │ 11 mo    ││ $30.0k   ││ 5.2%/mo  ││ Healthy  │           │
│        │ │ ▼ from 13││ ▲ +4.2%  ││ ▲ +0.8pp ││ LTV≈3.4× │           │
│        │ └──────────┘└──────────┘└──────────┘└──────────┘           │
│        │  THIS MONTH'S PLAN                          [Full advice →] │
│        │  ★ Product  ≈$12k (↑)   · retention is the priority         │
│        │    Marketing ≈$6k (↓)   · performance channels only         │
│        │    Hiring    wait       · Pricing  hold                     │
│        │  WHAT CHANGED — re-analysed early: churn rose ≥1.5 points   │
│        │  ⓘ In simulated cases like yours, most paths stagnated —    │
│        │    the plan targets retention first.   [See evidence]       │
│        │  ── Update your numbers (~2 min) ─────────── [ Update → ]   │
└────────┴─────────────────────────────────────────────────────────────┘
```

### 17.3 Advice (S6) — collapsed state

```
┌───────────────────────────────────────────────────────────────┐
│ Advice — August                     analysed Aug 1 · Moderate │
│                                     confidence · monthly review│
│ ⚠ Elevated risk. Growth slowing; shift budget from            │
│   acquisition to retention this month.                        │
│                                                               │
│ [★ PRODUCT ≈$12k ↑]  [MARKETING ≈$6k ↓]  [HIRING wait]        │
│ [PRICING hold]                       ← §10.2 cards, stacked   │
│                                                               │
│ ▸ Why this plan (board's focus: Product ▓▓▓▓ Efficiency ▓▓▓   │
│    Growth ▓ Market ▓; risk elevated → marketing scaled back)  │
│ ▸ Evidence — from simulations, not real companies (3 items)   │
│ ▸ Expected in simulation: stagnation risk unless retention    │
│    improves                                                   │
│                                                               │
│ NEXT ACTIONS                                                  │
│ [ ] Set marketing budget to ≈$6k                              │
│ [ ] Allocate ≈$12k to product work                            │
│ [ ] Update numbers Sep 1                     [remind me]      │
└───────────────────────────────────────────────────────────────┘
```

### 17.4 Stress-test (S8+S9 combined view)

```
┌───────────────────────────────────────────────────────────────┐
│ Stress-test a decision            [simulation, not a forecast]│
│ WHAT IF I…  (Price ±%) (Marketing $) (Product $) (Payroll $)  │
│   Price change: [ +10% ]     Horizon: (6)(12)(24) months      │
│   Market stress: [off] ▸ add a downturn / rate spike / rival  │
│                                          [ Run stress-test ]  │
│ ───────────────────────────────────────────────────────────── │
│ RESULT — 20 simulated runs, 12 months                         │
│ Median MRR +6% vs holding price (range −8%…+19%);             │
│ runway improved in 17 of 20 runs.                             │
│   $ ┤        ╭─────══════╮   ══ with +10%   ── baseline       │
│     ┤   ╭────╯░░░░░░░░░░░│   ░ middle-half band               │
│     ┤───╯                                                     │
│     └┬────┬────┬────┬────┬                                    │
│      0    3    6    9    12 months                            │
│ Trade-offs: churn +0.3pp median · LTV:CAC +0.4                │
│ ⚠ Caveats: simulated elasticity varies run to run; your       │
│   customers may respond differently.                          │
│ [Save] [Compare another] [Send to my plan] [Board comment ⏱]  │
└───────────────────────────────────────────────────────────────┘
```

### 17.5 History (S10)

```
┌───────────────────────────────────────────────────────────────┐
│ History        [All ▾]        MRR ▁▂▃▃▅ churn ▅▃▃▂▂ runway ▃▃▂│
│ ── AUGUST ──────────────────────────────────────────────────  │
│ ⚠ Elevated · MRR $30.0k (+4.2%) · churn 5.2% (+0.8pp)         │
│ Plan: retention first · You accepted 2 of 3 actions           │
│ ▸ open full advice                                            │
│ ── JULY ─────────────────────────────────────────────────────  │
│ ● Moderate · MRR $28.8k · churn 4.4%                          │
│ Stress-test run: "+10% price" → saved                         │
│ ── FEBRUARY ─────────────────────────── outcome matured ────  │
│ Plan: cut acquisition, protect product                        │
│ 6 months later: MRR +14%, churn −1.1pp  ▲ (simulator label:   │
│ growth) — what happened next, not credit                      │
└───────────────────────────────────────────────────────────────┘
```

### 17.6 Analysis in progress / degraded (S4 + failure)

```
┌───────────────────────────────────────────────┐   on LLM failure:
│   Your advisory board is deliberating…        │   ┌─────────────────────────────┐
│   ✓ Reading your numbers                      │   │ ⚠ The AI strategist couldn't │
│   ● Board deliberating   (~half a minute)     │   │ be reached. Showing the      │
│   ○ Writing up recommendations                │   │ board's rules-based plan —   │
│   elapsed 0:19                                │   │ still grounded in your       │
│   You can leave; we'll keep your seat.        │   │ numbers. [Retry analysis]    │
└───────────────────────────────────────────────┘   └─────────────────────────────┘
```

---

## 18. Visualization Inventory

Every visual answers a founder question; decoration-charts (the mockup's `mini-chart` gradient boxes) are removed.

| Visual | Question answered | Data | Why this form | Interaction | MVP? |
|---|---|---|---|---|---|
| Risk chip (4-state) | "How worried should I be?" | `risk_level` | categorical → color+word chip; no gauge theatrics | click → Advice | ✔ |
| Runway number + trend | "How long can I survive?" | snapshots (true burn) | a number founders already think in; chart adds nothing at n<3 | hover formula | ✔ |
| KPI delta arrows | "Better or worse?" | `TrendContext` deltas | direction+magnitude, no sparkline until ≥3 points | hover exact | ✔ |
| Plan allocation cards (with $ and % of MRR) | "Where does money go this month?" | `final_action` | cards beat donut: 4 items with actions attached; a donut invites proportional misreading of tier-based numbers | accept/expand | ✔ |
| Board focus mix bar | "What is the board optimising for?" | applied weights | single stacked bar, verbal labels, no numerals — communicates emphasis without exposing the weight machinery | tap → L3 | ✔ |
| Evidence outcome dots | "How did similar simulated cases end?" | memory `realized_outcome` | ▲▬▼ glyphs + sentence; anything fancier overstates 3 data points | tap → full item | ✔ |
| MRR / churn / runway mini-lines (History) | "My trajectory?" | ≥3 snapshots | small multiples, shared month axis | tap → expand | ✔ (appears at n≥3) |
| Timeline (History) | "What happened, what did I decide, what followed?" | months+decisions+outcomes | §14 rationale | expand months | ✔ |
| Stress-test band chart | "Range of outcomes if I do X?" | paired seeded runs | median + IQR band vs baseline is the honest shape for stochastic sim; single lines banned | hover month values | V1 |
| Plan-vs-actual chips | "Did I do what I planned?" | decisions vs reported | tiny paired chips in update diff | — | V1 |
| Shock/stress event markers | "When did conditions turn?" | `shock_label`s in scenario traces / event triggers in history | vertical markers on charts (pattern already in backend-branch chart) | hover label | V1 |
| Efficiency (LTV:CAC) health chip | "Is growth spend paying back?" | derived LTV:CAC vs 3.0/1.0 thresholds | banded chip; the raw ratio shown on hover for the curious | hover | ✔ |

---

## 19. Existing Technical Frontend — Reuse Analysis

Two artifacts assessed: the static mockup (checked-out `frontend` branch) and the API-wired app (`backend` branch). They share the shell and stylesheet; the wired one adds the data layer.

| Existing element | Current technical purpose | Founder relevance | Reuse as-is | Adapt | Remove | Replacement |
|---|---|---|---|---|---|---|
| Vite + React 18 + `@vitejs/plugin-react`, dev proxy (`vite.config.js`, backend branch) | build/dev toolchain | none directly — sound foundation | ✔ | | | — |
| `styles.css` design system (tokens: purple #3c3489, semantic green/amber/red/blue, Inter, light theme; sidebar/topbar shell; kpi-card, panel, status-pill, memory-card, table, form-grid components) | research dashboard skin | high — clean, calm, legible | ✔ tokens & primitives | extend: risk-chip states, provenance chips, timeline, band chart, progress stages | | — |
| `api()` fetch helper + error propagation (backend branch) | API access | high | ✔ | add auth header + typed errors | | — |
| 1.5s polling of run status (backend branch) | job progress | high — matches async-analysis UX | ✔ pattern | poll `/api/advise` + stress-test jobs; back-off | | (SSE Later) |
| Scenario form: `defaultScenario` + `fieldGroups` + controlled inputs (backend branch) | expose every EnvState field raw | medium — the bones of onboarding | | ✔ heavily: regroup per §7, founder labels, hide `product_quality`/`innovation_factor`/macro behind estimates, add helpers/validation/derivations | raw macro & internals fields | §7 onboarding |
| `TrajectoryChart` real-data SVG polyline + shock markers (backend branch) | plot episode 0 MRR | medium | | ✔ generalise into band/mini-line components | | §18 charts |
| Empty/Progress states (`EmptyState`, `RunProgress`, spinner) | run lifecycle | high | ✔ pattern | staged copy per §7 O5 | | — |
| Formatters `money/percent/decimal/months` | display | high | ✔ | rounding rules §10.3 | | — |
| lucide-react icon set | icons | high | ✔ | | | — |
| Sidebar/topbar shell + nav model | page switching | medium | | ✔ founder IA (§8); add router for shareable URLs [C4-small] | | — |
| **Policy `<select>` in topbar** (backend branch) | choose research policy | **anti-founder** | | | ✔ | fixed server-side default (boardroom+oracle_v3-class for analysis; boardroom for stress-tests) |
| **Episodes / seed / oracle-frequency inputs** | experiment config | anti-founder | | | ✔ | server-side constants |
| **Evaluation page (runs table, policy comparison)** | research results | anti-founder | | | ✔ | History (different concept) |
| **Oracle & memory page showing similarity scores** | retrieval inspection | leaks machinery | | ✔ content → Evidence layer with §15.3 rewriting | similarity/recency numerals | §10 L4 |
| **Shock tracker page + hardcoded Neo4j chain graphic** | shock research view | leaks machinery; graphic is fake | | fold real events into History/stress results | ✔ page + fake graphic | §14, §11 |
| Mockup's hardcoded KPI/agent/memory/policy constants | visual filler | none | | | ✔ all | live data |
| Sidebar "oracle_v4_causal · every 5 months" status | mode display | jargon | | | ✔ | "Advisor: active" + data freshness |
| Backend: FastAPI app, CORS, static serving, SQLite schema & WAL, threaded runs, `summary_json` builder | run execution & persistence | high — the service substrate | ✔ | extend per §21 endpoints; rename run→analysis concepts | | — |

**Foundation verdict:** keep the stack, the stylesheet, the fetch/polling pattern, the persistence approach; replace the information architecture and every research-facing control; the founder product is a **new set of pages on the existing chassis**, not a restyle of the research pages.

---

## 20. Frontend ↔ System Data Mapping

Per-component contract. Sources marked ⛔ have **no current backend support anywhere** (gap → §21); sources marked ⚠ exist only on the `backend` branch.

| UI component | Input required | Backend/system source | Transformation | UI output |
|---|---|---|---|---|
| Onboarding form | founder fields §5 | ⚠ `ScenarioConfig` (+⛔ `company_age_months`, `monthly_burn_override`, real headcount, maturity rating) | churn distribute (§5.2), burn→virtual headcount (§5.4), crowdedness→competitors, maturity→product_quality | persisted company |
| Position banner | latest analysis | ⛔ `POST /api/advise` → stored analysis (brief+trace) | enum→copy; guardrailed key_risk; staleness | one sentence + chip |
| KPI cards | snapshots + trends | ⛔ `company_months` store; `TrendContext` (C1 logic, needs history routing) | §9 formulas; bands | 4 cards |
| Plan cards | `decision_trace.final_action`, weights, modifier deltas | same analysis record | §10.2; $/%-of-MRR; payroll translation | recommendation cards |
| "Why this plan" | `base_weights`, `applied_weights`, `*_change_pct`, `recommended_focus` | analysis record | multiplier→verbal scale; mix bar | reasoning bullets |
| Advisor narratives | per-agent refined `expected_impact` | ⚠ `use_llm` path (fixed on backend branch) + ⛔ productised toggle & storage | §15.4 guardrail | 2-sentence rationales |
| Evidence items | `retrieved_memories[]`, `GraphContext.causal_summary` | analysis record (v3+/v4_causal) + ⛔ per-company memory scoping (`run_id:=company_id`) | §15.3 rewriting; strip scores | labelled evidence cards |
| Expected-in-simulation | `expected_outcome` (v3+); V1: mini-rollout | ⛔ rollout endpoint (physics-only, from state) | qualitative headline; V1 band | L5 |
| Confidence & freshness strip | `confidence`, `refresh_reason`, `brief_source`, timestamps, provenance | analysis record + company store | banding; reason copy | L6 strip |
| Failure banner | parse/LLM failure flag | ⛔ parser/client must expose failure (§15.2) | — | rules-based mode banner |
| Stress-test builder/result | overrides + horizon + stress toggle | ⛔ `POST /api/stress_tests` (paired seeded no-LLM runs; ⚠ engine pieces exist: `environment_config`, seeding, traces; ⛔ action-override wrapper, shock-schedule param) | §11.4 aggregation to median/IQR | headline+band+trade-offs |
| Update ritual | new month numbers | ⛔ `POST /api/companies/{id}/months` (+ route through `Oracle.observe_state`) | diff; event-trigger check (C1 thresholds) | what-changed + re-analysis |
| History timeline | months, analyses, decisions, stress runs, matured outcomes | ⛔ company store + ⚠ run persistence pattern | §14 assembly; outcome labels via `classify_realized_outcome` (C1) | timeline |
| Checklist | plan + founder responses | ⛔ decisions store | — | checkable plan |
| Settings/auth | account | ⛔ auth service | — | — |

**Confirmed gap candidates from the prompt, verdicts:** founder-state ingestion endpoint — ⚠ largely exists (`/api/scenarios` + `environment_config`), missing only the advise-now form; **single-month "advise now" entry point — ⛔ confirmed gap** (runner only loops episodes; but `Boardroom.decide(state)` is env-free, so the endpoint is thin) ; user accounts/persistence — ⛔ confirmed; multi-startup workspaces — ⛔ (schema trivially extends: scenarios already row-per-config; memory scoping supports it via run_id); scenario endpoint — partially ⚠ (full-episode runs exist; decision-override stress-tests ⛔); narrative productisation — ⚠ mechanism exists, wiring ⛔; per-company memory scoping — ⛔ one-line change plus lifecycle; honest failure surfacing — ⛔ confirmed.

---

## 21. Missing Capabilities / Product Gaps

Complete, deduplicated register. "Size" is a relative engineering judgment given the verified codebase.

| # | Gap | Class | What must be built / changed | Size |
|---|---|---|---|---|
| G1 | **Advise-now endpoint** — analysis of a founder state without running episodes | [C2] | `POST /api/advise`: build `EnvState` from company record (+history via `Oracle.observe_state` replay), call `Boardroom.decide(state)` once, store brief+trace as an analysis row; async job with status | S |
| G2 | **Company months store** — per-company snapshots, analyses, decisions, matured outcomes | [C4] | new tables (`companies`, `company_months`, `analyses`, `decisions`); the existing SQLite layer is the pattern | M |
| G3 | **Honest failure flag** in parser/LLM client (kills the silent neutral brief) | [C2] | return/parse-ok flag or meta field; runner/API propagate; UI banner | S |
| G4 | **Company age → `months_elapsed`** | [C2] | ScenarioConfig field + env mapping (§5.3) | S |
| G5 | **Burn override** (or virtual-headcount mapping server-side) | [C2]/[C3] | §5.4; config field read by env & runway estimators | S |
| G6 | **Stress-test endpoint** — paired seeded runs with first-month action override | [C2] | wrapper agent + `POST /api/stress_tests` + result aggregation | M |
| G7 | **Shock-schedule parameterisation** for founder scenarios | [C2] | `shock_months`/`shock_type` via initial_config replacing hardcoded {24,48,72}/seed%3 | S |
| G8 | **Per-company memory scoping & lifecycle** | [C2] | `run_id := company_id` at Oracle construction; monthly `observe_state` on update; maturation runs on real months | S–M |
| G9 | **Narrative layer productisation** | [C2] | settings toggle → `use_llm` agents on analysis path (backend-branch fix already exists); store narratives with analysis; timeout/fallback per narrative | S–M |
| G10 | **Accounts/auth + multi-company workspaces** | [C4] | standard auth; `companies.owner_id`; every query scoped | M |
| G11 | **Scale calibration of tiers/floors** (dollar advice honest across MRR sizes) | [C2] | scale boardroom absolute constants by `initial_mrr/50k` (flagged: changes research-comparability — keep behind product flag) | S code / M validation |
| G12 | **Mini-rollout for L5 expected impact** | [C2] | reuse stress-test machinery with plan-vs-flat actions | S (after G6) |
| G13 | **Brief-level founder narrative** (optional alternative to per-agent) | [C2] | add one text field to the brief prompt/schema + guardrail | S |
| G14 | Macro auto-fill from real data | [C4-Later] | external data source; until then defaults+override [C1/C3] | M |
| G15 | Routing/shareable URLs, reminders/notifications, export | [C4-small] | react-router; email jobs; CSV/JSON export of own data | S–M |
| G16 | Real-world integrations (Stripe/accounting), benchmarks, conversational advisor | [C4-Later] | §24; never implied before existing | L |

Latent defects found during inspection that the product work should fix in passing: `oracle_v3_hetero` crash on this branch (fixed on backend branch); backend-branch UI referencing nonexistent `brief.rationale`; burn-constant inconsistency ($8k/$10k, §2.2-6); initial LTV=7k vs derived ≈1.7k inconsistency (§2.2-9).

---

## 22. MVP — the smallest coherent founder product

**Scope sentence:** *One founder, one company: describe it in 8 fields, get an analysed position and an explained monthly plan in under two minutes of waiting, come back next month, update six numbers, and see what changed.*

**In:** onboarding §7 (minimum set + enrichment accordion) · advise-now analysis (1 LLM call; narrative layer **off** by default) · Home §9 · Advice §10 L1–L4+L6–L7 (L5 = qualitative `expected_outcome` only) · monthly update ritual + what-changed · History as list-of-months (mini-charts appear at n≥3) · honest states §17 incl. rules-based degradation · provenance/staleness/simulation labels §15 · single account (G10 minimal: email magic-link).

**Backend for MVP:** G1, G2, G3, G4, G5, G8 (S/M items only) on the merged backend-branch chassis. Explicitly **out:** stress-tests (G6/G7/G12), narrative default-on (G9 ships behind a flag), multi-company, calibration scaling (G11 — MVP relies on §5.6 percent-first presentation + envelope note), reminders, export, integrations, conversational anything.

**Why this is coherent:** it exercises the engine's full reasoning spine (state → brief → weighted/modified plan → trace) and the loop that makes month 2 better than month 1 (trends) — the two things that differentiate the product — while deferring everything whose absence doesn't break the core promise.

**MVP proves:** a founder who knows 8 numbers gets advice they can explain to a co-founder, with every number traceable to something real.

## 23. V1 — the strong usable product

Adds, in priority order: **Stress-tests** (G6+G7, §11) with save/compare and send-to-plan · **narrative layer on by default** (G9; settings toggle to disable, latency copy) · **L5 simulated-impact bands** (G12) · **CAC/LTV enrichment flow** + per-segment churn · **plan-vs-actual** tracking and decision records (§13 full) · **multi-company workspaces** (G10 full) · **calibration scaling** (G11) or, failing validation, hard envelope gating · matured-outcome badges in History (first available at month 7 of usage) · per-company evidence ("your own May") once memories exist · routing/shareable archived analyses · CSV export.

## 24. Later — valuable, deferrable

Reminders/email digests ("your monthly update is due; churn was the watch-item") · macro auto-fill (G14) · billing/accounting integrations that pre-fill the update (the single biggest friction-remover; honestly labelled as the moment "monitoring" language becomes permissible) · causal-graph evidence surfaces once enough per-company/global v4_causal history accumulates (Neo4j provisioning + corpus) · a constrained conversational advisor (only if grounded per §12) · team seats/read-only investor links · benchmark data (only with a real dataset; otherwise J11 stays refused) · mobile apps (mobile web suffices meanwhile).

---

## 25. Founder Cognitive-Load Audit

Audit questions applied per §4.24: most-important-thing obvious? next action clear? technical leakage? unknowable data requested? information without a decision? metric overload? recommendations actionable? can users see why / what changed / what's missing?

| Screen | Finding | Simplification adopted |
|---|---|---|
| Onboarding | churn is the one high-error field; "total costs" risks gross/net confusion | unit toggle + one-line definitions inline (not tooltips); live runway feedback converts data entry into immediate value; 8 required fields is the ceiling — every additional required field must displace one |
| Home | risk of KPI creep; original mockup showed 5 research KPIs with no action | hard cap: 1 banner + 4 KPIs + 1 plan block + 1 change block; every element clicks somewhere; Rule-of-40 and valuation excluded (no founder decision attaches at this stage) |
| Advice | seven layers could read as seven walls | collapsed default shows exactly: verdict, 4 cards, checklist — one screen; everything else is opt-in expansion; jargon test: the words oracle/boardroom/policy/trace/memory/enum never appear (§26 enforced) |
| Recommendation detail | multiplier chains are inherently technical | verbal scale only ("scaled back sharply"); numbers appear once, in the final amount |
| Stress-test | sliders invite spurious precision; results invite forecast-reading | chip-based decision types; ranges mandatory; caveat block non-dismissible; "not a forecast" in the header |
| History | timelines bloat | one entry per month, expandable; filters instead of tabs |
| Update ritual | re-entering numbers feels like chores | pre-filled diff editing (~6 edits), instant what-changed payoff, and the explicit promise ("2 minutes") kept honest |
| Global | is anything shown the founder can't act on? | remaining deliberate exception: Evidence layer is informational by design — justified as trust-building, capped at 3 items |

---

## 26. Product Language (terminology translation)

The only permitted mappings from engine vocabulary to founder surfaces. Accuracy over style; nothing renames itself into implied capability.

| Internal / technical term | Founder-facing term | Reason |
|---|---|---|
| trajectory / episode | *(hidden)*; in stress-tests: "simulated run" | "episode" is RL jargon; "run" is honest and plain |
| simulation (the engine) | "simulation" — kept, never softened to "projection/forecast" | the honesty contract depends on this word |
| shock (`COMPETITOR_SURGE` / `RATE_HIKE` / `RECESSION`) | "market stress": "new competitors flood in" / "interest-rate spike" / "downturn" | founder-recognisable events; labels stay truthful to the mechanics |
| agent (CFO/CMO/CPO) | "your advisory board": finance advisor, growth advisor, product advisor | role framing without implying human/AGI agency |
| oracle | "the strategist" (or unnamed: "your advisory board's analysis") | "oracle" implies prophecy — the exact wrong promise |
| oracle brief | "analysis" | plain |
| boardroom / merge / weights | "the board's plan"; weights → "the board's focus this month" (mix bar) | mechanism → outcome language |
| action modifier / modifier deltas | "strategic adjustment" ("scaled back marketing because risk is elevated") | says what happened, not how |
| decision trace | "why this plan" / "the board's reasoning" | it *is* the reasoning record |
| retrieval / memory / ChromaDB | "similar simulated situations" | provenance-honest; "memory" would imply your data or real firms |
| causal graph / Neo4j | "patterns from simulated market stress" | only surfaced with v4_causal + data present |
| confidence (0–1 scalar) | "how confident the board is: low / moderate / high" | scalar→band prevents fake precision |
| experiment / policy / seed / ablation | *(never surfaced)* | research controls |
| refresh / cadence / event trigger / cache_hit / reuse | "monthly review" / "re-analysed early because {specific reason}" / "consistent with your last analysis" | truthfully explains timing without machinery |
| Rule of 40 | not shown as a headline metric; where efficiency must be named: "growth + efficiency check" with the formula on tap | the engine's variant (MoM growth% + margin vs burn) is nonstandard; presenting it as the canonical Rule of 40 would mislead SaaS-literate founders — flagged as a deliberate deviation from "keep" |
| LTV:CAC | kept (founders know it), banded healthy/watch/unhealthy | standard founder vocabulary |
| runway (engine: cash ÷ salary-only burn) | "runway" but always computed from true total burn (§5.4) | same word, corrected substance |
| bankruptcy / `cause=Bankruptcy` | in stress-tests: "ran out of cash in N of 20 runs" | concrete, non-melodramatic |
| innovation_factor / product_quality | *(hidden)*; reflected only in advice copy ("product momentum") | unmeasurable internals presented as inputs would be data theater |
| `recommended_focus[]` | "focus this month" | direct |
| expected_outcome (GROWTH/STAGNATION/DECLINE) | "in simulation, the next 6–12 months most often looked like: growth / flat / decline" | preserves the horizon and the simulated framing |

---

## 27. End-to-End Founder Walkthrough (day in the life)

**Persona:** Maya, solo-founder of an 8-month-old B2B SaaS. $30k MRR, $220k in the bank, ~$48k/month total costs, $85/user/month, churn ~5.2%/month, ~7 competitors, one part-time hire plus herself.

**Tuesday, 21:40 — first visit.** Welcome screen sets the frame ("simulated scenarios, not real company data — a structured second opinion"). She signs up. Onboarding: company + age 8 months + "Crowded" (→ 9 competitors). Money: MRR 30,000; cash 220,000; costs 48,000 — the panel answers instantly: *Runway ≈ 12 months*. That number alone is worth the visit. Customers: price 85; churn — she knows 46% annually; the toggle converts to 5.0%/month. She adds new customers last month (41) and marketing spend ($6,000) when the helper says it unlocks acquisition-cost analysis (CAC ≈ $146; LTV derives to ≈ $1,700; LTV:CAC ≈ 11.6 — "healthy", with the derivation on hover). Total input time: a bit under five minutes.

**21:46 — first analysis.** Staged progress: "board deliberating (~half a minute)". It takes 24 seconds (one Ollama call; narratives off in MVP). Behind the scenes: her numbers became an `EnvState` (engine headcount = 48000/8000 = 6 units; age 8 → tenure decay ~0.62 already softening effective churn), the strategist returned risk HIGH / growth STABLE / innovation urgency HIGH, and the board's plan came back with marketing scaled ×0.5 by the risk modifier, R&D floored up by the deficit rule.

**21:47 — the advice.** Banner: *"Elevated risk — churn is eating growth and runway is about 12 months. This month: protect retention."* Cards: **Product ≈ $9k (priority)** — "your churn (5.0%/mo) is the board's top concern"; **Marketing ≈ $3k, performance channels** — "scaled back while risk is elevated; your acquisition cost is sustainable, so this is about cash discipline, not channel failure"; **Hiring: wait** — "revisit when runway exceeds two years"; **Pricing: hold**. She taps "Why this plan": focus bar Product-heavy; "risk elevated → marketing scaled back sharply; product investment protected." Evidence: two simulated cases ("a simulated company at a similar stage with high churn declined over the following 6 months"), each prefixed *from simulations, not real companies*, plus "first analysis — trends appear after your next update." Confidence: moderate. She accepts two actions, skips one ("I'm cutting marketing to $4k, not $3k" — recorded in V1; in MVP she just unticks). Elapsed since sign-up: ~9 minutes.

**Wednesday (V1 behaviour).** She stress-tests "raise price 15%": 20 paired runs, ~3 seconds. *Median MRR +4% at 12 months (range −9%…+17%); runway improved in 15 of 20 runs; churn +0.4pp median.* The caveat block reminds her simulated elasticity varies run to run. She saves it and asks the board to comment (one labelled ~20s call).

**September 3 — the return.** Home shows the staleness footer; she taps Update. Pre-filled diff: MRR 31,900 (+6.3%), cash 208,000, costs 47,000, churn 4.6% (−0.6pp), new customers 44, marketing 4,000. Submit → this is a scheduled monthly review (no panic trigger fired) → 20 seconds → new advice. **What changed:** "Churn improved 0.6 points — retention work may be landing (that's a pattern, not proof). Risk: still elevated (runway now 11 months as planned spend landed). Marketing: the board now suggests ≈$5k — efficiency pressure eased." The KPI arrows exist now; trend context activated (`history_points = 2`). History shows two months, her accepted plan, and her deviation. In February, her August month will mature and the timeline will annotate what actually followed — labelled as what happened next, never as credit.

**What Maya never saw:** a policy picker, a seed, an oracle mode, a similarity score, a Rule-of-40 headline, or a forecast.

---

## 28. Final Recommended Frontend Blueprint

**Stack:** keep Vite + React 18 + lucide + the existing `styles.css` token system; add react-router; keep FastAPI + SQLite (WAL) from the `backend` branch as the service chassis; polling (1.5s pattern) for job status; Ollama stays the only model dependency; Chroma per-company memories; Neo4j remains optional (Later-gated surfaces).

**Pages (7):** Onboarding · Home · Advice (+card detail) · Stress-test (V1) · History · My company · Settings.

**Endpoints to build (beyond the backend branch):** `POST /api/advise` (G1) · `POST/GET /api/companies/{id}/months` (G2) · `POST /api/stress_tests` (G6, V1) · auth (G10-minimal) — plus config extensions G4/G5/G7/G8 and the failure flag G3.

**Engine changes, all small and enumerated:** G3 failure flag · G4 company age · G5 burn override · G7 shock-schedule param · G8 memory scoping · G9 narrative wiring (port backend-branch fix) · G11 calibration scaling (V1, flagged).

**Build sequence:**
1. **Foundation:** merge/port the `backend` branch (restores API source lost from the working tree); add G3/G4/G5.
2. **MVP loop:** companies + months store (G2), advise-now (G1), Onboarding/Home/Advice/History-lite, honest states.
3. **V1 depth:** stress-tests (G6/G7/G12), narratives on (G9), multi-company (G10), calibration (G11), decision records.
4. **Later:** integrations, reminders, causal surfaces, constrained conversation.

**The five commitments that define the product** (each traceable to verified code): (1) analysis from ~8 founder-known numbers via the real Boardroom+Oracle spine; (2) every recommendation explains itself from the decision trace; (3) all simulation-derived evidence self-identifies as simulated; (4) failure degrades to rules-based advice, never to fake confidence; (5) the monthly ritual compounds — trends at update 2, the founder's own outcome-labelled history from month 7.

**Top risks and their mitigations:** small-company calibration (G11 + percent-first presentation + envelope copy) · LLM free-text hallucination (§15.4 guardrail; enum spine never depends on free text) · single-host Ollama latency/availability (async jobs, cached reuse semantics, rules-based fallback) · honesty drift in future copy (§26 is the contract; any new surface must map through it).

---

*End of specification. Sources: repository state on branches `frontend` (HEAD 67bb71c), `backend`, and `startup-multi` as of 2026-08-24; all file citations refer to the checked-out `frontend` branch unless marked backend-branch. Items marked UNVERIFIED: none load-bearing remain; `startup-multi` internals beyond §2 citations were only partially reviewed and nothing in this spec depends on them.*
