# Founder product — what to build next

Two tracks, split by blast radius. Track A cannot change research results; Track B
changes how the simulator behaves or what the corpora contain, so it lives on its own
branch and needs its own validation.

| Track | Branch | Touches | Research impact |
|---|---|---|---|
| A — correctness & product | `founder-integration` | product surfaces, gated engine flags | none |
| B — calibration & corpora | `founder-calibration` | simulator physics, memory + graph data | **invalidates comparability** |

Both branches exist and are pushed. `founder-calibration` currently equals
`founder-integration`; it is the fork point, not work in progress.

Baseline for everything below: `advice_audit.py` reports **2 violations across 6
profiles** on `founder-integration` (down from 9). Re-run it after every change.

---

## Track A — finish the product (safe)

### A1. Fix the two remaining audit violations

The `pre_revenue` profile ($1k MRR, $60k cash, $16k costs, 4 months runway) still fails:

- discretionary spend $1,250 exceeds MRR $1,000 (125%)
- `risk=LOW` at 4.0 months runway

The second is the more serious. Burn and runway now reach the prompt
(`include_burn_context`), and that was enough to move `cash_crisis` (1.1 months) to
`MEDIUM` — but 4 months still reads `LOW`. The model is being asked for a judgment call
it makes inconsistently.

**Do:** stop asking. Derive a risk floor from runway in code and take the worse of
(model risk, floor) — under 6 months cannot be `LOW`, under 3 cannot be below `HIGH`.
The brief keeps the model's read everywhere else.

- Files: `backend/advise_service.py` (post-process brief), or `boardroom/boardroom.py`
  if it should apply to all product surfaces
- Gate it like the others so research is untouched
- Effort: **S**
- Done when: no profile reports a risk/runway contradiction

For the spend violation, add a cash-aware cap: total discretionary spend must not push
post-plan runway below some floor (3 months is a defensible default). This is a bound,
not a floor, so it belongs beside the hiring guard in `_apply_sanity_bounds`.

### A2. Measure variance before trusting any of this

Every number in the audit is one run per profile. The direction is clear; the magnitudes
are not yet established.

```bash
venv\Scripts\python.exe advice_audit.py --repeat 5 --profile small_struggling
```

**Do:** run `--repeat 5` on at least `small_struggling`, `cash_crisis`, and
`pre_revenue`. Record the spread of risk level, marketing %, and hires.

- Effort: **S** (about 15 minutes of wall clock per profile)
- Done when: you can state "risk level is stable / flips X% of the time" as fact
- **This gates everything else.** If the same inputs produce `LOW` and `HIGH` across
  runs, the fix is determinism (seeded sampling, or majority vote over N briefs), not
  more prompt engineering.

### A3. Make `/api/advise` asynchronous

The spec (G1) calls for an async job with status. It is currently synchronous and blocks
a worker for the whole 30–90s call. Fine for one tester, not for a demo.

**Do:** reuse the pattern already in `backend/simulation_service.py` — `create_run` /
`start_run` / status polling — and give analyses the same `status` column they already
have in the `analyses` table (currently always written as `'complete'`).

- Files: `backend/advise_service.py`, `backend/main.py`, `frontend/src/api.js`,
  `frontend/src/pages/Analyzing.jsx`
- The Analyzing screen already has a progress UI and elapsed timer; it needs to poll
  rather than await
- Effort: **M**
- Done when: two browsers can run analyses concurrently without either blocking

### A4. Two honesty fixes in the UI

Both are small, and both are the kind of thing this product claims not to do.

1. **Macro assumptions surface as findings.** Watch-outs like "Low Unemployment Rate"
   derive from system defaults the founder never supplied. The confidence strip counts
   them as *estimated input*, but the bullet itself reads as a finding about their
   business. Either filter macro-derived bullets out of `key_risks`, or tag them
   inline as assumptions.
2. **"Three AI advisors" is one prompt.** CFO, CMO and CPO are three JSON sections of a
   single LLM response (`BatchedCausalProposalGenerator`), not three agents. The welcome
   copy implies more independence than exists. Either reword, or genuinely split the
   call — the second costs 3× latency for unclear benefit.

- Files: `frontend/src/pages/Welcome.jsx`, `frontend/src/copy.js`,
  `backend/advise_service.py`
- Effort: **S**

---

## Track B — calibration (new branch, research-affecting)

Everything here changes what the engine believes or what the corpora contain. Do not mix
it into `founder-integration`, and re-run the thesis comparison before claiming any
research result on top of it.

### B1. Scale-aware marketing physics — the root cause

`compute_new_mrr` draws the Hill curve's half-saturation point as
`gamma = random.uniform(15_000, 50_000)` **regardless of company size**. Response at
realistic founder spend, as a share of maximum potential:

| Monthly spend | ppc | brand |
|---|---|---|
| $1,000 | 8.0% | 0.1% |
| $3,000 | 16.3% | 1.1% |
| $15,000 | 38.7% | 24.5% |
| $50,000 | 60.2% | 73.2% |

A $12k-MRR founder spending $3k on brand reaches **1.1%** of potential. The simulator
believes their marketing does essentially nothing, so every policy on top of it correctly
concludes they should spend far more. This is physics, not the boardroom — the G11
scaling fixed the floors above it, but the curve underneath still says small spend is
wasted.

**Do:** make `gamma` and `beta` scale with company size rather than being absolute. The
cleanest form is to express them relative to MRR (a company's half-saturation point is
some multiple of its own revenue, not a fixed dollar figure), with the current values as
the $50k-MRR case so the calibration point is preserved.

- Files: `env/business_logic.py` (`compute_new_mrr`), `config/sim_config.py`
- Keep it behind an `initial_config` key so existing runs reproduce exactly
- Effort: **M** code, **L** validation
- Done when: a $12k-MRR company spending 20% of MRR sees a plausible response, and the
  $50k-MRR case reproduces current behaviour within noise

### B2. Regenerate the memory corpus at founder scale

Current corpus: median **$973,808 MRR**, minimum $14,672, **0.4%** below $25k. A $12k
founder sits below the entire corpus — retrieval cannot find a single comparable
situation. Every "similar simulated situation" shown to them is roughly 80× their size.

**Do:** `run_simulation()` already accepts `environment_config`, so this is a parameter
change, not new machinery. Run a batch with founder-scale initial conditions
(`initial_mrr` 5k–50k, `initial_cash` 30k–300k, `initial_headcount` 2–8), pointed at a
fresh Chroma path via `OracleMemoryStore(chroma_path=...)`.

- Files: a new script beside `run_causal_hetero.py`; no engine changes
- **Sequence after B1.** Regeneration is an overnight job — memories are written by
  `Oracle.end_episode()` after full episodes, and oracle refreshes are LLM calls. Time a
  single episode before launching a batch.
- Keep the existing `chroma_db/` untouched; write to a third path and switch
  `FOUNDER_CHROMA_PATH`
- Effort: **M** code, **L** compute
- Done when: the p10–p90 MRR band of the founder corpus brackets $5k–$100k

### B3. Populate the causal graph at SEED tier

`get_mrr_tier` classifies anything under $100k MRR as `SEED`. The graph holds **zero**
SEED-tier shock records and no `StateSnapshot` below **$270k MRR**, so
`query_similar_shocks` returns nothing for the target user. The founder currently sees
only `_ensure_seed_edges()` output — a hand-authored prior, now correctly labelled as an
assumption rather than evidence.

**Do:** the B2 batch writes graph records automatically when run under
`oracle_v4_causal` with shocks enabled. Run it against a **separate Neo4j database or
instance**, not the research graph — `neo4j_backup.py dump` first regardless.

- Effort: **S** on top of B2
- Done when: `MATCH (n:Shock) WHERE n.mrr_tier = 'SEED' RETURN count(*)` is non-zero, and
  the Evidence panel shows *observed* lines (not only *assumption* lines) for a
  small-company profile

### B4. External calibration — the part code cannot fix

Everything above makes the engine internally consistent. None of it makes it *right*,
because no constant in the stack was estimated from real companies:

- `gamma`, `beta`, `alpha` — marketing response, hand-chosen ranges
- `elasticity = random.uniform(-0.9, -0.2)` — redrawn every step
- `tenure_decay = exp(-0.15 × …)` — invented
- expansion MRR flat at `mrr * 0.02` — invented

The memory corpus and causal graph are derived *from* this simulator, so they amplify its
assumptions rather than correcting them. More simulated episodes add confidence, never
information about real startups.

**Sources, cheapest first:**

1. **Published SaaS benchmarks** (ChartMogul, OpenView, SaaS Capital) — churn, CAC
   payback, NDR and growth banded by ARR. Exactly the shape needed to fit `gamma`/`beta`
   per band and replace blended churn. Check licensing before redistributing; fitting
   constants you ship is normally fine, republishing their tables is not.
2. **First-party outcomes** — `company_months` and `decisions` already exist (G2). Every
   founder who enters numbers monthly and reports results yields a real
   (state → action → outcome) tuple at exactly the scale that is missing. Slowest to
   accumulate, most valuable, uniquely yours. Instrument the capture now even though the
   payoff is months away.
3. **Public outcome datasets** (Crunchbase-style) — useful for survival base rates and
   risk calibration, weak on monthly operating detail.

- Effort: **L**, and partly not a coding task
- Done when: at least the churn-by-ARR-band and CAC-payback constants trace to a cited
  source rather than a literal in `business_logic.py`

---

## Suggested order

1. **A2** first — variance. If advice is unstable across identical runs, everything else
   is measuring noise.
2. **A1** — closes the audit to zero violations.
3. **A3 / A4** in parallel — independent of each other and of Track B.
4. **B1**, then **B2 + B3** together (regeneration is expensive; do it once, after the
   physics change).
5. **B4** — start sourcing while B1–B3 run.

## Guardrails

- Re-run `advice_audit.py` after every change; it is the regression test for advice.
- Re-run `pytest tests/ -q` — 36 tests, currently all passing.
- `neo4j_backup.py dump` before anything that could write to the graph, and
  `neo4j_backup.py verify` after.
- Every engine change stays behind a default-off flag, as `include_burn_context`,
  `hiring_runway_guard_months` and `scale_absolutes` already do. Research runs must
  reproduce byte-identically until you deliberately choose otherwise on
  `founder-calibration`.
