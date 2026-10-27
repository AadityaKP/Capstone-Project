# OEFA loop + multi-month HITL cycle — plan

Branch: `round2`. Written 2026-09-22. Supersedes the single-month `/api/advise`
surface as the product's primary flow.

Everything in §0–§7 was verified against the code on this branch, not inferred
from the spec or from screenshots. Every file:line reference is real; appendix A
says how to re-check any claim.

**Read §2 before starting any other phase.** Three of the four OEFA steps are
no-ops against the code as it stands, and two of them fail silently. Building
Phases 1B–3 on top of them produces a product that demonstrates a loop which is
not running.

---

## Review feedback this plan answers

| What the review asked for | Where it lands |
|---|---|
| Get the old frontend back, without the EDGAR filings | §1 Phase 0 |
| **The internal reasoning of the agents needs to change** | §3 Phase 1A — *absent from the previous draft* |
| It shows only one month; it needs to go further | §4–§5 Phases 1B, 2 |
| The agent should follow Observe → Execute → Feedback → Adapt | §4 Phase 1B |
| Show 3–4 months at once, with the projected state of the company | §5 Phase 2 |
| The user says whether the improvements were actually made — the HITL | §6 Phase 3 |

Two positions from the working notes are dropped because the review overrides
them. They are recorded here so the reasoning is not lost and re-litigated:

- **Ordering.** The notes argued for HITL (Phase 3) before the multi-month view
  (Phase 2), on the grounds that a single-month closed loop carries the novel
  contribution with none of the compounding-error risk. The review states the
  order explicitly. Phase 2 comes first.
- **Layout.** The notes argued against side-by-side month columns in favour of a
  vertical timeline. The review asked for 3–4 months *simultaneously*, and a
  timeline you scroll is not that. Columns it is (§5); the timeline returns
  below them as the history half, where it does not compete.

---

## 0. Where the code actually stands

`POST /api/advise` → `advise_service.run_analysis()` builds one `EnvState` from
the founder's numbers, replays their history through `Oracle.observe_state` for
trend context, calls `Boardroom.decide()` once, and returns brief + trace. It
never steps the environment, never calls `Oracle.end_episode`, never calls
`Oracle.write_causal_outcome`. The product today runs **Observe → Execute and
stops.**

`POST /api/whatif` → `whatif_service.run_whatif()` does roll 12 months forward
over 3 policies × 50 seeds, but is explicitly LLM-free and Oracle-free: the
board's month-1 action is held constant for all 12 months. So the multi-month
view that exists is agent-free, and the agent view that exists is single-month.
This plan joins the two.

The Feedback/Adapt machinery is written and unwired from the product path:
`Oracle.observe_state`, `_mature_pending_memories` (`MEMORY_HORIZON_MONTHS = 6`),
`classify_realized_outcome`, `Oracle.end_episode`, `Oracle.write_causal_outcome`,
`WeightAdapter.adjust_weights`, `Boardroom._get_oracle_refresh_reason` /
`_has_event_trigger`.

But "unwired" understates it in three places. The following are not integration
gaps; they are functions that return without doing anything:

| Step | Status in the product path today |
|---|---|
| **O**bserve | Works. `observe_state` is called on every `decide()` (boardroom.py:168). |
| **E**xecute | Works. |
| **F**eedback | Dead. `write_causal_outcome` returns immediately unless Neo4j is up *and* mode is `oracle_v4_causal` (oracle.py:245). Default profile is `review2`/`oracle_v3`. |
| **A**dapt | Half-dead. Weights adapt. Episodic memory does not: retrieval is scoped to a per-request UUID, so no analysis can read what any previous analysis wrote. |

---

## 1. Phase 0 — De-EDGAR the frontend

No git revert. The current frontend is the base; the EDGAR surfaces come out and
the remaining vocabulary goes general.

- Delete `frontend/src/pages/Dataset.jsx`, `pages/Run.jsx`, `pages/Review.jsx`.
- `App.jsx` — drop the three `NAV` entries (`review`, `dataset`, `run`), the
  `REVIEW_PAGES` array and every guard referencing it, the three `TITLES` keys,
  the three imports, the three `pages` entries. The route guard collapses back to
  "no company → welcome/onboarding".
- `api.js` — drop `reviewMeta`, `reviewCompare`, `reviewPanel`,
  `reviewBacktestCompanies`, `reviewBacktestRun`.
- `whatif.jsx` — remove the `wi-edgar-band` / `wi-edgar-median` overlay and its
  legend row. If a benchmark band is still wanted on the projection chart,
  re-source it from `calibration/bands.json` (the SaaS Capital medians the advise
  path already cites) so the comparison stays general rather than EDGAR-specific.
- `backend/main.py` — remove the five `/api/review/*` routes and the
  `review_service` import block.
- `styles.css` — drop the `.rv-*`, `.ds-*` and `.wi-edgar-*` blocks (23 matching
  lines).
- **Also delete `policyLabel` from `copy.js`** — it goes dead with Review and
  Run, and was missed in the previous draft.
- `npm run build` in `frontend/` so `frontend/dist` matches.

Leave `backend/review_service.py`, `data/edgar_ratios.csv` and everything under
`validation/` on disk: the thesis results reproduce from them, and deleting them
would break the validation package for no gain. **EDGAR stops being a product
surface; it stays a research asset.**

Coupling was checked before writing this: nothing outside the three deleted pages
imports the five `review*` API functions, and `FanChart` is imported only by
Review and Run (both going) besides its own module. Phase 0 is safe to do first
and in one commit.

---

## 2. Phase 0.5 — Make the loop's plumbing real *(blocking)*

Three fixes. Without them the rest of the plan builds a UI for a loop that is not
turning, and two of the three fail silently rather than erroring.

### 2.1 Founder memory is write-only

`sim_profile.get_oracle_kwargs()` constructs the store with
`run_id=str(uuid.uuid4())` (sim_profile.py:136), and retrieval filters on it:

```python
# oracle/memory.py:262
where={"run_id": self.run_id},
```

A fresh UUID per request means an analysis can only retrieve memories written
inside that same request. Nothing a previous analysis wrote is ever reachable.
The `review2` profile is no better — `Oracle.__init__` also defaults `run_id` to
a fresh UUID.

**Fix:** a stable per-company scope, `run_id = f"company:{company_id}"`. One
line. It unblocks Adapt and the whole of Phase 3.

**Verify:** run two analyses for the same company; the second's trace must show
`memory_count > 0` with memories the first wrote. This is a five-minute check
and it de-risks everything downstream.

### 2.2 The causal write is a no-op in the default configuration

`Oracle.write_causal_outcome` returns immediately unless `graph_store.enabled`
(oracle.py:245), and the graph store only exists when mode is `oracle_v4_causal`
**and** Neo4j answers on `bolt://localhost:7687`. The default `SIM_PROFILE` is
`review2` → `oracle_v3` → no graph store at all.

**Fix:** extend `/api/health` with `graph_store_enabled` and `memory_scope`, and
surface both in Settings as a capability panel. The failure mode here is silence,
not an error, and a demo that claims a learning loop while the graph is
unreachable is the worst possible outcome of this project.

### 2.3 Simulated and observed evidence must not share edges

Phase 1B's Feedback step writes `env.step()` deltas — the model's own physics —
through `write_causal_outcome`. Phase 3 writes the founder's real outcomes
through the same call. Both land on the same `MAY_CAUSE` relationship and the
same confidence counter (graph_store.py:434), and both are retrieved back into
the prompt indistinguishably.

Four simulated months per cycle against one real month per month means the
simulator's opinion permanently outweighs reality, and the board ends up citing
its own physics back to itself as evidence. `physics_v2` graded that simulator B.

**Fix:** add `source: "sim" | "observed"` to the edge, and either namespace them
or weight `observed` higher at retrieval. Decide the weighting explicitly and
write it down — it is a thesis claim, not a default.

### 2.4 Decide state ownership now

The store is `localStorage`-only (store.jsx:15) while the `company_months` table
sits unused in SQLite. Cycles server-side plus months in the browser means the
server cannot answer *"what was the actual state at month N"* — which Phase 3's
"the next cycle starts from the actual state" requires.

Pick one in this phase, not on discovery in Phase 3:

- **(a)** Cycles carry their own actuals in `cycle_feedback`; the browser stays
  the founder's record. Cheaper, and keeps the local-first property.
- **(b)** Months move server-side into the table that already exists. More work,
  but it is the only option under which a second device, a reset browser, or a
  reviewer's machine sees the same history.

Recommendation: **(a)** for this milestone, with the schema shaped so (b) is a
migration and not a rewrite.

---

## 3. Phase 1A — Change what the agents reason about

*This is the review point the previous draft did not address at all, and it turns
out to gate the Feedback step and the entire HITL phase.*

### 3.1 What the agents do today

They do not reason. They **rationalize**.

```python
# agents/proposal_agents.py:63
def propose(self, state: EnvState) -> Proposal:
    action = self.act(state)                       # heuristic picks the action

    proposal = Proposal(
        agent="CFO",
        objective="Preserve runway and improve efficiency",
        actions=action,
        expected_impact="Lower burn, improved survival probability",   # constant
        risks=["Slower growth"],                                        # constant
        confidence=0.8,                                                 # constant
    )

    reasoning = self._get_rationale(                # LLM writes the explanation
        state, "You are the CFO of a SaaS startup. Be concise.",
        f"... The proposed action is: {action}. "
        f"In 2 sentences, explain the strategic rationale for this decision ...",
    )
```

The heuristic chooses; the LLM is then asked to justify a decision already made.
`expected_impact`, `risks` and `confidence` are hardcoded strings and a constant.
The agent sees exactly one state: no knowledge of what it proposed last month,
and no knowledge of whether it worked.

The founder profile's `BatchedCausalProposalGenerator` is better — there the
model does choose the action and emits its own `expected_impact` — but that field
is still free text (`expected_impact: str` on the Proposal schema).

### 3.2 The three changes, in order of how much the loop depends on them

**(a) `expected_impact` must become falsifiable.** A slogan cannot be checked
against an outcome, so "did the improvement happen?" has nothing to compare
against, and both Feedback and HITL have nothing to score. Add:

```python
expected_delta: dict[str, float]   # per-KPI, signed, with a horizon
# e.g. {"mrr_pct": +4.0, "churn_pp": -0.3, "horizon_months": 2}
```

Everything in Phases 1B, 3 and 4 depends on this one field. It is what turns
"the board gave advice" into "the board made a prediction", and a
predict-then-check contract is a thesis result rather than plumbing.

**(b) Agents must see their own track record.** `previous_final_action` is
already in the decision trace (boardroom.py:357) but only for display. Feed each
agent what it proposed last month, what it predicted, and what actually happened.
That is what "the internal reasoning needs to change" means concretely: a CFO who
raised marketing last month and watched CAC worsen should reason differently this
month than one seeing the state cold.

**(c) Decide whether the LLM chooses or explains.** The rule-based path
rationalizes; the causal path decides. Running both and calling each "the board"
invites exactly this review comment again. Pick one for the product path, state
it in the report, and keep the other as a research arm.

### 3.3 What not to break

`agents/baseline_agents.py` and the legacy flags are frozen — the recorded
research runs must stay byte-identical. `expected_delta` is additive with a
default of `None`; nothing in the Review 2 arms may start populating it.

---

## 4. Phase 1B — The loop engine

New `backend/cycle_service.py`. One cycle = H months, H configurable, default 4.
Per month, in order:

**Observe** — `oracle.observe_state(state)` appends a snapshot and recomputes
`TrendContext`; `oracle.get_context()` retrieves similar past months;
`get_causal_graph_context()` supplies graph evidence under v4. Everything
retrieved is recorded for the trace, because the retrieval is the explanation.

**Execute** — `boardroom.decide(state)` → action + full decision trace
(proposals, score vectors, applied weights, brief, and now `expected_delta` per
proposal). The founder-product guards still apply: `_apply_spend_ceiling` and the
24-month hiring-runway guard.

**Feedback** — `env.step(action)` produces the next state. Compute `kpi_delta`
(MRR, cash, churn, runway, spend-ratio) against the month's opening state, score
it against the month's `expected_delta`, and call
`oracle.write_causal_outcome(action, kpi_delta, stress_node=..., source="sim")`.
This is the step the product has never had: the board finds out what its own
decision did, and how far off its own prediction was.

**Adapt** — the delta shapes the next month. `WeightAdapter.adjust_weights`
re-weights from the brief with 70/30 smoothing; `_mature_pending_memories` fires
as `global_month` advances; `_has_event_trigger` fires on runway < 12 months, MRR
down 15%, churn up 1.5pp, confidence down 15, unemployment up 2. Surface an
explicit `adapt` block per month: which weights moved, whether the brief was
refreshed and why, which memory or causal edge changed the proposal.

### 4.1 Four corrections to the previous draft

**Do not call `end_episode()` at cycle end.** It force-matures every pending
memory against the latest snapshot regardless of horizon (oracle.py:361), so a
4-month cycle writes memories labelled by `classify_realized_outcome`'s 6-month
±10% rule from 4 months of evidence — into the same store the thesis reads. Let
memories mature from real HITL months instead, which genuinely do arrive a month
apart.

**The `oracle_frequency` knob is the opposite of what the draft said.**
`FOUNDER_ORACLE_FREQUENCY = 1` (sim_profile.py:50) makes
`months_elapsed % 1 == 0` always true, so *cadence* fires every month and
`_has_event_trigger` never gets a turn. To let events earn the refresh, set it to
**0** — `_get_oracle_refresh_reason` guards the cadence branch with
`self.oracle_frequency > 0`, leaving `initial` + `event`.

**Persist the Oracle's per-company state** (`global_month`, `pending_memories`)
or accept that the pending queue dies with the request. Today `Oracle` is
constructed fresh per call (advise_service.py:271) with `global_month = 0`
(oracle.py:77), so "memories written this cycle mature next cycle" is false. It
is a table; decide in Phase 0.5, not in Phase 3.

**Brief freshness is already traced — surface it.** `brief_source` is in the
decision trace (boardroom.py:347) with values `llm` / `cache_hit` / `reuse` /
`none`. This matters because the cache brackets MRR at `int(mrr / 50_000)`
(oracle.py:337) and the memory signature is always `"none"` outside `oracle_v3`,
so a stable company will reuse month 1's brief for months 2–4. That is the cache
doing its job, but it means "the board deliberates every month" is not literally
true. Showing `fresh` vs `reused` per month turns an awkward fact into an
explainability artifact.

### 4.2 Shape

Asynchronous, following `simulation_service.start_run`:

- `POST /api/cycles` → `202 {cycle_id}`, work on a background thread
- `GET /api/cycles/{id}` → status plus the months finished so far, so the UI
  renders each month as it lands instead of blocking on all four
- new `cycles` / `cycle_months` tables in `backend/database.py`

Per-month response:

```json
{
  "month_index": 1,
  "observe":  { "memories": [], "trend": {}, "graph": {} },
  "execute":  { "action": {}, "brief": {}, "proposals": [],
                "weights": {}, "expected_delta": {},
                "brief_source": "llm" },
  "feedback": { "state_after": {}, "kpi_delta": {},
                "prediction_error": {} },
  "adapt":    { "what_changed": [], "refresh_reason": "event" }
}
```

plus a cycle summary: projected state at month H, survival, runway at horizon.

One `Boardroom.decide()` is 20–90s on local Ollama; four is 1.5–6 minutes, well
past the client's 120s budget (api.js:16). Streaming months as they land covers
most of it, and the brief cache will often collapse the real LLM call count below
H — see the freshness note above.

---

## 5. Phase 2 — The 3–4 month view

> **Superseded (2026-09-23)** by `docs/ui_simplification_plan.md` and
> decision 12: the Plan page described here was built and then folded into
> This month (the Outlook chart and the "next 3 months" list) with the OEFA
> beats under Why this plan → "How the board got here". Kept as history.

New `frontend/src/pages/Cycle.jsx`, nav label **"Plan"**.

### 5.1 The month strip — the review's "simultaneously"

Four month columns across the top of the page, all visible without scrolling or
clicking, each filling in as its month completes: month label, the board's
action, the projected state after it (MRR, cash, runway, churn).

Two rules on how they behave:

- **Month 1 is visually dominant.** Full weight, decision controls live. Months
  2–H lighter, each with a `projection` chip. Only month 1 is a decision the
  founder can act on this week; four identical columns claim otherwise, and
  months 2–H are the model's physics compounded, which `physics_v2` graded B.
- **Each column carries its own OEFA strip** — Observed / Decided / Expected /
  Changed — as one component used everywhere, including retroactively on the
  single-month Advice page. One vocabulary across the product, and it is the
  Module 6 explainability payload in a form that screenshots.

### 5.2 Chart

One trajectory chart across the H months below the strip, reusing the
`whatif.jsx` primitives minus the EDGAR band.

Do **not** reuse the FanChart's solid → dashed → terminal-marker grammar for
projection confidence. In that module it already means *survival* (whatif.jsx
header comment), and overloading it makes both unreadable. Projection uncertainty
gets the existing IQR band; nothing else.

### 5.3 The timeline, below

The history half: `History.jsx`'s existing vertical rail, running forward past a
"now" marker. Its payoff arrives in Phase 3 — when a month closes, the projected
card is replaced **in place** by the actual card, so the prediction error is the
diff in the same slot, in a component the founder already reads.

### 5.4 Navigation

Fold Advice into Plan. "Advice" and "Plan" side by side are indistinguishable to
a founder. `Plan` becomes the nav entry; `/advice/:id` stays as a detail route
reached by clicking a month in the strip or the timeline. Nav returns to five
items — Home, Plan, History, My company, Settings — and it matches "the cycle
becomes the primary surface".

### 5.5 Demo mode

`SAMPLE` (sample.js) has three months, two analyses and decisions, but no cycle.
The Plan page must either ship canned cycle data in `sample.js` or be hidden in
demo mode. Ship the canned data: the sample company is how a reviewer sees the
product without Ollama running, and a "Plan" tab that is empty in the one mode
that always works is worse than no tab. See §8.

---

## 6. Phase 3 — HITL: did the improvements actually happen

The founder acts on month 1 in the real world and comes back a month later.

### 6.1 One screen, not two

**Merge the close-the-month step into `UpdateRitual`** (Company.jsx:96) rather
than adding a step beside it. That flow is already pre-filled diff editing with
instant what-changed pills and a straight path to re-analysis. A separate close
step gives the founder two places to type the same numbers, and they will
diverge.

Add a section above the number grid:

> **Last month the board asked for these three things — what happened?**
> `did` / `partly` / `didn't` per action, plus an optional note.

then the numbers, then one submit. The `SET_DECISION` reducer (store.jsx) already
carries `state` and `note` per domain, so the client shape exists.

Stored as `{cycle_id, month_index, per_action:[{action_key, done, note}],
actuals:{mrr, cash, churn, costs}}`; the backend gains
`POST /api/cycles/{id}/feedback` and a `cycle_feedback` table.

### 6.2 What the feedback does — the whole point

1. **Prediction error.** Predicted vs actual per KPI, computed against Phase
   1A's `expected_delta` and shown. Not behind a disclosure — see §6.3.
2. **Actions marked done whose KPI moved as predicted** →
   `write_causal_outcome(..., source="observed")`, strengthening the edge.
3. **Actions marked didn't** → the month is not evidence about that action.
   Write the state observation only, no causal edge. Without the per-action flag,
   every skipped recommendation would write a false causal link and the memory
   would be actively wrong. Say this to the founder too — *"you didn't do this,
   so we're not counting this month as evidence about it"* reads as candour, not
   as a disclaimer.
4. **`oracle.observe_state(actual_state)`** puts the real month into episodic
   memory, where `classify_realized_outcome` grades it six months later through
   the existing maturation path.
5. **The next cycle starts from the actual state**, not the projected one,
   carrying the previous cycle's prediction error into the prompt context.

That closes both architecture loops: M6 → M1 (approved actions re-enter the
simulation) and M6 → M3/M4 (outcomes refine memory and the causal graph).

### 6.3 Where prediction error lives

On **Home's "What changed" panel** (Home.jsx, section 4). "We projected $47k, you
did $44k — we were 6% high" belongs in the same list as "Revenue grew 4%",
because that panel is where the founder already looks. With `expected_delta` it
is a real number rather than a narrative.

### 6.4 Two corrections to the previous draft

**"Done, and it worked" does not produce a `CONFIRMED_CAUSE` edge.** Promotion
requires `r.confidence >= 0.85 AND r.positive_observations >= 3`
(graph_store.py:460), and the increment is +0.05 from a 0.6 base — five positive
observations minimum. And one `write_causal_outcome` call writes one edge **per
KPI in the delta**, not one edge total (graph_store.py:415, the
`for metric, delta in kpi_delta.items()` loop).

**"Founder-confirmed → confidence raised" does not work as written.**
`base_confidence` reaches the query only through
`coalesce(r.confidence, $base_confidence)`, so it applies to a *new* edge and has
no effect on an existing one. Making confirmed evidence count more needs a new
parameter on `write_action_outcome` — the increment itself, or a `source`
weighting from §2.3.

---

## 7. Phase 4 — Verification

`tests/test_cycle_api.py`:

- a cycle produces exactly H months
- a cycle with `use_oracle=False` runs deterministically end to end
- same seed → same actions → same projected states
- feedback that is all-"didn't" writes **zero** causal edges
- feedback that is "done" with a matching delta writes **one edge per KPI in the
  delta and promotes nothing on the first observation** (see §6.4)
- `expected_delta` is present on every proposal in the product path and absent
  in the Review 2 arms

Plus:

- **Prediction-error harness.** Replay historical months as if they were HITL
  closes and report calibration of `expected_delta` against actuals. A thesis
  number, computable only because of Phase 1A.
- **Loop-is-live check.** Assert `graph_store_enabled`, and assert a second
  analysis for the same company retrieves memories the first wrote. Both current
  failure modes are silent.
- **Per-month latency** in the run log, so the H× LLM cost is visible rather than
  a surprise.
- **Manual:** run a cycle with Ollama stopped and confirm the UI shows the honest
  `llm_ok:false` state rather than a fabricated plan.

One honesty fix comes free with this phase: Analyzing.jsx:121 promises *"You can
leave this page; we'll keep your seat"*, which is false today — the run lives in
a page-local effect and navigating away kills it. The async cycle pattern makes
it true.

---

## 8. Demo workflow

### 8.1 The design principle

**The demo must open on a company that already has history.**

This is the single most important decision in the whole showcase. A first cycle
has no memory to retrieve, no prediction error to report and nothing learned —
it demonstrates a product with amnesia. Everything this project is *for* is only
visible on cycle two and later. So the live demo starts in the middle of the
story, and the first thing on screen is the system being graded on a claim it
made last month.

The second principle follows from it: **lead with the prediction error, do not
bury it.** A demo that hides how wrong the board was will draw exactly the review
comment this plan exists to answer. A board that says "we were 6% high last month
and here is what we changed because of it" is a stronger artifact than one that
projects four confident months and is never checked.

### 8.2 What the demo must not do

| Don't | Because |
|---|---|
| Onboard a new company live and run a cold first cycle | No memory, no error, no learning — the loop looks like a single-shot recommender |
| Open with four live LLM months | 1.5–6 minutes of dead air before the first pixel of value |
| Claim the learning loop while Neo4j is down | `write_causal_outcome` returns silently; the Adapt story is a lie, and §2.2's health panel will say so on screen |
| Demo in sample mode while claiming the loop works | Sample data is canned by design; be explicit about which mode is on screen |
| Present months 2–4 as forecast | They are the B-grade simulator compounded; the `projection` chip exists for this reason |

### 8.3 Setup (T-60 minutes, before anyone is watching)

1. **Stack up, prod mode.** `.\start.ps1 -Prod` — one port, one URL, no stale
   bundle. Confirm the banner reports the advisor mode.
2. **Confirm the loop is live.** `GET /api/health` must show
   `graph_store_enabled: true` and a `memory_scope` that is a company key, not a
   UUID (§2.2). If Neo4j is down, either fix it or change what you claim.
3. **Back up the causal graph** (`neo4j_backup.py`) — the demo writes to it, and
   the research graph must survive.
4. **Seed the demo company.** A script that creates the company, six months of
   real-looking history, and **two completed cycles with HITL feedback already
   closed** — including at least one action marked `didn't`, so §6.2 item 3 has
   something to show. This script is a Phase 3 work item, not an afterthought.
5. **Pre-warm the brief cache.** Run the cycle you will show once, end to end, so
   the state buckets are warm (oracle.py:337). The live run in Act 5 will still
   make real calls where the state has moved.
6. **Have the sample company ready as a fallback** — it needs no backend and no
   Ollama, and if the stack dies mid-demo it is the difference between a pause
   and a collapse.

### 8.4 Run of show (~12 minutes)

**Act 1 — Home. "We were wrong by 6%, and here's what we did about it." (60s)**

Open on Home, not Plan. The "What changed" panel (§6.3) leads with last month's
prediction error against the actual numbers the founder entered. One sentence
frames the whole demo: *this system makes a prediction, finds out whether it was
right, and changes.* Nothing else on the first screen.

**Act 2 — This month → Outlook → Why this plan. (2 min)**

*(Re-scripted 2026-09-23, decision 12; the original walked the Plan page's
month strip.)* Stay on This month: the plan cards are the decision for this
month; scroll to the Outlook and switch the metric — the founder's own
number is the opening point, everything to its right sits on the shaded
"projected" band, and the closed month is the marked point labelled "you".
Open "Next 3 months of the plan". Then "Why this plan" → "How the board got
here", where month 1's strip is already open: walk the four beats in order —
what it observed, what it decided, what it expects (the numeric
`expected_delta`, not a slogan), what changed as a result, labelled as a
model consistency check when it is the simulator scoring itself and as
"your numbers" when it is the close. The loop lines at the top of that
section (fresh vs reused reads, written back as simulated evidence) answer
the brief-freshness question (§4.1) before anyone asks.

**Act 3 — Close the month. The HITL moment. (3 min)**

The heart of the demo. Open the merged update ritual (§6.1). Three actions from
last month, and mark them honestly: one `did`, one `partly`, one `didn't`. Enter
the actual numbers. Submit.

Say the quiet part out loud on the `didn't`: *the board is not going to learn
anything from this one, because it never happened.* That single design choice —
refusing to write a causal edge for an action nobody took — is the most
defensible thing in the system and takes ten seconds to explain.

**Act 4 — What the board learned. (2 min)**

The learning surface: weights that moved and why, the memory that matured with
its realized outcome, the causal edge that gained an observation, the edge that
did *not* because the action was skipped. Be straight about §6.4 — one confirmed
month does not promote an edge to `CONFIRMED_CAUSE`; it takes several. A system
that says "this is one observation, not proof" reads as competent.

**Act 5 — Run a live cycle. (2 min)**

*(Re-scripted 2026-09-23, decision 12.)* Press Re-run on This month and stay
there: the plan section becomes "Your board is planning" with the staged
list and "Month k of 4 · m:ss"; the plan cards appear when month 1 lands,
and the Outlook line and the "next 3 months" list fill in one month at a
time as the background thread finishes them (§4.2). Navigate to History and
back mid-cycle to show the run does not live on the page. This is the moment
the audience sees real computation rather than a rendered artifact. Keep
talking over it — the elapsed counter is honest and worth narrating rather
than apologising for.

**Act 6 — Break it on purpose. (1 min)**

Stop Ollama and re-run. The UI shows `llm_ok:false` and the rules-only state
(runbook §4.2), and says so plainly instead of inventing a plan. Closing on a
deliberately triggered failure state is the strongest possible trust close, and
it is the one thing most demos of this kind cannot do.

### 8.5 Contingencies

| If | Then |
|---|---|
| Ollama dies unprompted | You were going to do Act 6 anyway — do it now and continue |
| Neo4j is unreachable at setup | Drop Act 4's causal half; keep weights + memory. Say which half is missing rather than narrating it as if it ran |
| The live cycle in Act 5 stalls | Cut to the pre-warmed cycle from setup; the streaming behaviour has already been shown in Act 2 |
| The whole stack dies | Sample company from the welcome screen — labelled, canned, needs nothing running |
| Someone asks "is month 4 a forecast?" | No. It is the simulator compounded four times, the simulator is graded B, and that is why prediction error is on the home screen |

### 8.6 Have ready, do not show unprompted

- The prediction-error calibration numbers from §7's harness — the answer to
  "how do you know any of this works"
- The Phase 0.5 health panel — the answer to "how do you know the loop is
  actually running"
- `validation/` and the EDGAR research assets — removed from the product surface
  in Phase 0, still the backing for the thesis results, and worth being able to
  produce in thirty seconds if asked why they went away

---

## 9. Risks worth deciding on before Phase 1B

**Latency.** Four live deliberations is the honest reading of "deliberate every
month" and it is minutes per cycle. Streaming months as they land covers most of
it; the brief cache covers more, at the cost of months 2–4 not being fresh
deliberations (§4.1). Surface the trade-off rather than hiding it.

**Compounding simulation error.** Month 4's state is the model's physics
compounded four times, and `physics_v2` graded the simulator B rather than A.
Mitigated by the `projection` chip and by showing prediction error once feedback
arrives — but it is a real limitation and belongs in the report, not only in the
UI.

**Memory horizon.** `MEMORY_HORIZON_MONTHS = 6` against a 4-month cycle means
memories written during a cycle mature during the next one — *if* §2.4 persists
the pending queue. Within-cycle Adapt runs off weights and the causal graph, not
off matured episodic memory. State this explicitly in the report rather than
discovering it during the demo.

**Evidence provenance.** Until §2.3 lands, every simulated month writes into the
same causal edges as real founder outcomes, at four times the rate. This is the
risk with the longest tail, because it silently degrades the graph the thesis
reports on.

---

## Appendix A — how to re-check the claims in this document

| Claim | Check |
|---|---|
| Founder memory is write-only | `grep -n "run_id=str(uuid.uuid4())" backend/sim_profile.py` and `grep -n 'where={"run_id"' oracle/memory.py` |
| Causal write is a no-op by default | `sed -n 245,250p oracle/oracle.py`; `get_profile()` defaults to `review2` (sim_profile.py:69) |
| `CONFIRMED_CAUSE` needs >= 3 positive observations | `sed -n 458,468p oracle/graph_store.py` |
| One call writes one edge per KPI | `sed -n 415,420p oracle/graph_store.py` |
| Cadence fires every month under the founder profile | `grep -n "FOUNDER_ORACLE_FREQUENCY = 1" backend/sim_profile.py`; `sed -n 590,600p boardroom/boardroom.py` |
| Agents rationalize rather than reason | `sed -n 61,88p agents/proposal_agents.py` |
| `expected_impact` is a free-text string | `grep -n "expected_impact" boardroom/schemas.py` |
| Oracle is constructed fresh per request | `grep -n "Oracle(" backend/advise_service.py`; `grep -n "self.global_month = 0" oracle/oracle.py` |
| `brief_source` is already traced | `grep -n "brief_source" boardroom/boardroom.py` |
| Phase 0 has no reverse coupling | `grep -rn "FanChart\|reviewMeta\|reviewCompare\|reviewPanel\|reviewBacktest" frontend/src/` |
| Trend context sees only 5 snapshots | `grep -n "state_history = deque" oracle/oracle.py` |
