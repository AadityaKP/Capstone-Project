# Founder UI — component inventory

A complete description of the founder-facing web application on the
`oefa-loop` branch: every screen, every panel and component, what each one
shows, where its numbers come from, and the rules it obeys. Written so that
a reader (human or model) who has never opened the code can reason about
the interface and audit it.

Code lives in `frontend/src/`. Stack: React 18, Vite 6, `lucide-react`
icons, one hand-written stylesheet (`styles.css`), no router library
(hash routing), no state library (one `useReducer` store persisted to
`localStorage`). Backend: FastAPI (`backend/`), reached at `/api`.

---

## 1. Architecture in one paragraph

The UI is **local-first**. The company, its monthly numbers, every board
analysis it has received, every multi-month cycle and every close-the-month
answer live in the browser (`store.jsx`, key `ssom_founder_v1`). The engine
is only needed to *produce* an analysis or cycle; everything already
produced renders without it. When the API is unreachable the UI shows an
honest failure card and never fabricates a result. A separate **sample
company** (`sample.js`) runs entirely in memory, is labelled everywhere,
and never touches the founder's stored data.

Two vocabularies exist and the boundary between them is enforced. The
engine speaks in `innovation_factor`, `consumer_confidence`,
`churn_smb`, Rule of 40. The founder never sees those words: the server
translates them once (`backend/founder_view.py`), the client translates its
own local numbers with the same thresholds (`founderView.js`, both reading
`config/founder_view.json`), and `copy.js` is the only place engine enums
become sentences. A test (`tests/test_founder_contract.py`) fails if a
component source file mentions an engine-only field or renders the ∞ glyph.

---

## 2. Shell and navigation (`App.jsx`)

- **Hash routes**: `#/` welcome, `#/onboarding`, `#/analyzing`, `#/home`,
  `#/plan`, `#/advice/:id` (single-month detail), `#/history`, `#/company`,
  `#/update` (close the month), `#/settings`. Bare `#/advice` folds into
  `#/plan`.
- **Route guard**: without a company every route redirects to welcome; with
  one, welcome redirects to home.
- **Bare shell** (no sidebar) for welcome, onboarding and analyzing.
- **Sidebar**: brand mark, company name, "AI advisory board"; five nav
  items — Home, Plan, History, My company, Settings; a status block
  ("Advisor · Ready / Sample company · numbers from <date>").
- **Topbar**: page title, company name, the demo badge in sample mode, and
  a "Close the month" button (hidden on the close page and in sample mode).

---

## 3. Screens

### 3.1 Welcome (`pages/Welcome.jsx`)

Honest framing before any input: three advisors plus a strategist, whose
experience comes from *simulated* scenarios, not real company data. Two
actions: **Get started** (onboarding) and **Explore a sample company**
(enters sample mode). A collapsible "How it works" with three cards
(describe your company, get analysed advice, update monthly).

### 3.2 Onboarding (`pages/Onboarding.jsx`)

Three steps, draft saved in the store as you type.

| Step | Required | Optional |
|---|---|---|
| Your company | name, company age (months), market crowdedness (3 choices → competitor count 2/5/9) | what you sell |
| Money | MRR, cash in the bank, total monthly costs | marketing spend last month |
| Customers | average price per customer, monthly churn (annual→monthly converter) | new customers last month, product maturity (3 choices → quality proxy 0.2/0.5/0.8), acquisition cost, team size, per-segment churn |

Live feedback lines: the runway sentence as soon as cash and costs exist;
the unit-economics verdict ("Healthy" / "Worth watching" / "Costs more than
it returns" / "Not enough data yet") once price, churn and either CAC or
spend+customers exist. Finishing creates the company and month 0 and goes
to Analyzing.

### 3.3 Analyzing (`pages/Analyzing.jsx`)

Starts one cycle (`POST /api/cycles`, returns 202 immediately) with the
current month's numbers, the prior months as history, a 4-month horizon,
and the previous close's **track record** if one exists. Stores the cycle
record and hands off to Plan, which polls. Shows a spinner, the staged
progress list, and "You can leave this page; the cycle keeps running on the
engine" — true, because the work runs server-side. On failure: an honest
card (numbers are saved, engine unreachable), Retry, Continue without a
plan, and a pointer to the sample company. In sample mode it redirects to
Plan.

### 3.4 Home (`pages/Home.jsx`) — "where am I, what should I do, what changed"

Top to bottom:

1. **Position banner** (click → Plan): risk chip + one sentence assembled
   from brief enums only (`positionSentence`): risk level, growth outlook,
   the board's top focus. If the latest analysis is not for the current
   month: "Your numbers changed since the last analysis — run a fresh one."
   If no analysis: "No analysis yet."
2. **Stale-analysis banner** with a Run / Re-analyse button when the
   analysis is not for the current month.
3. **"How last month's plan held up"** (only after a close-the-month whose
   cycle was made on the previous month): the prediction error as
   sentences — "We projected $40k of revenue, you did $35k — we were 15%
   high", churn expected vs moved, cash expected vs ended; one line per
   action the founder marked *didn't* ("we're not counting this month as
   evidence about it"); and, if nothing could be written back, the server's
   reason. Each line carries a tone (good / warn / bad) from tolerance and
   direction.
4. **KPI row** (four tiles, client-computed from the month's numbers):
   *Cash lasts* (runway against net burn; "Not burning" when revenue covers
   costs; never ∞), *Revenue* (MRR, with "you spend $X per $1 earned"),
   *Customers lost* ("1 in N" per month), *Winning customers* (LTV:CAC
   verdict with the payback sentence). Delta arrows against the previous
   month. A "watch" band on runway < 12 months and on unhealthy economics.
5. **This month's plan**: the four plan cards in compact form (see §4.5),
   link "The next 4 months" → Plan.
6. **What changed**: revenue grew/fell %, churn rose/improved pp, and
   "Last analysis: <why it ran>" (initial / scheduled / re-analysed early
   because runway/revenue/churn moved). Link to History.
7. **Evidence peek** (→ Advice detail): the strategist's one-line expected
   outcome ("In simulation, the next 6–12 months most often looked like:
   growth") with the "Simulated scenarios, not real companies" label.
8. **Freshness footer**: numbers-from date, days ago, "Close the month"
   button (primary once the numbers are over 35 days old).

### 3.5 Plan (`pages/Cycle.jsx`) — the primary surface

Renders the latest cycle. Polls `GET /api/cycles/{id}` every 2.5 s while it
runs, dispatching months as they land; when month 1 exists it also becomes
an *analysis* record so Home and the Advice detail work unchanged.

- **Banners**: "This plan was made on your <month> numbers — close that
  month" (when numbers moved on); cycle failed with the engine's reason
  (e.g. the engine restarted mid-cycle) and a Re-run; poll error; and, when
  no month got a real strategist read, "The AI strategist couldn't be
  reached for this plan. Every month here comes from the board's built-in
  rules."
- **Header**: "4 months, planned together"; while running, "Month n of 4 is
  being deliberated · m:ss elapsed"; when started from a close, "Started
  from your actual numbers, carrying last month's prediction error."
- **Month strip**: one column per month of the horizon, all visible at
  once (wrapping only below ~700 px). Each column: month label, a chip
  (**this month** for month 1, **projection** for months 2–H, **closed**
  once month 1 has real numbers), a brief-freshness chip (**fresh read** =
  the strategist was called, **reused** = cache hit or carried over,
  **rules only** = no brief), the risk chip, four action lines (marketing
  $, product $, hiring, price), a 2×2 state grid *after* the month
  (revenue, cash, cash lasts, churn — the founder's real numbers once
  closed), a "ran out of cash in simulation" warning if the month died, the
  **OEFA strip** (§4.9; open by default on month 1), and on month 1 a "Full
  advice" link to the detail page. Month 1 is visually dominant (full
  opacity, accent border); months 2–H are lighter and dashed-chipped
  because they are the model's physics compounded.
- **Loop lines** under the strip: fresh vs reused reads, whether each
  month was written back as simulated evidence (or "causal evidence graph
  off"), "memory scoped to your company", total deliberation seconds.
- **Trajectory chart** ("Where this plan takes you"): three small
  fan-charts (monthly revenue, cash, customers lost) from "now" through the
  horizon; the line is the stepped path, the band is the per-month spread
  across simulated worlds (server `projection_band`, p25–p75 one month
  ahead), the line goes dashed then ends with a marker if runs die. Caveat
  text under it: months 2 onward are the model compounded, not a forecast;
  the simulator scored a B against real companies; the real numbers are
  checked when the month closes.
- **Timeline** ("Your months, then the plan"): the founder's recorded
  months oldest→newest (MRR with % change, churn, cash lasts), a **NOW**
  rule, then one card per planned month ("this month's plan" / "projected")
  with the after-state and the plan line. When month 1 has been closed, its
  card is replaced in place by **"what actually happened"** with the
  prediction-error sentences and the score line ("3 of 4 predictions moved
  in the right direction; 2 landed within tolerance").
- **Empty state** (no cycle yet): explains the four-month loop and offers
  "Run the plan".

### 3.6 Advice detail (`pages/Advice.jsx`, `#/advice/:id`)

The single-month analysis in layers with progressive disclosure. Reached
from a month in the strip/timeline, Home's evidence peek, or History.

1. Archived banner if it is not the latest analysis ("shown as it was"),
   link to the current plan.
2. Amber banner when `llm_ok` is false: the strategist could not be
   reached; the plan is the board's built-in rules.
3. Position banner (as on Home, static).
4. **Confidence strip**: one server-built sentence in which the count of
   estimated inputs *caps* the confidence band ("Moderate confidence — 2 of
   these numbers are estimates, not yours"), why the analysis ran, whether
   the brief was reused, numbers-from date.
5. **OEFA strip** (§4.9), open, built from the cycle month when the
   analysis came from a cycle, else synthesised from the trace.
6. **Watch-outs / Working in your favor**: the strategist's free-text
   bullets, guard-railed (`guardBullets`: max 3, 140 chars, a bullet whose
   numbers match nothing the founder typed is dropped).
7. **Plan cards** (four; §4.5) with "I'm doing this" toggles, or a single
   "Nothing to change this month" panel when no card is an action.
8. **What-if panel** (§4.10), run on demand.
9. Expandables: **Why this plan** (focus-mix bar from the applied weights
   + reasoning bullets from modifier percentages and recommended focus);
   **Evidence** (memories as founder sentences, causal-graph lines);
   **Numbers we guessed (n)** (every input the server filled in, its value
   and why, with "Fill these in" → close-the-month; engine internals
   collapse to "assumes normal market conditions"); **Expected, in
   simulation** (the strategist's qualitative expected outcome, labelled as
   not a forecast).
10. **Next actions** checklist: the action cards as checkboxes plus "Update
    numbers around <next month>".

### 3.7 History (`pages/History.jsx`)

Newest-first vertical rail of recorded months. From three months on, three
mini sparklines (MRR, churn, cash lasts) sit above it. Each entry: month
name, the risk chip of that month's analysis, "MRR $ (±%) · churn % (±pp) ·
cash lasts", the plan focus line, "You accepted n of m actions", and an
**outcome badge** once a month six or more months later exists ("6 months
later: growth / flat / decline — what happened next, not credit", the same
±10% rule the engine's memory uses). Expanding shows cash/costs/price, new
customers and marketing, each decision with its glyph (✓ done, ✎ adjusted
with the founder's note, ○ suggested / declined), and "Open full advice".

### 3.8 My company (`pages/Company.jsx` → `CompanyView`)

The data-honesty ledger: every value the board uses with a provenance chip
— **You provided · <date>**, **Derived**, **Estimated by the system**.
Sections Money (MRR, cash, costs, cash lasts), Customers (price, churn, new
customers, marketing spend, acquisition cost with its source, lifetime
value), Company & market (age, crowdedness, maturity, team size, "Market
conditions: typical conditions assumed"). Footer banner: nothing else is
collected or observed.

### 3.9 Close the month (`pages/Company.jsx` → `UpdateRitual`, `#/update`)

One screen, one submit (the HITL step of the loop).

- **"Last month the board asked for these — what happened?"** (shown when
  the latest cycle was made on the month being closed and is not yet
  closed): one row per *action* card from that plan — domain, headline, a
  three-way toggle **Did it / Partly / Didn't**, an optional note. Choosing
  *Didn't* shows "You didn't do this, so this month won't count as evidence
  about it." Every action must be answered before submitting.
- **Number grid**, pre-filled with last month: MRR, cash, costs, monthly
  churn, new customers, marketing spend, price. Instant diff pills (MRR ±%,
  churn ±pp, cash ±$).
- **Submit** ("Close the month & plan again"): saves the new month; records
  the answers as the planned month's decisions; posts the close
  (`POST /api/cycles/{id}/feedback`) with the answers and the actual
  numbers; stores the server's result (prediction error, what was written
  back, the track record); on engine failure keeps the numbers and says the
  plan could not be scored; then starts the next cycle. In sample mode the
  form is disabled with a banner.

### 3.10 Settings (`pages/Settings.jsx`)

- **Advice**: toggle for richer per-advisor explanations.
- **Analysis service**: connected / not reachable, then the **loop
  capability panel** from `/api/health`: advisor mode, and three lines each
  on/off with the server's reason — Strategist (language model), Memory
  scoped to your company, Causal evidence graph ("Off — … The board still
  advises and remembers, but what happens next is not written back as
  evidence").
- **Seeded demo company** (when the engine has one): load it, with a
  replace-confirmation.
- **Sample company** (in sample mode): leave.
- **Your data**: delete everything in this browser, with confirmation.
- Footer: decision support from a calibrated simulation, not financial
  advice, not a forecast.

---

## 4. Shared components (`components.jsx`, `whatif.jsx`)

| Component | What it shows |
|---|---|
| 4.1 `RiskChip` | Risk enum → "Low / Moderate / Elevated / Critical risk", green/blue/amber/red, shield or warning icon. |
| 4.2 `ProvChip` | Provenance: You provided (with date) / Estimated by the system / Derived / Simulated. |
| 4.3 `DeltaArrow` | ±value with up/down arrow; colour depends on whether up is good (`goodWhenDown` for churn); "flat" under 0.05. |
| 4.4 `Banner`, `SimulatedTag`, `DemoBadge` | Info/warn banners with optional actions; the flask "From simulations, not real companies" tag; "Sample company — data is illustrative". |
| 4.5 `KpiCard` | Label, big value, sub-line or delta, optional watch band, hover hint with the definition. |
| 4.6 `buildPlanCards` / `PlanCard` | The board's final action → four cards: **Product & retention** ("Invest ≈$X in product" / "Hold product spend"), **Marketing & growth** ("Spend ≈$X on performance channels / brand building", "up/down from the ≈$Y you reported"), **Hiring** ("Room to add ≈$/mo of payroll" / "Wait on hiring"), **Pricing** ("Consider a ≈N% price increase" / "Hold pricing"). Each has % of MRR, a rationale sentence, a "Why this number?" chain (base rule → strategic adjustment in words like "scaled back" → floor), an `isAction` flag, a Priority pill on the domain matching the board's top weight, and an accept toggle. |
| 4.7 `FocusBar` | The board's applied weights as a four-segment bar: Product / Growth / Efficiency / Market. |
| 4.8 `EvidenceList`, `ConfidenceStrip`, `RiskBullets`, `ProgressStages`, `MiniLine`, `OutcomeBadge` | Memories rewritten as "A simulated company at <stage, churn, momentum> grew/declined/stayed flat over the following 6 months"; causal-graph lines split into *observed in past runs* vs *the board's working assumption (a built-in prior)*; the capped confidence sentence; the guarded bullets; the three-stage progress list; sparklines; the 6-months-later badge. |
| 4.9 `OefaStrip` | **Observed · Decided · Expected · Changed**, one component for every month everywhere. Header: toggle, freshness chip, "strategist unreachable" chip. *Observed*: "n similar past months recalled", revenue trend word, "the board's read: cash ran tight / churn spiked / …" (stress node in founder words), "causal evidence graph off" when it is. *Decided*: the four action lines, why the analysis ran, brief fresh/reused. *Expected*: "revenue +9.5%, churn −0.3pp, cash −3.5% over 2 months" + simulated tag, or "no numeric prediction on this analysis". *Changed*: the prediction-error sentences against the simulated next state ("the simulation did $39k — we were 8% high") or, once closed, against the founder's real numbers; the score line; what the board changed (brief refreshed/reused and why, weight moves, per-agent adaptation sentences such as "Last month's plan expected churn −0.3pp and saw +0.2pp; product spend is held back 25% this month"); whether the month was written back as *simulated* evidence, kept apart from anything real, or the close's evidence reason. |
| 4.10 `WhatIfPanel` / `FanChart` | 12-month projection under two arms — the board's plan vs doing nothing (a third research arm exists server-side and is not drawn): four fan-charts (revenue, cash, customers lost, and spend-per-$1 or Rule of 40 depending on size), a headline if the board's own plan ran out of cash, a competitor-shock toggle, the caveat sentence directly under the charts, a survivor note when lines go dashed, a summary table (revenue/cash at 12 mo, survives %, efficiency, shock cost, recovery), the seed count, a "conditions diverged" warning, and an expandable list of every assumption with its basis and source (costs, gross margin, price held flat, plan persistence, research shocks, current marketing spend). |

---

## 5. Copy and translation layers

- **`copy.js`**: risk / outlook / outcome enums → words; focus labels;
  confidence band (never a percentage); refresh-reason copy; brief-source
  copy; `positionSentence`; `scaleWord` (modifier % → "scaled back",
  "nudged up" …); `guardBullets`; `rewriteMemory`; `expectedOutcomeCopy`;
  domain titles; channel words; causal stress/effect node names → founder
  phrases (an unmapped node is dropped, never shown raw).
- **`loopView.js`**: cycle vocabulary → sentences: `expectedLine`,
  `predictionSentences` (the "we were N% high/low" builder, sign convention
  realized − expected), `scoreLine`, `briefFreshness`, `actionSummary`,
  the did/partly/didn't states and their mapping to decision states.
- **`derive.js`**: CAC = spend ÷ new customers (or provided), LTV = price ÷
  churn, crowdedness/maturity maps, money rounding ($500 under $20k, $1k
  above; per-unit to the dollar), pct/pp formatting, month deltas, and the
  client-side mirror of the engine's re-analysis triggers.
- **`founderView.js`** (thresholds from `config/founder_view.json`):
  runway (null = not burning, never Infinity), churn phrases ("1 in N"),
  the LTV:CAC verdict with its unmeasurable ceiling and small-sample refusal,
  Rule-of-40 gate (≥ $1M ARR), spend-per-$1, confidence caps.

---

## 6. State (`store.jsx`)

```
{
  demo: bool,                       // sample mode, never persisted
  company: { id, name, whatYouSell, ageMonths, crowdedness, maturity, headcountReal, createdAt },
  months: [{ id, index, enteredAt,
             values: { mrr, cash, costs, price, churnMonthly, newCustomers?, marketingSpend?,
                       cacDirect?, churnEnt?, churnSmb?, churnB2c? },
             decisions: [{ id, domain, text, state: accepted|custom|declined|suggested, note? }] }],
  analyses: [{ id, monthId, cycleId?, monthIndex?, createdAt, source: api|cycle|sample,
               llm_ok, reason, brief, trace, display, narratives }],
  cycles:   [{ id, monthId, createdAt, source, status: queued|running|completed|failed,
               horizon, months: [<cycle month>], summary, meta, error,
               feedback: [{ monthIndex, submitted, result, error, closedAt }],
               startedFromTrackRecord }],
  settings: { narratives }, onboardingDraft
}
```

Actions: ENTER_DEMO / EXIT_DEMO, IMPORT_STATE (seeded workspace),
SAVE_DRAFT, CREATE_COMPANY, ADD_MONTH, ADD_ANALYSIS, ADD_CYCLE,
UPDATE_CYCLE, SET_CYCLE_FEEDBACK, SET_DECISION, SET_SETTING, RESET_ALL.
Selectors: latest/previous month, latest analysis, analysis for month,
latest cycle, cycle for month/by id, latest closed feedback, feedback for a
cycle month, `analysisFromCycle` (month 1 of a cycle → an analysis record).

---

## 7. API surface used by the UI (`api.js`)

| Call | Used by | Notes |
|---|---|---|
| `GET /api/health` | Settings | includes the `loop` capability block |
| `POST /api/cycles` | Analyzing | 202; body = founder numbers + history + horizon + previous track record |
| `GET /api/cycles/{id}` | Plan | status, months so far, summary, closes |
| `POST /api/cycles/{id}/feedback` | Close the month | answers + actuals → prediction error, evidence written, track record |
| `POST /api/whatif` | Advice detail | 12-month projection, no LLM |
| `POST /api/advise` | (kept; no longer called by the UI) | single-month analysis |
| `GET /api/demo/bootstrap` | Settings | the seeded workspace, if one exists |

Every request returns `{ok, data}` or `{ok:false, offline, error}`; the UI
never invents a result on failure. Timeouts: 4 s health, 20 s cycle start,
15 s poll, 60 s close, 20 s what-if.

**Cycle month shape** (what Plan, the OEFA strip and Home read):

```
{ month_index, projection, latency_s,
  observe:  { state_before{mrr,cash,churn_pct,runway_months}, memory_count, memories[],
              trend{mrr_trend,…}, graph{stress_node, contexts, summary, enabled},
              memory_scope, pending_memories, matured_memories },
  execute:  { action{marketing,product,hiring,pricing}, brief, llm_ok, brief_source: llm|cache_hit|reuse|none,
              refresh_reason: initial|event|cadence|null, proposal_source, proposals[],
              weights, base_weights, expected_delta{mrr_pct,cash_pct,churn_pp,runway_months,
              horizon_months,band,basis:"simulated"}, spend_ceiling, display, trace },
  feedback: { state_after{…,survived}, kpi_delta, prediction_error{per KPI: expected, realized,
              error, within_tolerance, sign_agrees; summary}, projection_band, evidence_written,
              evidence_source:"sim", basis:"simulated" },
  adapt:    { what_changed[], refresh_reason, brief_source, weight_moves[], adaptations[],
              track_record_for_next_month, memory{pending, matured_this_month} } }
```

---

## 8. Honesty rules the UI enforces

1. No fabricated analysis: engine down → failure card; strategist down →
   amber banner and "rules only" chips; graph down → said in words, never
   silently.
2. Every non-founder number is labelled: **Estimated**, **Derived**,
   **Simulated**, and the assumptions list is one click away.
3. Confidence is capped by how much was assumed; it is a band, never a
   percentage.
4. Months 2–H are chipped **projection**; the chart caveat says they are
   the model compounded; the only uncertainty grammar is the seed band.
5. The prediction error is shown, not hidden, on Home first, in the same
   list as "revenue grew 4%".
6. An action the founder did not take is never counted as evidence, and
   the UI says so on the close form, on Home and in the OEFA strip.
7. Engine vocabulary never reaches a component (test-enforced); infinity
   is never rendered; the sample company is labelled on every screen.
