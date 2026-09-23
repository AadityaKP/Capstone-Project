# Founder UI — component inventory

A complete description of the founder-facing web application on the
`ui-simplify` branch (the result of `docs/ui_simplification_plan.md`
applied to the `oefa-loop` inventory): every screen, every panel and
component, what each one shows, where its numbers come from, and the rules
it obeys. Written so that a reader (human or model) who has never opened
the code can reason about the interface and audit it.

Code lives in `frontend/src/`. Stack: React 18, Vite 6, `lucide-react`
icons, one hand-written stylesheet (`styles.css`), no router library
(hash routing), no state library (one `useReducer` store persisted to
`localStorage`). Backend: FastAPI (`backend/`), reached at `/api`. Tests:
`vitest` + `jsdom` + `@testing-library/react` under `frontend/test/`
(the honesty checklist), plus the Python source scans in
`tests/test_founder_contract.py`.

---

## 1. Architecture in one paragraph

The UI is **local-first**. The company, its monthly numbers, every board
analysis it has received, every multi-month cycle and every close-the-month
answer live in the browser (`store.jsx`, key `ssom_founder_v1`). The engine
is only needed to *produce* a cycle; everything already produced renders
without it. When the API is unreachable the UI shows an honest notice and
never fabricates a result. A separate **sample company** (`sample.js`) runs
entirely in memory, is labelled everywhere, and never touches the founder's
stored data.

The cycle's lifecycle lives at app level (`cycleRun.jsx`,
`CycleRunProvider`): starting a cycle, polling it while it runs, and
promoting its first month to an analysis record happen whichever page is
open, so a cycle asked for by the Close form lands even if the founder
walks off to History. Pages read `{ start, requestStart, starting,
startError, pollError, elapsed }` from `useCycleRun()`.

Two vocabularies exist and the boundary between them is enforced. The
engine speaks in `innovation_factor`, `consumer_confidence`,
`churn_smb`, Rule of 40. The founder never sees those words: the server
translates them once (`backend/founder_view.py`), the client translates its
own local numbers with the same thresholds (`founderView.js`, both reading
`config/founder_view.json`), and `copy.js` / `loopView.js` are the only
places engine enums and cycle vocabulary become sentences. A test
(`tests/test_founder_contract.py`) fails if a component source file
mentions an engine-only field or renders the ∞ glyph.

---

## 2. Shell and navigation (`App.jsx`, `main.jsx`)

- **Hash routes**: `#/` welcome, `#/onboarding`, `#/analyzing`, `#/home`
  (This month), `#/advice/:id` (Why this plan), `#/history`, `#/company`,
  `#/update` (Close the month; `#/update/fill` opens it with the
  estimated numbers expanded), `#/settings`. `#/plan` and bare `#/advice`
  render This month. Route names are unchanged from the previous
  inventory; only the labels changed (decision D5).
- **Route guard**: without a company every route redirects to welcome; with
  one, welcome redirects to This month.
- **Bare shell** (no sidebar) for welcome, onboarding and analyzing.
- **Sidebar**: brand mark, company name, "AI advisory board"; four nav
  items — **This month, History, My company, Settings**. No status block.
- **Topbar**: page title, company name, the demo badge in sample mode, and
  a **Close the month** button (hidden on the close page and in sample
  mode). It switches to the primary style when the numbers are older
  than 35 days, and on nothing else: "not current" and "cycle failed" are
  conditions for the notice slot on This month, not for this button.
- `main.jsx` is the only place that mounts the app; `App.jsx` exports the
  shell so the tests can render it against a fixture state.

---

## 3. Screens

### 3.1 Welcome (`pages/Welcome.jsx`)

Honest framing before any input: three advisors plus a strategist, whose
experience comes from *simulated* scenarios, not real company data. Two
actions: **Get started** (onboarding) and **Explore a sample company**
(enters sample mode, lands on This month). A collapsible "How it works"
with three cards (describe your company, get analysed advice, update
monthly).

### 3.2 Onboarding (`pages/Onboarding.jsx`)

Three steps, draft saved in the store as you type. Each step shows its
required fields; the optional ones sit behind one **"Add more detail —
raises confidence"** toggle per step.

| Step | Required | Behind "Add more detail" |
|---|---|---|
| Your company | name, company age (months), market crowdedness (3 choices → competitor count 2/5/9) | what you sell, team size |
| Money | MRR, cash in the bank, total monthly costs | marketing spend last month, acquisition cost if tracked |
| Customers | average price per customer, monthly churn (annual→monthly converter) | new customers last month, product maturity (3 choices → quality proxy 0.2/0.5/0.8), per-segment churn |

Live feedback lines: the runway sentence as soon as cash and costs exist;
the unit-economics verdict ("Healthy" / "Worth watching" / "Costs more than
it returns" / "Not enough data yet") once price, churn and either CAC or
spend+customers exist; "That's enough for your first analysis." on the
last step. Finishing creates the company and month 0 and goes to
Analyzing.

### 3.3 Analyzing (`pages/Analyzing.jsx`)

The onboarding wait screen only; every later run happens on This month.
Calls the provider's `start()` (which posts `POST /api/cycles` with the
current month's numbers, the prior months as history, a 4-month horizon
and the previous close's track record if one exists, then stores the
cycle record) and hands off to This month. Shows a spinner, the staged
progress list, and "You can leave this page; the cycle keeps running on
the engine" — true, because the work runs server-side and the polling
lives in the provider. On failure: an honest card (numbers are saved,
engine unreachable), Retry, Continue without a plan, and a pointer to the
sample company. In sample mode it redirects to This month.

### 3.4 This month (`pages/Home.jsx`, `#/home`) — "am I OK, what do I do, how did last month go, where does this take me"

Top to bottom, at most six sections:

1. **Notice slot** (0–1 banners, chosen by `notice.js`, §4.11): start
   failed (Retry / Continue without a plan) · cycle failed with the
   engine's reason (Re-run) · lost contact with a running cycle ·
   rules-only (the whole plan, or "Months 3–4 of this plan used the
   board's built-in rules"). A condition that loses the slot becomes one
   sentence in the plan section.
2. **Status line** (static): risk chip + one sentence assembled from brief
   enums only (`positionSentence`) + "Based on your <month> numbers · n
   days ago". While a cycle deliberates before its first month lands: "The
   board is reading your numbers." With no plan: "No plan yet — run your
   first one."
3. **KPI row** (four tiles, client-computed from the month's numbers):
   *Cash lasts* (runway against net burn; "Not burning" when revenue
   covers costs; never ∞), *Revenue* (MRR), *Customers lost* ("1 in N"
   per month), *Winning customers* (the LTV:CAC verdict). Delta arrows
   against the previous month. The spend-per-$1 and payback sentences live
   in each tile's hover hint. A "watch" band on runway < 12 months and on
   unhealthy economics.
4. **How last month's plan held up** (only after a close-the-month whose
   cycle was made on the previous month and produced a result): the
   prediction error as sentences — "We projected $40k of revenue, you did
   $35k — we were 15% high", churn expected vs moved, cash expected vs
   ended; one line per action the founder marked *didn't* ("we're not
   counting this month as evidence about it"); the server's reason if
   nothing could be written back; the score line; and **"What the board
   changed:"** — read only from the cycle that answered this close (a
   different cycle, planned on the latest month): nothing while it is still
   running; its per-agent adaptation sentences once month 1 has landed;
   otherwise, when it started from the track record, "The board planned
   this month with last month's result in hand." "Details" → History.
5. **This month's plan**: header = the server's confidence sentence
   (verbatim; the assumption count caps the band) with "Why this plan →"
   and, when idle, a Re-run / Plan again link; an inline amber note when
   the plan is on the previous month's numbers or lost the notice slot;
   full plan cards (§4.6) for the domains that are actions; one line
   "Holding: hiring, pricing." for the rest; the "Nothing to change this
   month" panel (with "Show what each advisor said") when no card is an
   action; one red line "In simulation this plan runs out of cash around
   month N" when a horizon month died.
   States rendered in place: **no cycle** → a "No plan yet" panel with
   *Run the plan*; **planning** → "Your board is planning" with the staged
   progress list and "Month k of 4 · m:ss" (elapsed derives from the
   cycle's own `createdAt`, so it survives a reload), then the plan
   appears as month 1 lands; **failed** / **start failed** → the notice
   slot. Run, Re-run and Retry call `start()` and stay on This month.
6. **Outlook** (`outlook.jsx`, "Where this plan takes you"): **one** fan
   chart with a metric switch Revenue / Cash / Customers lost (default:
   Cash when the runway tile is on watch, Revenue when the efficiency tile
   is, else Cash). The opening point is the founder's own numbers; the
   line is the stepped path through the horizon and the band the
   per-month spread across simulated worlds (server `projection_band`).
   Everything right of the founder's real numbers sits on a light
   **"projected"** background band (decision D2: shaded means "this is the
   model"; dashed keeps meaning "some simulated runs ran out of cash", and
   the line ends at a marker when none are left). A closed month 1 plots
   the founder's actual numbers as a marked point labelled "you". One
   caveat line under the chart (`OUTLOOK_CAVEAT` in `loopView.js`). A
   collapsed **"Next 3 months of the plan"** lists each later month's four
   action lines, filling in as months land ("deliberating…" until then).

### 3.5 Why this plan (`pages/Advice.jsx`, `#/advice/:id`)

Why the board recommends the plan and how far to trust it; every trace two
clicks from This month. Reached from the plan's header or a History entry.
The actions themselves are not repeated here.

1. **Notice slot** (0–1, `notice.js`): rules-only (whole plan or
   partial months) · archived analysis ("shown as it was", with *Current
   plan*). The loser becomes one line under the slot.
2. **Summary**: "The board's top focus is Product." + the confidence
   strip (the capped server sentence, why the analysis ran, whether the
   brief was reused, numbers-from date).
3. **Watch-outs / Working in your favor**: the strategist's free-text
   bullets, guard-railed (`guardBullets`: max 3, 140 chars, a bullet whose
   numbers match nothing the founder typed is dropped).
4. **The plan against doing nothing** (§4.10), run on demand.
5. **Evidence — what this is based on** (collapsed): the Observed lines
   from the cycle month ("2 similar past months recalled", the revenue
   trend, "the board's read: churn spiked"), the memories as founder
   sentences, the causal-graph lines, and the strategist's qualitative
   expected outcome with the simulated tag — the only place that tag
   appears.
6. **Assumptions — numbers we guessed (n)** (collapsed): every input the
   server filled in that the founder could supply, its value and why,
   "Fill these in" → `#/update/fill`; the projection's own assumptions
   once it has run; engine internals collapse to "also assumes normal
   market conditions".
7. **How the board weighed it** (collapsed): the focus-mix bar from the
   applied weights + reasoning bullets from the modifier words and the
   recommended focus.
8. **How the board got here** (collapsed): the loop lines from the cycle
   summary (fresh vs reused reads, written back as simulated evidence or
   "causal evidence graph off", memory scope, deliberation seconds), the
   sentence that months 2 onward are the model compounded, then one
   **OEFA strip** (§4.9) per horizon month — month 1 open by default,
   with the founder's close when there is one. A pre-cycle analysis
   renders one strip synthesised from its trace.

### 3.6 History (`pages/History.jsx`)

Newest-first vertical rail of recorded months. From three months on, three
mini sparklines (MRR, churn, cash lasts) sit above it. Each entry: month
name, the risk chip of that month's analysis, "MRR $ (±%) · churn % (±pp) ·
cash lasts", the plan focus line, and **"Did n of m · partly k"** counted
over one decision per domain (the Close form's answer wins over any older
row; legacy "suggested" rows still render with ○ but leave the
denominator). Expanding shows cash/costs/price, new customers and
marketing, each decision with its glyph (✓ did, ✎ partly with the
founder's note, ○ didn't / suggested), **How the plan held up** (the
close's scored prediction sentences, the score line, and what the board
changed in the cycle planned on the following month), the **outcome
badge** once a month six or more months later exists ("6 months later:
growth / flat / decline — what happened next, not credit", the same ±10%
rule the engine's memory uses), and "Why this plan".

### 3.7 My company (`pages/Company.jsx` → `CompanyView`)

The data ledger: one line says where the numbers come from ("From your
September 2026 close (Sep 23). Values without a marker are yours as you
entered them."), then every value the board uses. Markers appear only on
exceptions — **Derived**, **Estimated by the system** — never on the
founder's own values. Sections Money (MRR, cash, costs, cash lasts),
Customers (price, churn, new customers, marketing spend, acquisition cost
with its source, lifetime value), Company & market (age, crowdedness,
maturity, team size, "Market conditions: typical conditions assumed").
One closing sentence: estimated values are the system's assumptions, not
measurements.

### 3.8 Close the month (`pages/Company.jsx` → `UpdateRitual`, `#/update`)

One screen, one submit (the HITL step of the loop).

- **"Last month the board asked for these — what happened?"** (shown when
  the latest cycle was made on the month being closed and is not yet
  closed): one row per *action* card from that plan — domain, headline, a
  three-way toggle **Did it / Partly / Didn't**, an optional note. Choosing
  *Didn't* shows "You didn't do this, so this month won't count as evidence
  about it." Every action must be answered before submitting.
- **Number grid**, pre-filled with last month: MRR, cash, costs, monthly
  churn. New customers, marketing spend and price sit inline when the
  board guessed nothing, and otherwise under a collapsed **"Numbers we
  estimated — replace them if you know them (n)"** group that lists what
  the latest analysis assumed (`#/update/fill` opens it). Instant diff
  pills (MRR ±%, churn ±pp, cash ±$).
- **Submit** ("Close the month & plan again"): saves the new month; records
  the answers as the planned month's decisions; awaits the close
  (`POST /api/cycles/{id}/feedback`, 60 s timeout) with the answers and
  the actual numbers; stores the server's result (prediction error, what
  was written back, the track record); on engine failure keeps the numbers
  and says the plan could not be scored; then asks the provider for the
  next cycle (`requestStart()`, so it is built from the state that already
  holds the new month and the feedback) and lands on This month, where the
  plan builds in place. In sample mode the form is disabled with the one
  sample-company notice.

### 3.9 Settings (`pages/Settings.jsx`)

- **Advice**: toggle for richer per-advisor explanations.
- **Engine status** (collapsed; the title says connected / not reachable):
  the connection line, then the loop capability list from `/api/health`:
  advisor mode, and three lines each on/off with the server's reason —
  Strategist (language model), Memory scoped to your company, Causal
  evidence graph ("Off — … The board still advises and remembers, but what
  happens next is not written back as evidence").
- **Seeded demo company** (when the engine has one): load it, with a
  replace-confirmation.
- **Sample company** (in sample mode): leave.
- **Your data**: delete everything in this browser, with confirmation.
- Footer: decision support from a calibrated simulation, not financial
  advice, not a forecast.

---

## 4. Shared components (`components.jsx`, `whatif.jsx`, `outlook.jsx`, `notice.js`)

| Component | What it shows |
|---|---|
| 4.1 `RiskChip` | Risk enum → "Low / Moderate / Elevated / Critical risk", green/blue/amber/red, shield or warning icon. |
| 4.2 `ProvChip` | Provenance, exceptions only: Estimated by the system / Derived / Simulated. Renders nothing for a value the founder provided. |
| 4.3 `DeltaArrow` | ±value with up/down arrow; colour depends on whether up is good (`goodWhenDown` for churn); "flat" under 0.05. |
| 4.4 `Banner`, `Notice`, `SimulatedTag`, `DemoBadge`, `Expandable` | Info/warn banners with optional actions; `Notice` renders what `pickNotice` chose; the flask "From simulations, not real companies" tag (Evidence only); "Sample company — data is illustrative"; one collapsed panel section. |
| 4.5 `KpiCard` | Label, big value, sub-line or delta, optional watch band, hover hint with the definition and the derived sentence. |
| 4.6 `buildPlanCards` / `PlanCard` | The board's final action → four cards: **Product & retention** ("Invest ≈$X in product" / "Hold product spend"), **Marketing & growth** ("Spend ≈$X on performance channels / brand building", "up/down from the ≈$Y you reported"), **Hiring** ("Room to add ≈$/mo of payroll" / "Wait on hiring"), **Pricing** ("Consider a ≈N% price increase" / "Hold pricing"). Each has a rationale sentence, a "Why this number?" chain (% of monthly revenue → base rule → strategic adjustment in words like "scaled back" → floor), an `isAction` flag and a Priority pill on the domain matching the board's top weight. No accept toggle: the founder answers once, on the Close form. Rendered in exactly two places — This month and the Close form. |
| 4.7 `FocusBar` | The board's applied weights as a four-segment bar: Product / Growth / Efficiency / Market. |
| 4.8 `EvidenceList`, `observedLines`, `ConfidenceStrip` / `confidenceLine`, `RiskBullets`, `ProgressStages`, `MiniLine`, `OutcomeBadge` | Memories rewritten as "A simulated company at <stage, churn, momentum> grew/declined/stayed flat over the following 6 months"; causal-graph lines split into *observed in past runs* vs *the board's working assumption (a built-in prior)*; the Observed beat's sentences, shared by the strip and the Evidence section; the capped confidence sentence (server verbatim, client fallback for pre-display analyses); the guarded bullets; the three-stage progress list; sparklines; the 6-months-later badge. |
| 4.9 `OefaStrip` | **Observed · Decided · Expected · Changed**, one component for every cycle month, under "How the board got here" (Why this plan). Header: toggle with the month label; no chips. *Observed*: "n similar past months recalled", revenue trend word, "the board's read: cash ran tight / churn spiked / …" (stress node in founder words), "causal evidence graph off" when it is. *Decided*: the four action lines, why the analysis ran, brief fresh/reused, "strategist unreachable — built-in rules" when so. *Expected*: "revenue +9.5%, churn −0.3pp, cash −3.5% over 2 months", or "no numeric prediction on this analysis". *Changed*: labelled **Changed (in simulation)** with "model consistency check, not accuracy" when scored against the simulated next state ("the simulation did $39k — we were 8% high"), or **Changed (your numbers)** once closed against the founder's real numbers; the score line; what the board changed (brief refreshed/reused and why, weight moves in founder words, per-agent adaptation sentences such as "Last month's plan expected churn −0.3pp and saw +0.2pp; product spend is held back 25% this month"); whether the month was written back as *simulated* evidence, kept apart from anything real, or the close's evidence reason. |
| 4.10 `WhatIfPanel` / `FanChart` / `WhatIfAssumptions` | 12-month projection under two arms — the board's plan vs doing nothing (a third research arm exists server-side and is not drawn). Visible: a headline if the board's own plan ran out of cash, the summary table (revenue/cash at 12 mo, survives %, efficiency, and shock cost / recovery when a shock is on) and the server's caveat sentence. Behind **"Show the charts"**: the legend, the competitor-shock toggle, four fan-charts (revenue, cash, customers lost, and spend-per-$1 or Rule of 40 depending on size), the survivor note when lines go dashed, the seed count and the "conditions diverged" warning. The projection's assumptions render in Why's Assumptions section. `FanChart` also takes `shadeFrom` (the projected band), `markers` (filled points) and `xEndLabel` for the Outlook. |
| 4.11 `Outlook` / `defaultOutlookMetric` | The This-month chart (§3.4.6). |
| 4.12 `pickNotice` / `rulesOnlyMonths` (`notice.js`) | The one-notice rule: at most one notice per page, in priority — engine unreachable / cycle failed / start failed (with the action) · sample company (Close only) · rules-only (whole plan, or "Months 3–4") · archived analysis (Why only) — plus one inline sentence per condition that lost the slot. |

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
  realized − expected; `basis` picks "you did" vs "the simulation did"),
  `scoreLine`, `loopLines`, `OUTLOOK_CAVEAT`, `cashDeathMonth`,
  `briefFreshness`, `actionSummary`, the did/partly/didn't states and their
  mapping to decision states.
- **`derive.js`**: CAC = spend ÷ new customers (or provided), LTV = price ÷
  churn, crowdedness/maturity maps, money rounding ($500 under $20k, $1k
  above; per-unit to the dollar), pct/pp formatting, month deltas,
  `monthOffsetLabel` for horizon months, and the client-side mirror of the
  engine's re-analysis triggers.
- **`founderView.js`** (thresholds from `config/founder_view.json`):
  runway (null = not burning, never Infinity), churn phrases ("1 in N"),
  the LTV:CAC verdict with its unmeasurable ceiling and small-sample refusal,
  Rule-of-40 gate (≥ $1M ARR), spend-per-$1, confidence caps.

No page authors a sentence about a prediction; every such sentence comes
from these files or from a server `display` block.

---

## 6. State (`store.jsx`, `cycleRun.jsx`)

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
SAVE_DRAFT, CREATE_COMPANY, ADD_MONTH, ADD_ANALYSIS (idempotent per
cycle month), ADD_CYCLE (idempotent per id), UPDATE_CYCLE,
SET_CYCLE_FEEDBACK, SET_DECISION, SET_SETTING, RESET_ALL. `StoreProvider`
takes an optional `initialState` for tests. Selectors: latest/previous
month, latest analysis, analysis for month, month by id, latest cycle,
cycle for month/by id, latest closed feedback, feedback for a cycle month,
`analysisFromCycle` (month 1 of a cycle → an analysis record).

Stored data from before the simplification still renders: analyses without
`display`, decisions with `state: "suggested"`, cycles without `feedback`,
analyses without `cycleId`.

`CycleRunProvider` (mounted inside the store provider): `start()` reads the
store from its current render and never navigates; `requestStart()` sets a
ref-held flag and an effect calls `start()` on the next render, so a caller
that has just dispatched gets a cycle built from the state it changed. The
poll (2.5 s, keyed on cycle id and status) and the month-1 promotion live
here; both survive `React.StrictMode`'s doubled effects because the
reducers are idempotent.

---

## 7. API surface used by the UI (`api.js`)

| Call | Used by | Notes |
|---|---|---|
| `GET /api/health` | Settings | includes the `loop` capability block |
| `POST /api/cycles` | `CycleRunProvider.start()` (Analyzing, This month, Close) | 202; body = founder numbers + history + horizon + previous track record |
| `GET /api/cycles/{id}` | `CycleRunProvider` poll | status, months so far, summary, closes |
| `POST /api/cycles/{id}/feedback` | Close the month | answers + actuals → prediction error, evidence written, track record |
| `POST /api/whatif` | Why this plan | 12-month projection, no LLM |
| `POST /api/advise` | (kept; no longer called by the UI) | single-month analysis |
| `GET /api/demo/bootstrap` | Settings | the seeded workspace, if one exists |

Every request returns `{ok, data}` or `{ok:false, offline, error}`; the UI
never invents a result on failure. Timeouts: 4 s health, 20 s cycle start,
15 s poll, 60 s close, 20 s what-if.

**Cycle month shape** (what This month, the Outlook, the OEFA strip and the
Last-month card read):

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
  adapt:    { what_changed[], refresh_reason, brief_source, weight_moves[{key,from,to,delta}],
              adaptations[{agent,sentence}], track_record_for_next_month,
              memory{pending, matured_this_month} } }
```

---

## 8. Honesty rules the UI enforces

Each rule keeps one visible surface and one reachable surface; the vitest
checklist (`frontend/test/honesty.test.jsx`) renders the real shell against
the sample fixture and pins them.

1. No fabricated analysis: engine down → the notice slot (start failed /
   cycle failed / lost contact) with the action; strategist down → one
   rules-only notice per page (whole plan or the months affected) and
   "strategist unreachable — built-in rules" in the trace; the engine
   status in Settings says what the board can actually do.
2. Every non-founder number is marked — **Estimated**, **Derived**,
   **Simulated** — and never a founder's own; the full list is one click
   away under Assumptions and on My company.
3. Confidence is capped by how much was assumed and is a band, never a
   percentage; the server's sentence is shown verbatim on the plan header
   and the Why summary.
4. The Outlook shades everything right of the founder's numbers as
   projected and carries one caveat; dashed means some simulated runs
   died; a sentence comparing the model with itself ("the simulation did")
   appears only inside How the board got here.
5. The prediction error is shown, not hidden: the Last-month card sits
   directly under the KPIs.
6. An action the founder did not take is never counted as evidence, and
   the UI says so on the close form (at the moment of choice) and on the
   Last-month card.
7. Engine vocabulary never reaches a component (test-enforced); infinity
   is never rendered; the sample company is labelled on every screen
   (test-enforced).

At most one banner on This month and on Why this plan; This month opens
with no expander open and exactly one caveat (test-enforced).
