# Founder UI simplification — execution plan

Source: the UX / information-architecture audit of `docs/ui_components.md`
(received 2026-09-23). This file turns it into steps I can execute on a new
branch, **`ui-simplify` from `oefa-loop`**, one commit per phase. Executed on
that branch on 2026-09-23 (seven commits, A → B → D → C → E → F → G); the
result is inventoried in `docs/ui_components.md` and recorded as decision 12.

Companion documents: `docs/ui_components.md` (the inventory the audit read),
`docs/oefa_loop_plan.md` §5 and §8 (what the Plan page was built to show),
`docs/oefa_loop_decisions.md` (decisions of record; this plan adds one).

---

## 0. What I verified against the code before planning

The audit was written from the inventory, not the source. Each claim it
depends on was checked; where the code disagrees, the plan follows the code.

| # | Audit claim | Verified | Consequence for the plan |
|---|---|---|---|
| 1 | Polling lives in `Cycle.jsx` and promotes month 1 to an analysis | **True.** `Cycle.jsx:199-219` polls, `:221-227` dispatches `ADD_ANALYSIS`. Also page-local: `Analyzing.jsx:27-67` owns `startCycle`, and the close hands off via `navigate("/analyzing")` (`Company.jsx:232`). | Move all three into an app-level provider **before** deleting the Plan page (Phase A). Otherwise a cycle started from Close would never land. |
| 2 | "No API changes required anywhere" | **True for everything listed.** One promoted item has thin data: "what the board changed" after a real close comes from the *next* cycle's month-1 `adapt.adaptations` (`backend/cycle_service.py:463-466`), which only the rule-based agents fill (`agents/proposal_agents.py:75-78`). The LLM path sees the track record in its prompt but returns no sentence. | The Last-month card needs an honest fallback when the list is empty. A backend addition is optional (decision D4). |
| 3 | History's "accepted n of m" reads pre-commit toggles; check before deleting them | **Worse than the audit thought.** `History.jsx:30` counts `month.decisions` with `state === "accepted"`. Advice's toggle upserts by decision id (`Advice.jsx:123-135`); the close always appends a *new* id (`Company.jsx:198-207`). Toggling on Advice and then closing produces two decisions per domain, and a toggle switched off leaves a `"suggested"` row that inflates the denominator. | Removing the toggles fixes a live bug. History must still render legacy `"suggested"` rows (glyph ○) and exclude them from the denominator (Phase D). |
| 4 | "Dashed = model output" as the honesty convention | **Conflicts with the chart it would live on.** In `FanChart` dashed already means "some simulated runs have died" (`whatif.jsx:12-17`, `:178-185`), and `oefa_loop_plan.md` §5.2 forbids overloading it. | Decision D2: a shaded "projected" region + the IQR band carry "this is the model"; dashed keeps meaning survival. Audit §4/§9 rows that say "dashed" read "shaded". |
| 5 | Delete the month strip, timeline and Plan page | **Contradicts thesis-facing requirements.** `oefa_loop_plan.md` §5.1 asked for the strip with an OEFA strip per column "in a form that screenshots"; §8.4 Act 2 walks that strip on stage and Act 5 watches months land in it. | Decision D1. The plan keeps the two things those acts need — months visibly landing one at a time, and the four OEFA beats one click away — and re-scripts the run of show (Phase G). |
| 6 | Add a frontend test for the honesty checklist | **No frontend test runner exists** (`frontend/package.json` has no devDependencies). The only tests that touch the UI are Python source scans: `tests/test_founder_contract.py:148-200` forbids `ENGINE_ONLY_FIELDS` strings and the infinity glyph in every `frontend/src` file; `tests/test_founder_view.py:139` pins that `founderView.js` reads `RULES.*`. | Decision D3. Every new `.jsx` file is scanned by those tests, so new components must not name engine fields. |
| 7 | Sample mode goes to Plan | `Analyzing.jsx:70` sends demo to `/plan`; Welcome enters demo at `/home`. `SAMPLE` carries two canned cycles with month feedback (`sample.js:392-402`), so an Outlook chart renders offline. | Redirect demo to `/home` in Phase C; nothing else needed. |
| 8 | CSS can be deleted per component | `.plan-head` is defined twice (`styles.css:1212` for `PlanCard`, `:1809-1811` for the Plan page header) and the later rule wins on gap and alignment, so deleting the page rule changes `PlanCard` spacing. History still uses every `timeline-*` class; only `.timeline.forward`, `.timeline-now`, `.timeline-head.static`, `.timeline-entry.projected/.actual` and `.pe-diff` (`styles.css:1892-1900`) are Plan-only. | Phase C lists exactly which selectors go; check `PlanCard` after the `.plan-head` removal. |
| 9 | Put a one-line confidence on the plan header | The sentence is computed server-side with the assumption-count cap (`analysis.display.confidence.sentence`, read by `ConfidenceStrip` `components.jsx:313-318`) and `test_founder_view.py` pins that cap. | Reuse the server sentence verbatim; never shorten or recompose it client-side. |
| 10 | Home already has most of "This month" | Yes: the last-month card (`Home.jsx:24-48`, `:123-136`), the KPI row (`:138-167`) and plan cards (`:169-182`) exist. The cash-death flag needs `months[i].feedback.state_after.survived === false`, already read at `Cycle.jsx:53`. | Phase C is mostly re-composition, not new code. |

### Review corrections folded in (2026-09-23)

A review of the first draft found four must-fixes and seven should-fixes;
all are in the phases below. Two of its supporting claims were checked and
corrected on the way:

- Close must not `await start()` in its own handler (the closure is stale;
  the cycle would be built without the month just closed and without the
  track record). Fixed with `requestStart()` + a provider effect (A.2, C.7).
  The review also said this spared the founder a long wait on the Close
  page; it does not — `POST /api/cycles` returns 202 immediately
  (`backend/main.py:159-170`). The fix is for correctness only.
- The review suggested showing month-1 `weight_moves` as "what the board
  changed". There never are any: `previous_weights` starts as `None`
  (`cycle_service.py:352`) and `_weight_moves(None, …)` returns `[]`
  (`:291-293`). Month-1 `adaptations` are the only reaction signal; the
  fallback sentence is weakened instead (D4, C.3).
- The topbar CTA turns primary on age only (B.5); "not current" and "failed"
  belong to the notice slot.
- Vitest files live outside `frontend/src` (D3), which the Python scan reads.
- Phase D (Why this plan) runs before Phase C so the months 2–4 OEFA beats
  are reachable at every commit.
- History dedupes decisions per domain, last entry wins (D.2).
- `elapsed` derives from `cycle.createdAt`; the poll effect keys on id and
  status (A.3). Shading is done once, in C.5. "Details" points at History
  as a page until F adds per-entry records (C.3). The sim-vs-sim check in §6
  is narrowed to prediction-error sentences.

Second review, same day, two more fixes and three text corrections:

- A closed cycle whose follow-up start failed stays `latestCycle`, and its
  own `months[0].adapt` would be read as the reaction to this close. C.3 now
  reads the adapt block only from a cycle that is not the closed one and was
  planned on the latest month.
- `App.jsx:169` renders under `React.StrictMode`, which runs mount effects
  twice in dev. The pending-start flag is cleared through a ref before
  `start()` runs, and `ADD_CYCLE` / `ADD_ANALYSIS` become idempotent (A.2,
  A.4).
- §3 said the topbar button also goes primary when the analysis is not
  current; it is age-only, as B.5 says. C.10 measured from submit; the
  feedback POST is awaited first with a 60 s timeout (`api.js:120-125`), so
  it measures from the feedback response.
- The review asked to move the `buildPlanCards` check in §6 to after Phase C
  because `Cycle.jsx` would still import it after D. It never did
  (`Cycle.jsx` uses `actionSummary`); today's importers are `Home.jsx`,
  `Advice.jsx` (a multi-line import) and `Company.jsx`, and D removes
  Advice's. The check stays "after Phase D".

---

## 1. Decisions to make before the This-month phase (C); none block A, B or D

| # | Decision | Recommendation |
|---|---|---|
| D1 | Accept the audit's removal of the Plan page even though `oefa_loop_plan.md` §5/§8 built the demo around it? | **Accept.** The audit is right that the strip is a trace viewer in the founder's main path. Keep what the demo needs: months land visibly on This month (Outlook chart and the "next 3 months" list fill in as they arrive, with the elapsed counter), and the OEFA beats sit one click away under Why this plan → "How the board got here", expanded for the month you click. Re-script Acts 2 and 5 (Phase G). Add decision 12 to the decisions doc. |
| D2 | "Dashed = model" vs the FanChart's survival grammar | **Shaded, not dashed.** Everything right of "now" on the Outlook chart gets a light background band labelled "projected"; the IQR band stays the uncertainty grammar; dashed keeps meaning "some runs died". One caveat sentence under the chart. |
| D3 | How to test the honesty checklist | **Add vitest + jsdom + @testing-library/react as devDependencies** and write one small checklist file under `frontend/test/` — never under `frontend/src`, which `test_founder_contract.py:174-192` scans for engine fields and which any cycle fixture would trip (rules-only notice when `llm_ok` is false; estimated markers present; DemoBadge on every route in sample mode; at most one banner on This month and Why; no "the simulation did" sentence outside the trace section). Fallback if you would rather add no dependencies: extend the Python source scans in `test_founder_contract.py` (e.g. `OefaStrip` is imported only by Advice; `Home.jsx` contains no `basis: "simulated"` render). Source scans are brittle; the recommendation is vitest. |
| D4 | The LLM path produces no "what the board changed" sentence | **Ship the honest fallback first.** The "What the board changed" line renders only once month 1 of the new cycle has landed (`months[0].adapt` exists) and only when the close's feedback has a `result` — a failed feedback POST leaves no track record, and `latestClosedFeedback` already hides the whole card then. With `months[0].adapt.adaptations` non-empty, show those sentences. Otherwise, when `cycle.startedFromTrackRecord`, say "The board planned this month with last month's result in hand." (true: the record is in the causal prompt, `agents/causal_proposal_agents.py`) and nothing stronger — month-1 `weight_moves` is always empty (`cycle_service.py:291-293`, `:352`), so there is no other reaction signal on the LLM path. Optional later: ask the causal generator for one `adaptation` sentence per role when a record is present. That is a backend change with prompt and parsing risk — separate branch, not this plan. |
| D5 | Rename routes as well as labels? | **Keep routes** (`#/home`, `#/advice/:id`, `#/update`); change labels only ("This month", "Why this plan"). History deep links and `data/demo_bootstrap.json` keep working; `#/plan` redirects. |
| D6 | Projected risk per month → "Outlook tooltip" | **Skip the tooltip.** `FanChart` has no hover layer; adding one is real work for an L3 fact. Projected risk per month lives only in the trace section. |

---

## 2. Ground rules

- Branch `ui-simplify` from `oefa-loop`; one commit per phase; push at the end.
- UI-only. Backend files are not edited. Run the API from the `founder-api-founder` launch config (no reload) and never edit backend files while a cycle runs (a reload kills the running cycle).
- After every phase: `venv\Scripts\python.exe -m pytest tests/test_founder_contract.py tests/test_founder_view.py -q` and `npm run build` in `frontend`. Never run the broad LLM pytest sweep.
- Engine vocabulary: no `ENGINE_ONLY_FIELDS` string and no infinity glyph in any file under `frontend/src`, including new ones.
- Frontend tests (D3) live in `frontend/test/`, never under `frontend/src`: the Python scan reads every file there and fixtures carry engine fields.
- Stored data from before the change must still render: analyses without `display`, decisions with `state: "suggested"`, cycles without `feedback`, analyses without `cycleId` (pre-loop).
- Every honesty rule in `docs/ui_components.md` §8 keeps one visible surface and one reachable surface; Phase E lists the mapping.
- Copy stays in the founder's vocabulary and comes from `copy.js` / `loopView.js` / server `display` blocks; no new sentence about a prediction is authored inside a page.

---

## 3. Target structure (from audit §7–§8, adjusted by §0–§1)

Navigation: **This month · History · My company · Settings**, plus the topbar
**Close the month** button, which switches to the primary style when the
numbers are older than 35 days, and on nothing else (B.5).

| Route | Screen | Job |
|---|---|---|
| `#/home` (`#/plan` redirects) | This month | Am I OK; what do I do; how did last month go; where does this take me |
| `#/advice/:id` | Why this plan | Why the board recommends it and how far to trust it; all traces two clicks deep |
| `#/update` | Close the month | Did / Partly / Didn't + numbers, one submit |
| `#/history` | History | Trend and past decisions, per-month "how the plan held up" |
| `#/company` | My company | Every number used, exceptions marked |
| `#/settings` | Settings | Preferences, data, engine status (collapsed) |
| `#/analyzing` | Analyzing | Only after onboarding; later runs happen on This month |

This month, top to bottom: notice slot (0–1) · status line · KPI row ·
Last month (conditional) · This month's plan · Outlook. States: no plan yet /
plan being made / cycle failed, all rendered in place.

---

## 4. Phases

Execution order: **A → B → D → C → E → F → G**. A first because deleting the
Plan page while polling lives in it strands cycles. D (Why this plan) before
C (This month): after C the Plan page is gone, and until D's "How the board
got here" section exists the OEFA beats for months 2–4 would be rendered
nowhere; D does not depend on C, so doing it first keeps every commit
complete. E after C and D because the one-notice helper needs both pages'
final shapes. G last because the docs describe the result. The phase letters
are identifiers, not the order.

### Phase A — Cycle start and polling move to app level (state flow only)

Goal: no visible change; a cycle completes and month 1 becomes an analysis
regardless of which page is open.

Files: new `frontend/src/cycleRun.jsx`; `App.jsx`; `store.jsx`;
`pages/Analyzing.jsx`; `pages/Cycle.jsx`; `pages/Company.jsx`.

1. Create `CycleRunProvider` + `useCycleRun()` in `cycleRun.jsx`, mounted
   inside `StoreProvider` in `App.jsx`. It exposes
   `{ start, requestStart, starting, startError, pollError, elapsed }`.
2. `start()` is the body of `Analyzing.run` (`Analyzing.jsx:27-67`): history
   from `state.months.slice(0, -1)`, `previousTrackRecord` from
   `latestClosedFeedback`, `startCycle`, `ADD_CYCLE`. It returns
   `{ ok, offline, error }` and never navigates. It reads the store from the
   provider's current render, never from a closure a caller saved.
   `requestStart()` only sets a pending flag; an effect in the provider calls
   `start()` on the first render in which the flag is set, so a caller that
   has just dispatched (Close: `ADD_MONTH`, `SET_CYCLE_FEEDBACK`) gets a cycle
   built from the state it just changed. Without this the cycle would be
   built without the month just closed and without the track record, and
   the feedback would be silently dropped. The flag is held in a ref and
   cleared synchronously *before* `start()` is called: `App.jsx:169` runs
   under `React.StrictMode`, which invokes mount effects twice in dev, and a
   state-held flag could start two cycles before it cleared.
3. Move the poll effect (`Cycle.jsx:199-219`, `POLL_MS = 2500`,
   `UPDATE_CYCLE`) and the analysis promotion effect
   (`Cycle.jsx:221-227`, guarded by `state.analyses.some(a => a.cycleId === cycle.id)`)
   into the provider. Key the poll effect on `cycle?.id` and `cycle?.status`
   plus `!state.demo` — not on the cycle object, or every `UPDATE_CYCLE`
   tears the interval down. `elapsed` derives from `cycle.createdAt` so it
   survives a reload (today's `started = Date.now() - 0` does not).
4. Store (`store.jsx`): make the two reducers idempotent. `ADD_CYCLE`
   ignores an id already present; `ADD_ANALYSIS` ignores an analysis whose
   `cycleId` (and `monthIndex`) is already stored — analyses without a
   `cycleId` (the pre-loop `/api/advise` path) are appended as before. The
   promotion effect's own guard reads `state.analyses` from the render it
   ran in, so a StrictMode double invocation would dispatch twice before the
   store updated; the reducer guard is what actually prevents the duplicate.
5. `Analyzing` keeps its UI; `run` becomes `start()` then `navigate("/plan")`
   (target unchanged in this phase). `Cycle.jsx` reads `pollError` and
   `elapsed` from the hook and loses its local effects. Close is untouched.
6. Verify: start a cycle, navigate to History immediately, wait, return to
   Plan: four months present, exactly one cycle and exactly one analysis for
   it in localStorage (this is the check that would catch a StrictMode
   duplicate; A.2 and A.4 are what prevent it). Sample mode: no polling.
   Stop the API mid-cycle: the poll error banner appears; restart: it
   catches up.
7. Commit: "Cycle start and polling live at app level, not on the Plan page".

### Phase B — Subtractive cleanup (audit Phase 1)

Files: `pages/Home.jsx`, `pages/Cycle.jsx`, `pages/Advice.jsx`, `App.jsx`,
`components.jsx`, `styles.css`.

1. Home: delete "What changed" (`Home.jsx:184-199`), the evidence peek
   (`:201-211`) and the freshness footer (`:213-222`). Keep the stale note in
   the position banner until Phase C replaces the banner with the status line.
2. Plan page: delete the loop lines (`Cycle.jsx:252-262`, `:322-324`), the
   "N months, planned together" header copy (`:290-306`, keep only the Re-run
   button and the running counter), the per-month fresh/reused chip (`:61`).
   `OefaStrip` `defaultOpen` false everywhere (`Cycle.jsx:74`, `Advice.jsx:181`).
3. Advice: delete the position banner (`Advice.jsx:170-175`) and the Next
   actions checklist (`:295-323`). Cards and toggles stay until Phase D.
4. Shell: delete the sidebar status block (`App.jsx:133-137`); `DemoBadge` in
   the topbar already satisfies rule 7.
5. Topbar Close button (`App.jsx:145-152`): `primary-button` when
   `daysSince(latestMonth.enteredAt) > 35`, and only then. This lands in B
   because B.1 deletes the freshness footer that carries the nudge today.
   "Analysis not current" and "cycle failed" are not conditions for it: the
   first is true from the moment a founder closes until month 1 lands, and
   the second wants Re-run, not Close. Both belong to the notice slot.
6. CSS: `.loop-lines`, `.freshness-footer`, `.evidence-peek`, `.changed-panel`,
   `.sidebar-status`, `.checklist-panel`, `.checklist`, `.check-item`.
7. Regression guard: the rules-only signal survives on the Plan banner
   (`Cycle.jsx:283-288`) and the Advice banner (`:161-167`).
8. Commit: "Remove the duplicate summaries: what-changed, evidence peek, footer, loop lines, checklist, sidebar status".

### Phase C — This month (Home absorbs Plan) — runs after D

Files: `pages/Home.jsx` (rewrite as a composition of existing parts), new
`frontend/src/outlook.jsx`, `components.jsx`, `App.jsx`, `pages/Analyzing.jsx`,
`pages/Company.jsx`, `pages/Advice.jsx` (one link), `styles.css`; delete
`pages/Cycle.jsx`.

1. Sections in order: notice slot (this phase still renders the existing
   banners; Phase E replaces them with `pickNotice`) · status line · KPI row ·
   Last month · This month's plan · Outlook.
2. Status line: `RiskChip` + `positionSentence` (from `Home.jsx:86-106`) +
   "Based on your {monthName} numbers · {n} days ago" (from the Plan stale
   banner `Cycle.jsx:266-275`). Static, not a button. The topbar rule stays
   B.5's `daysSince > 35` only. The Home stale banner (`Home.jsx:108-121`)
   goes; "the plan below is on your previous numbers" becomes one inline
   sentence at the top of the plan section until Phase E's notice helper.
3. Last month card: `predictionErrorLines` (`Home.jsx:24-48`) + `scoreLine`
   + a "What the board changed:" line. The card itself is shown only when the
   closed cycle was planned on the previous month and has a `result`
   (existing `latestClosedFeedback` rule, so a failed feedback POST shows no
   card and no "changed" line). The "changed" line follows D4 and reads only
   from the cycle that answered this close: `latestCycle(state)` must not be
   the closed cycle and its `monthId` must be the latest month's id. If the
   close succeeded but the follow-up start failed, the closed cycle is still
   the latest one and its own `months[0].adapt` (from its earlier run) would
   otherwise be shown as the reaction to this close; in that case there is
   no "changed" line at all. With the right cycle: nothing while it is
   still running; once `months[0].adapt` exists, its `adaptations` if any;
   else, when `startedFromTrackRecord`, "The board planned this month with
   last month's result in hand." "Details" links to
   `/history` as a page; the per-entry record arrives in Phase F.
4. This month's plan: header = the server confidence sentence (as
   `ConfidenceStrip` reads it) + "Why this plan →" to `/advice/{analysis.id}`;
   full `PlanCard`s (not `compact`) for `isAction` cards; one line
   "Holding: hiring, pricing" from the non-action cards' titles; when nothing
   is an action, the "Nothing to change this month" panel from
   `Advice.jsx:190-206`; when any `months[i].feedback.state_after.survived === false`,
   one red line "In simulation this plan runs out of cash around month N".
5. Outlook (`outlook.jsx`): `TrajectoryChart` (`Cycle.jsx:86-129`) reduced to
   **one** `FanChart` with a metric switch Revenue / Cash / Customers lost;
   default metric: Cash when the runway tile is on watch, Revenue when the
   efficiency tile is on watch, else Cash. Opening point =
   `months[0].observe.state_before`; a closed month 1 plots its
   `actual_state` as a marked point. Projected region shaded per D2. One
   caveat line under the chart (the audit's wording). Collapsed "Next 3
   months of the plan": `actionSummary` per month 2..H (`Cycle.jsx:65-67`),
   filling in as months land.
6. States: **no cycle** → sections 1–3 + "Run the plan" (copy from
   `Cycle.jsx:235-249`); **running** → sections 1–3 + `ProgressStages`
   (`stage = min(monthsLanded, 2)`) + "Month k of H · m:ss" from the hook,
   then the plan section and Outlook appear as month 1 lands and grow per
   month; **failed** → warn notice with the engine's reason + Re-run
   (`Cycle.jsx:276-281`); **start failed** (`startError`) → notice with Retry
   and "Continue without a plan" (from `Analyzing.jsx:76-99`). Run / Re-run /
   Retry call `start()` and stay on `/home`.
7. Routing: `NAV` → four items with label "This month"; `TITLES`;
   `parseRoute`: `plan` → `home`, `/advice` without id → `home`.
   `Analyzing.run` → `navigate("/home")`; demo redirect (`Analyzing.jsx:70`) →
   `/home`; Advice "Current plan" (`Advice.jsx:155`) → `/home`; Close submit
   (`Company.jsx:232`): after the `SET_CYCLE_FEEDBACK` dispatch, call
   `requestStart()` then `navigate("/home")`. The provider's effect starts
   the cycle from the state that already holds the new month and the
   feedback result. Never `await start()` inside the handler: `start` as
   captured at the previous render would build the cycle without the month
   just closed and without the track record. Analyzing is now only the
   onboarding wait screen; its failure card copy moves to the start-failed
   notice.
8. Delete `Cycle.jsx`. CSS to remove: `.month-strip`, `.month-col*`,
   `.month-state-grid`, `.month-action-lines`, `.month-dead`, `.plan-head`
   at `:1809-1811` (then check `PlanCard` spacing), `.timeline.forward …`,
   `.timeline-now`, `.timeline-head.static`, `.timeline-entry.projected/.actual`,
   `.pe-diff`, `.wi-grid.three`, `.stale-note`, `.plan-compact-grid`; the
   `.position-banner` button styles become a static line.
9. Keep: `analysisFromCycle`, `feedbackForCycleMonth`, `cycleForMonth`,
   `actionSummary`, `briefFreshness` (trace only from here), `predictionSentences`.
10. Verify with three fixtures. Seeded demo (Settings → load): Last month
    card reads "we were N% high" with the adaptation or fallback line; plan
    and Outlook present; `#/plan` redirects. Sample company (API off):
    Outlook renders from the canned cycles, no polling, no Close button.
    Fresh onboarding: Analyzing → This month in the running state → months
    land in the Outlook and the next-3 list → close the month → This month in
    the planning state → new cycle → Last month card appears. Confirm the
    new cycle's `POST /api/cycles` body carries `previous_track_record` and
    the stored cycle has `startedFromTrackRecord: true`; the founder is on
    This month within a second of the feedback response (the close awaits
    that POST first, `api.js:120-125`, 60 s timeout; the cycle start itself
    is the 202 and is not awaited on that page). Stop the API and Re-run: the
    notice with Retry. Scans + build.
11. Commit: "This month: Home absorbs the Plan page; one outlook chart".

### Phase D — Advice becomes Why this plan — runs before C

Files: `pages/Advice.jsx`, `components.jsx` (`PlanCard`, `OefaStrip`),
`whatif.jsx`, `pages/History.jsx`, `styles.css`.

1. Order: notice slot (archived or rules-only, one) · Summary (top-focus
   sentence from `FOCUS_LABELS`, the server confidence sentence, and
   `refreshReasonCopy` for why the analysis ran) · Watch-outs / Working in
   your favour (`RiskBullets`) · Plan vs doing nothing (`WhatIfPanel`) ·
   collapsed **Evidence** (`EvidenceList` + the OEFA Observed lines +
   `expected_outcome` copy with its `SimulatedTag`) · collapsed
   **Assumptions** ("Numbers we guessed" list + "Fill these in" + the what-if
   assumption list when loaded + the "normal market conditions" line) ·
   collapsed **How the board weighed it** (`FocusBar` + `reasoningBullets`) ·
   collapsed **How the board got here** (one `OefaStrip` per horizon month
   from `cycleById(analysis.cycleId).months`, month-1 `closed` feedback, the
   loop lines from `cycle.summary`, weight moves; `monthFromAnalysis` for
   pre-cycle analyses).
2. Remove the plan cards, `decide` and the toggles (`Advice.jsx:120-135`,
   `:186-218`); `SET_DECISION` stays in the store for Close. History
   (`History.jsx:30-31`): dedupe decisions per domain keeping the **last**
   entry in array order (Close appends, so its answer wins over an earlier
   toggle), then count `accepted` over the deduped list and exclude
   `"suggested"` from the denominator. Old months that already hold both a
   toggle row and a close row for one domain are thereby fixed without a
   migration; legacy rows still render with ○. The "Current plan" button
   keeps pointing at `/plan` until C.
3. `OefaStrip` (`components.jsx:452-518`): drop the header chips (`:472-473`);
   when `basis === "simulated"` the Changed beat is labelled "Changed (in
   simulation) — model consistency check, not accuracy"; drop the `compact`
   prop and its CSS.
4. `WhatIfPanel` (`whatif.jsx:232-441`): the death note, table and server
   caveat stay visible; legend, four charts, shock toggle, meta and
   assumptions move under one "Show the charts" expander. The API call and
   its guards are untouched.
5. `PlanCard` (`components.jsx:215-253`): remove the accept branch
   (`:234-243`) and the `decisionState`/`onDecide` props; "% of MRR" (`:225`)
   moves into the "Why this number?" expansion. CSS `.accept-button`.
6. Verify: History deep link to an archived analysis; an analysis with
   `llm_ok === false` shows one notice; a pre-cycle analysis still renders the
   trace section via `monthFromAnalysis`; sample company; what-if still runs
   and the shock toggle still re-runs it.
7. Commit: "Why this plan: reasons, evidence, assumptions and the loop trace, no second copy of the actions".

### Phase E — Honesty grammar and the one-notice rule

Files: new `frontend/src/notice.js`, `components.jsx` (`ProvChip`,
`SimulatedTag` uses), `pages/Company.jsx`, `outlook.jsx`, `pages/Home.jsx`,
`pages/Advice.jsx`; tests per D3.

1. `pickNotice(ctx)` returns at most one of, in priority: engine unreachable /
   cycle failed / start failed (with the action) · sample company (only on
   Close, where the form is disabled) · rules-only (whole plan, or partial:
   "Months 3–4 used built-in rules" from `months[i].execute.llm_ok`) ·
   archived analysis (Why only). Conditions that lose the slot render as one
   inline sentence in their section.
2. `ProvChip` (`components.jsx:35-43`) renders only for `estimated`, `derived`,
   `simulated`; Company rows drop `chip="provided"` (`Company.jsx:53-78`) and
   the section gets one "From your {month} close" line; the footer banner
   becomes one sentence.
3. `SimulatedTag` only inside Evidence. One caveat per page: This month's is
   the Outlook line; Why's is the what-if `result.caveat` from the server.
4. Checklist tests per D3 (vitest, in `frontend/test/`): rules-only notice
   when `llm_ok` is false; estimated markers present on estimated values;
   `DemoBadge` on every route in sample mode; at most one `.banner` on This
   month and Why; the sentence "the simulation did" appears only inside the
   trace section. (The Outlook shading landed in C.5; nothing here.)
5. Honesty rule → surface map to keep true: (1) failure / rules-only in the
   notice slot, engine status in Settings; (2) marker on every non-founder
   value, full list in Assumptions and My company; (3) confidence sentence on
   the plan header and the Why summary; (4) shaded projection + one caveat;
   (5) Last month card directly under the KPIs; (6) Close form at the moment
   of choice and the Last month card; (7) `DemoBadge` in the topbar.
6. Commit: "One notice per page; markers only on exceptions; one caveat per page".

### Phase F — History, Close, Company, Settings, Onboarding polish

1. History: per-entry collapsed "How the plan held up" from
   `cycleForMonth(state, m.id)` + `feedbackForCycleMonth` → `predictionSentences`,
   `scoreLine`, adaptations; the 6-months-later badge (`History.jsx:55`) moves
   inside the expansion; decisions line reads "Did n of m · partly k".
2. Close (`Company.jsx:142-283`): the optional fields of `UPDATE_FIELDS`
   (price, new customers, marketing spend) move into a collapsed "Numbers we
   estimated — replace them if you know them" group when the latest analysis's
   `trace.assumed_fields` marks any of them correctable; "Fill these in" on
   Why deep-links here. No new fields.
3. Settings: connection line + capability list (`Settings.jsx:68-109`) into one
   collapsed "Engine status".
4. Onboarding: optional fields per step collapsed behind "Add more detail —
   raises confidence" (read `Onboarding.jsx` step structure first).
5. KPI sub-lines ("you spend $X per $1", payback) into the `hint` title
   attribute; Outlook default metric already follows the watch band (C5).
6. Skip the aggregated accuracy line (audit P2.7) unless it replaces the
   per-month score lines; it does not.
7. Commit: "History carries the past-plan record; Close, Company, Settings, Onboarding polish".

### Phase G — Documents, demo script, memory

1. Rewrite `docs/ui_components.md` to the new inventory, same section
   structure (it is the input to the next audit).
2. `docs/oefa_loop_decisions.md`: add **decision 12** — the Plan page is
   folded into This month, the OEFA trace lives under Why this plan, the
   projection convention is shaded not dashed, and why (audit §1 items 3, 4
   and 7; §0 rows 4 and 5). Add one line at the top of `oefa_loop_plan.md` §5
   saying it is superseded by this plan; do not rewrite the history.
3. `docs/oefa_loop_plan.md` §8.4: re-script Act 2 (This month → Outlook →
   Why this plan → How the board got here, month 1 expanded) and Act 5 (watch
   the Outlook chart and the next-3 list fill in on This month with the
   elapsed counter).
4. `docs/founder_frontend_spec.md`: add a dated addendum pointing at
   `ui_components.md` for the post-simplification inventory rather than
   editing §8–§17.
5. README loop section: the nav names. Update the memory note for this
   branch. Push `ui-simplify`.

---

## 5. Verification matrix (after every phase)

| Check | Command / fixture |
|---|---|
| Source scans | `venv\Scripts\python.exe -m pytest tests/test_founder_contract.py tests/test_founder_view.py -q` |
| Bundle builds | `cd frontend; npm run build` |
| Sample company, API off | Welcome → Explore sample: every route renders, no polling, Close disabled with its banner, `DemoBadge` everywhere |
| Seeded demo, API on | Settings → Load the seeded demo company: Last month card, plan, Outlook, Why → trace for months 1–4, History per-entry record |
| Fresh company, API on | Onboarding → Analyzing → This month running → months land → Close → planning state → new cycle whose request carries `previous_track_record` (`startedFromTrackRecord: true`) |
| Failure states | Stop the API: start-failed notice with Retry; stop Ollama: rules-only notice, one per page |
| Loop tests untouched | `tests/test_cycle_api.py`, `test_loop_plumbing.py`, `test_expected_delta.py` still pass (no backend edits, so a single run at the end is enough) |

Never: the broad LLM pytest sweep; backend edits during a live cycle.

---

## 6. Audit success criteria → how each is checked

| Criterion (audit §13) | Check |
|---|---|
| One primary surface per concept | The §4 matrix of the audit, re-read against `ui_components.md` after Phase G |
| This month ≤ 6 sections, ≤ 1 banner, 0 open expanders, ≤ 1 caveat | Vitest checklist + a screenshot at 1280×800 with the "next 3 months" collapsed |
| ≤ 1 dominant CTA per page | Visual pass per page; topbar Close is the only primary button on This month when due |
| Chips only on exceptions | `ProvChip` renders nothing for `provided`; grep for `chip` classes on This month |
| Actions rendered in exactly two places | `buildPlanCards` is imported only by `Home.jsx` and `Company.jsx` from Phase D onward (`Cycle.jsx` never imported it; D removes Advice's multi-line import) |
| Traces, evidence, assumptions, weights, what-if charts ≤ 2 interactions from This month | This month → Why this plan (1) → expander (2) |
| No sim-vs-sim number on L1/L2 | Prediction-error sentences with a simulated basis ("the simulation did …") render only inside "How the board got here"; the Outlook legitimately draws simulated state, so the check is on the sentences, not on the data |
| Seven honesty rules pass per page | Phase E.4 tests |
| Close ≤ 3 minutes; after submit the founder lands on This month and sees the plan build | Phase C.10 fresh-company run, timed |

---

## 7. Left alone on purpose

- "Cash lasts" ignores the 83.5 % gross margin the physics apply, and the
  hiring-runway guard ignores the plan's own spend (decision 9 caveats).
  Numbers, not layout; separate change.
- The CMO rationale citing an `OBSERVED_WITH` co-occurrence (pre-existing).
- Two History entries for one calendar month when a close happens in the
  same month as the last entry; the timeline that also showed it goes in
  Phase C, History keeps the index-based order.
- The comprehension tests with founders (audit §13) are not something this
  plan can run; the structural criteria above are what I can verify.

## 8. Size estimate

| Phase | Rough change |
|---|---|
| A | ~150 lines moved into `cycleRun.jsx`, no net UI change |
| B | ~−150 lines |
| C | new `Home.jsx` ~250, `outlook.jsx` ~120, `Cycle.jsx` −344, CSS −120 |
| D | `Advice.jsx` rewrite ~250, `components.jsx` −40, `whatif.jsx` +30 |
| E | `notice.js` ~60, edits ~80, tests ~120 + devDependencies |
| F | ~150 across four pages |
| G | docs only |
