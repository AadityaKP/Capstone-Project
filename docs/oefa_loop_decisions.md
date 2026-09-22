# OEFA loop — decisions of record

Companion to `docs/oefa_loop_plan.md`. The plan asks for several choices to be
made explicitly and written down rather than left as defaults, because each is
a claim the thesis makes and a reviewer will ask about. This file is that
record. Branch `oefa-loop`, 2026-09-22.

## 1. State ownership — option (a)

Cycles carry their own actuals in `cycle_feedback`; the browser stays the
founder's record of months. The Oracle's per-company calendar (`global_month`,
pending memories, recent snapshots) is persisted server-side in
`company_oracle_state` so the pending queue no longer dies with the request.
`cycle_feedback` is shaped so that moving months server-side later is a
migration into `company_months`, not a rewrite.

## 2. Memory scope — one key per company

`run_id = "company:{company_id}"` for both profiles (`sim_profile.get_memory_scope`).
Under `founder` the store is the isolated `chroma_db_founder`; under `review2`
the Oracle still builds its store against `CHROMA_PATH` with the research
prompt template unchanged, but a second analysis of the same company can now
retrieve what the first matured. A request without a company id keeps a UUID:
pooling anonymous requests would be worse than the old isolation.

A month observed twice (a re-analysis of the same numbers) replaces the earlier
observation instead of appending a duplicate (`Oracle(dedupe_months=True)`,
product path only).

## 3. Evidence provenance — namespace and weight

Simulated and observed outcomes do not share an edge:

| source     | relationship     | increment (+ / −) | promotable to `CONFIRMED_CAUSE` |
|------------|------------------|-------------------|---------------------------------|
| `observed` | `MAY_CAUSE`      | +0.05 / −0.03     | yes (≥ 0.85 and ≥ 3 positive)   |
| `sim`      | `MAY_CAUSE_SIM`  | +0.02 / −0.012    | never                           |
| `None`     | `MAY_CAUSE`      | +0.05 / −0.03     | yes — the legacy research path, byte-identical |

Why 0.4×: a cycle writes four simulated months against one real month per
month. On a shared edge at equal weight the simulator's opinion would outweigh
reality four to one, and the board would cite its own physics back to itself.
Namespacing means simulated evidence can inform the proposal prompt (the
predicate name says what it is) but can never become a "confirmed" cause and is
never rendered to the founder as something that happened.

Partial credit in the HITL close (`partly`) scales the increment by 0.5 and
never changes which edge is written. `didn't` writes no causal edge at all.

Promotion is unchanged from the research graph: five positive observed
observations minimum from a 0.6 base. One confirmed month does not promote an
edge, and the UI says so.

## 4. `oracle_frequency = 0` inside a cycle

`FOUNDER_ORACLE_FREQUENCY = 1` makes the cadence branch fire every month, so
the event triggers never get a turn. The cycle runs the Boardroom with
`oracle_frequency=0`: the brief refreshes on `initial` and on `event` (runway
< 12 months, MRR down 15 %, churn up 1.5 pp, confidence down 15, unemployment
up 2) and is otherwise reused. `brief_source` (`llm` / `cache_hit` / `reuse`)
travels per month and is shown as fresh vs reused.

## 5. Who decides, and whose prediction it is

**The language model chooses the action on the causal path; the simulator
makes the prediction.** `expected_delta` is the same physics that runs the
what-if projection, rolled forward under the proposed action over a small set
of seeds, reported as the median per-KPI change at a two-month horizon with a
p25/p75 band. It is attached to every proposal (each lever alone, against a
no-spend base) and to the final action (what Feedback and the HITL close score).

The LLM is not asked for numbers because it has no calibrated numeric sense;
a checkable prediction is the whole point. The rule-based agents are the
fallback path and are labelled as such via `proposal_source`.

Two honesty notes carried in the payload: `basis="simulated"` (inside a cycle
the realized month is also simulated, so in-cycle error is seed noise plus the
compounding of the model's own physics; against a founder's real month it is
the real number, and that is what the thesis reports), and the band is the
only projection-uncertainty grammar on the Plan page.

## 6. Track record — how the agents' reasoning changes

Every agent receives what the board proposed last month, what it predicted,
and what happened (`boardroom.expectation.build_track_record`). The causal
generator's prompt shows it before the model chooses. The rule-based agents
apply one documented rule: when last month's prediction for the role's own KPI
(CFO: cash, CMO: MRR, CPO: churn) missed on **direction**, the role pulls its
lever back toward hold by 25 % (CFO: hiring waits) and says so in one sentence
(`Proposal.adaptation`). A miss on size alone is calibration and changes
nothing. Research arms never receive a record and are untouched.

## 7. Never call `end_episode()` at cycle end

It force-matures every pending memory against the latest snapshot regardless
of horizon, which would label 4-month evidence with the 6-month ±10 % rule into
the store the thesis reads. Memories mature from real HITL months instead,
which genuinely arrive a month apart.

## 9. The founder prediction runs the fitted marketing curve (closed)

Found while seeding the demo: the founder profile's projection physics used
the customer/CAC-anchored marketing curve with the *assumed* saturation rate
0.20 (`scale_aware_marketing`), the constant physics_v2 falsified when it
fitted 0.0727 (CI 0.0475–0.1113) on the CAL panel. Both curves were tested
side by side before deciding (2026-09-22):

| evidence | assumed 0.20 | fitted 0.0727 |
|---|---|---|
| CAL median 4q growth error (real companies, `calibration_report.md` §4) | 49.7 pp | 12.9 pp |
| HOLDOUT (one touch) | falsified | 8.1 pp, 100 % sign agreement, PASS |
| sample company, 10 % of MRR on marketing, 2-month MRR prediction | +16.9 % | +8.4 % |
| same, 20 % of MRR | +19.7 % | +9.1 % |
| seeded history, harness mean revenue error (synthetic, illustrative) | 20 pp high | 4 pp high |

With no marketing the curves agree, so the whole gap is the marketing
response: the assumed rate predicts roughly twice the fitted one at founder
scale, which would have made every close-the-month read "we were high".
Decision: `FOUNDER_MARKETING_CURVE` defaults to `v2`; `scale_aware`
reproduces the pre-loop product. Research runs never read `sim_profile`.

Three consequences were handled with the switch. (1) Two what-if tests had
pinned the assumed curve's growth (a fixture's recommended arm surviving,
another's hold arm growing on $10 of marketing); they now assert what they
were written to protect, that ragged paths are handled and twelve distinct
months are simulated. (2) The demo company's cost line was chosen so it
survives its own plan under the fitted physics. The board's plan for a
company this size is ~$17k of monthly spend plus one hire, which at the
83.5 % gross margin burns ~$27k a month on top of the cost line; at ~$33k
MRR, $30k of costs with $350k of cash keeps that plan alive in every run
with ~$100k left after twelve months, where the previous $47k on $200k died
in every run under *either* curve. (3) The canned sample company was moved
onto the same footing.

Found on the way: the what-if projection repeated the whole plan every
month of its horizon, including the hire, so a plan with `hires: 1` hired
twelve people in a year (each adding a salary slot to burn) and killed
every company the board suggested one hire to. A hire is a one-off; the
projection and the `expected_delta` rollout now apply it in month 1 only
and let the state carry its payroll (`whatif_service._without_hires`). The
assumptions list on the projection says so.

Caveats that stay: neither curve has founder-scale data behind it (the fit
is from companies at $10M+ revenue, and a small startup can plausibly add
more than 7 % of its customer base in a month), so the HITL harness is the
instrument that will settle it. The UI's "cash lasts" figure ignores the
83.5 % gross margin the physics apply, which is why a projection can look
harsher than the KPI tile; not changed here.

## 10. The demo runs the founder profile

`SIM_PROFILE` defaults to `review2` for Review 2 parity, under which the
advisor mode is `oracle_v3`, no causal graph exists, the research physics run
unscaled at founder size, and every founder-facing guard is off. The loop's
Feedback step therefore cannot be live under the default. `start.ps1` sets
`SIM_PROFILE=founder` unless told otherwise (`-Profile review2`), and
`/api/health` names the profile so the claim on stage matches the process.

## 11. A restart fails the cycles it killed

Cycles run on a thread inside the API process. A restart (uvicorn `--reload`
on a code edit, a crash) killed the thread silently and left the row
`running`, with the client polling a plan that would never land. On startup
every `queued`/`running` cycle is marked `failed` with a reason, and the Plan
page shows it and offers a re-run.

## 8. Memory horizon versus cycle length

`MEMORY_HORIZON_MONTHS = 6` against a 4-month cycle means memories written in
one cycle mature during the next — now true because the pending queue is
persisted (decision 1). Within-cycle Adapt runs off weights, the track record
and the causal graph, not off matured episodic memory.
