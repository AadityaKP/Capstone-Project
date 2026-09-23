# Demo runbook — running the founder product for a review

How to bring the whole stack up, what to load, exactly what to type, and
what each screen shows when you do. Every number and every quoted line
below was produced by a live run on 2026-09-23 against the `ui-simplify`
branch with Ollama (`llama3.1:8b`) and Neo4j up, engine profile `founder`.
Where something was *not* verified live it says so.

Companion documents: `docs/ui_components.md` (what every screen contains),
`docs/founder_testing_runbook.md` (the longer testing guide; its Part 3
screen descriptions predate the simplification), `docs/oefa_loop_plan.md`
§8 (why the demo is shaped this way).

---

## 1. What runs

| Piece | What | Port | Started by |
|---|---|---|---|
| API | FastAPI + SQLite (`backend/`), runs the cycles on a thread | 8000 | `start.ps1` |
| Frontend | Vite dev server, or the built bundle served by the API | 5173 (dev) / 8000 (`-Prod`) | `start.ps1` |
| Ollama | the strategist and the causal proposals, `llama3.1:8b` | 11434 | service; `start.ps1` starts it if it can |
| Neo4j | the causal evidence graph | 7687 | service (optional; without it there is no evidence line) |

The API needs `SIM_PROFILE=founder`; `start.ps1` sets it. Under the
default research profile the loop's Feedback step cannot run.

## 2. Start it

One command, from the repo root, in PowerShell:

```powershell
.\start.ps1 -Prod
```

`-Prod` rebuilds `frontend/dist` and serves the app and the API from one
port, so there is no proxy and no stale bundle. Open
`http://localhost:8000`. If a page ever looks like an older build, hard
reload (Ctrl+Shift+R).

For working on the UI use the default dev mode instead:

```powershell
.\start.ps1
```

and open `http://localhost:5173`. Vite proxies `/api` to 8000. Two dev-mode
gotchas: an edit to any frontend file reloads the page, which drops the
in-memory sample company; and never pass `-Reload` for a demo, because a
backend edit would restart the API and kill the cycle in flight.

If the script refuses to start because a port is held: `.\start.ps1 -Prod -Force`.

The console should end with `api ready - advisor mode: oracle_v4_causal`
and `Open http://localhost:8000`. Confirm the loop is live:

```powershell
curl http://127.0.0.1:8000/api/health
```

Expected in the `loop` block (verified): `sim_profile: "founder"`,
`marketing_curve: "v2"`, `llm_reachable: true`, `graph_store_enabled: true`,
`memory_store_enabled: true`, `loop_live: true`. Settings → **Engine status**
shows the same three capabilities in words.

Optional, before anyone is watching (makes the two silent failure modes loud):

```powershell
venv\Scripts\python.exe experiments\loop_live_check.py
```

## 3. Load the demo company (T-10 minutes)

The demo must open on a company that already has history: a first cycle
has no memory, no prediction error and nothing learned. The seed script
creates **Acme Analytics** with seven months of history and two completed
cycles on the engine, the first of them closed with real numbers.

If `data/demo_bootstrap.json` exists and you have **not** run the demo
against it yet, skip the seed. Otherwise, with the stack up:

```powershell
venv\Scripts\python.exe experiments\seed_demo_company.py
```

About two minutes with Ollama (two 4-month cycles). It writes
`data/demo_bootstrap.json`, which the API serves at `/api/demo/bootstrap`.

Then in the app: **Settings → Load the seeded demo company → Replace with
the seeded company**. You land on This month.

The company's current numbers (September 2026): MRR $33,600 · cash $350,000
· costs $30,000 · price $85 · churn 4.4 %/mo · 46 new customers ·
marketing $5,000. Chosen so the board's own plan survives twelve simulated
months under the fitted physics; a structurally dying company teaches
nothing about the loop.

## 4. Run of show (about 10 minutes)

Four navigation items: **This month · History · My company · Settings**.
Everything the reviewer needs is on This month; every trace is two clicks
from it.

### Act 1 — This month (60 s). "We were 5 % high, and here is what we did about it."

What is on screen (verified), top to bottom:

- No banner. Status line: **Low risk** — "You're in good shape — growth is
  accelerating. This month: protect retention and product momentum." ·
  "Based on your September 2026 numbers · 22 days ago".
- KPI row: Cash lasts **Not burning** (revenue covers costs) · Revenue
  **$34k** (+5.3 %) · Customers lost **1 in 23** (−0.2pp) · Winning
  customers **Healthy**. The definitions are in each tile's hover hint.
- **How last month's plan held up**:
  "We projected $35k of revenue, you did $34k — we were 5% high." ·
  "Churn: the board expected 0.0pp, it moved −0.2pp — the wrong direction."
  · "Cash: expected $299k, ended at $350k." · "You didn't do the pricing
  step, so we're not counting this month as evidence about it." · "2 of 3
  predictions moved in the right direction; 0 landed within tolerance." ·
  "What the board changed: The board planned this month with last month's
  result in hand."
- **This month's plan**, header "Low confidence - 6 of these numbers are
  estimates, not yours" with **Why this plan →**. Four cards: Product &
  retention (Priority) **Invest ≈$14k in product this month** · Marketing &
  growth **Spend ≈$8.0k on performance channels** (up from the ≈$5.0k you
  reported) · Hiring **Room to add ≈$8.0k/mo of payroll** · Pricing
  **Consider a ≈1% price increase**.
- **Where this plan takes you**: one chart, Cash selected, the founder's
  $350k on the left, everything to its right on the shaded **projected**
  band, ending around **$181k** at Jan 2027. One caveat line. "Next 3
  months of the plan" collapsed.

Two sentences to say. *This system makes a prediction, finds out whether it
was right, and changes.* And, on the confidence header: *the board is only
as sure as its inputs; six of the numbers behind this plan are the engine's
own defaults (interest rate, consumer confidence, and so on), so the
confidence band is capped at Low however sure the model claims to be.*
That sentence comes from the server verbatim and the cap is test-pinned.

"Not burning" is correct for this company: revenue exceeds the cost line.
The projection below shows what the *plan's* spend does to cash.

### Act 2 — Outlook → Why this plan (2 min)

1. Switch the Outlook metric to **Revenue** and back to **Cash**. Open
   **Next 3 months of the plan**: each later month's four action lines,
   each a link.
2. Click **Why this plan** in the plan header. Verified screen: "The
   board's top focus is Product." with the confidence sentence and "your
   first analysis"; Watch-outs ("Churn rate slightly above benchmark",
   "High CAC compared to LTV") and Working in your favor ("Increasing MRR
   trend", "Decreasing churn rate") — the strategist's own words, guard-
   railed against numbers the founder never typed.
3. Press **Run the projection** (under a second, no LLM). Verified table:

   | Plan | Revenue in 12 mo | Cash in 12 mo | Survives (of simulated runs) | Spend per $1 |
   |---|---|---|---|---|
   | Take the board's plan | $49k | $49k | 100 % | $1.22 |
   | Keep doing what you're doing | $45k | $332k | 100 % | $0.78 |

   Say it straight: *the board's plan buys about $4k of monthly revenue for
   roughly $280k of cash over a year, and survives every run; doing
   nothing keeps the cash. The product shows both, labelled as a simulated
   counterfactual, and lets the founder decide.* The caveat under the table
   is the server's. "Show the charts" opens the four fan charts and the
   competitor-shock toggle if anyone asks.
4. Open **How the board got here**. Verified: "3 fresh strategist reads, 1
   reused · what happened each month was written back as simulated
   evidence · memory scoped to your company · 33s of deliberation", then
   four strips (Sep–Dec 2026), September open. Walk its four beats:
   Observed ("1 similar past month recalled · see Evidence above") ·
   Decided (Marketing ≈$8.0k · Product ≈$14k · Hire 1 · Price +1%) ·
   Expected ("revenue +11.1%, churn 0.0pp, cash −19.5% over 2 months") ·
   **Changed (in simulation)** — "model consistency check, not accuracy"
   — "We projected $37k of revenue, the simulation did $36k — we were 5%
   high" and the score line. Point out that this beat is the simulator
   scoring itself; the founder's real numbers arrive in Act 3.
5. If asked what the model knows: **Evidence** (the recalled month as a
   founder sentence, the causal-graph line marked as a built-in prior) and
   **Assumptions** (the six engine defaults, and the projection's own
   assumptions once it has run).

### Act 3 — Close the month (3 min). The human-in-the-loop moment.

Topbar → **Close the month**. The form lists the four actions from the plan
above. Enter exactly this:

| Action | Answer | Note |
|---|---|---|
| Product & retention — Invest ≈$14k | **Did it** | |
| Marketing & growth — Spend ≈$8.0k | **Partly** | `Did $6k, not $8k` |
| Hiring — Room to add ≈$8.0k/mo | **Didn't** | |
| Pricing — Consider a ≈1% increase | **Did it** | |

| Field | Enter |
|---|---|
| Monthly recurring revenue | **35200** |
| Cash in the bank | **338000** |
| Total monthly costs | **30000** |
| Monthly churn | **4.2** |
| New customers last month | **49** |
| Marketing spend last month | **6000** |
| Average price | **86** |

The diff pills read **MRR +4.8% · churn −0.2pp · cash $−12k**. On the
*Didn't* row the form says "You didn't do this, so this month won't count
as evidence about it" — say the quiet part: *the board will learn nothing
from that one, because it never happened.*

Press **Close the month & plan again**. Verified: the form disables and
says "Scoring last month's plan…", and about **3 seconds** later you are on
This month with the status line reading "· new plan in progress", the
plan section replaced by **Your board is planning** with the staged list
and "Month 1 of 4 · 0:00", and the Last-month card already rewritten from
the close you just made:

"We projected $37k of revenue, you did $35k — we were 6% high." · "Churn:
the board expected 0.0pp, it moved −0.2pp — the wrong direction." · "Cash:
expected $282k, ended at $338k." · "You didn't do the hiring step, so we're
not counting this month as evidence about it." · "2 of 3 predictions moved
in the right direction; 0 landed within tolerance."

### Act 4 — Watch the plan build (about 40 s)

Keep talking. Verified timings with `llama3.1:8b`: month 1 lands in about
20 s; the whole four-month cycle completes in about 40–45 s. You can
navigate to History and back while it runs — the cycle lives in the app,
not on the page.

As month 1 lands the plan section fills in: Product **≈$14k** · Marketing
**≈$3.5k on performance channels** (down from the ≈$6.0k you reported) ·
Hiring **≈$8.0k/mo of payroll** · Pricing **≈1%**. The Last-month card
gains **"What the board changed: The board planned this month with last
month's result in hand."** — the one sentence that is true on the LLM
path (decision 12). The Outlook line grows one month at a time and "Next 3
months" fills in from "deliberating…".

### Act 5 — What the board learned (2 min)

- **History** → open the September 2026 entry (or press **Details** on the
  Last-month card, which opens it directly). Verified: "Did 2 · partly 1 ·
  didn't 1 of 4 actions", the four decisions with ✓ ✎ ✕ marks and the
  marketing note, then **How the plan held up** with the same scored
  sentences and "What the board changed".
- **Why this plan → How the board got here** on the new plan: the
  September strip now reads **Changed (your numbers)** with "you did $35k"
  in place of "the simulation did". The previous cycle's page (History →
  August → Why this plan) shows the close's evidence line — "4 edge(s)
  strengthened or weakened for the actions taken (half weight: partly
  done)" — and the hiring step absent from it. Be straight that one
  observed month does not promote an edge; it takes several.

### Act 6 — Break it on purpose (optional, 1 min)

The strongest trust close is a failure state that says so. Two ways:

- **Engine down.** Stop the stack (Ctrl+C on `start.ps1`) with the app
  still open, then Close the month with any numbers. Verified in code and
  tests, not in this live run: the Close form keeps the numbers and says
  the engine couldn't be reached to score the plan; This month shows one
  notice — "The analysis service couldn't be reached, so no plan was
  started" — with **Retry**. (Verified live: the same notice appears when a
  cycle start fails with the API down.)
- **Strategist down.** Restart with `.\start.ps1 -Prod -NoOllama` and
  close a month: the plan comes from the built-in rules and This month
  says so in its one notice ("The AI strategist couldn't be reached for
  this plan…"); Why this plan carries the same. Test-pinned, not run live
  here.

Do not stop Ollama *during* a cycle for effect: the months already landed
stay, the rest come from rules, and the notice names the months.

## 5. Reset between runs

- **Browser only** (fast): Settings → Load the seeded demo company →
  Replace. The screens return to the Act 1 state. The server, however,
  still holds the close you made: a second close of the same seeded cycle
  inserts another feedback row and writes its evidence to the graph again.
  Fine for a rehearsal; not for a clean run.
- **Clean run**: re-run `experiments\seed_demo_company.py` (new cycle ids
  on the server, a fresh bootstrap file), then load it from Settings.
- **Everything**: Settings → Delete all my data (browser); stop the stack
  and delete `data\startup_society.db*` (server); `chroma_db_founder\` and
  the graph are the memory stores — see `founder_testing_runbook.md` Part 5
  before touching them, and back the graph up first (Part 0.5).

## 6. If the numbers surprise you

- **"Low confidence - 6 of these numbers are estimates"** on a company that
  supplied everything: the six are engine defaults (interest rate, consumer
  confidence, unemployment, valuation multiple, innovation factor, lifetime
  value). The count caps the band on purpose. Assumptions on Why lists
  them; "Fill these in" appears only for a guess the Close form can take.
- **The projection's plan line ends at $49k of cash** while This month
  says "Not burning": the KPI is today's cost line against today's revenue;
  the projection adds the plan's own spend for twelve months. Both are
  labelled with their horizon so they cannot read as a contradiction.
- **"Churn … the wrong direction"** when churn improved: the board expected
  0.0pp and it moved −0.2pp; the sign test is the engine's, and the
  sentence reports it rather than tidying it.
- **A second cycle costs money** only in engine time: about 45 s.
  Re-run / Plan again is offered on This month only when the plan is on
  older numbers or came from the rules; a fresh plan on current numbers
  has no button to press, by design.

## 7. What was verified, and how

Live on 2026-09-23 (this runbook's numbers): health block; seeded company
loaded from Settings; This month, Why this plan (projection run, all four
expanders), History as quoted; the Act 3 close with the inputs above,
landing on This month in 2.9 s; the new cycle completing with
`startedFromTrackRecord: true`; the Last-month card and plan cards as
quoted in Act 4. Earlier the same day, with a fresh company: onboarding →
Analyzing → This month, the cycle completing while History was open, and
the start-failed notice with the API stopped.

Automated: `npm test` in `frontend/` (the honesty checklist, one fixture
per notice case), `venv\Scripts\python.exe -m pytest
tests/test_founder_contract.py tests/test_founder_view.py -q`, and the
loop suites `tests/test_cycle_api.py tests/test_loop_plumbing.py
tests/test_expected_delta.py`.
