# Founder frontend — testing runbook

How to bring up the full stack and exercise every screen yourself, start to finish.

Branch: `founder-calibration`. Everything below is verified on Windows (PowerShell) against
this branch.

**New since the last revision:** the what-if projection (Part 3.5) and the assumed-values
expander (Part 3.4). Both are on the Advice screen. If you only have time for one thing,
do Part 3.5 — it is the surface a reviewer is most likely to ask about, because it is the
only one that shows the simulator actually running.

---

## What you are running

Three processes, two of which you start:

| Piece | What it is | Port | You start it? |
|---|---|---|---|
| Frontend | Vite dev server (React) | 5173 | Yes |
| Backend | FastAPI + SQLite | 8000 | Yes |
| Ollama | Local LLM (`llama3.1:8b`) | 11434 | Runs as a service |
| Neo4j | Causal graph | 7687 | Runs as a service |

The frontend proxies `/api` to the backend. The backend calls Ollama for Oracle reasoning
and batched causal proposals, and reads the Neo4j causal graph for evidence.

**Neo4j is read-only here.** Graph writes require an active shock label or
`Oracle.end_episode()`, and one `Boardroom.decide()` triggers neither — verified by
comparing node counts before and after a run (Part 0.5). Your research graph is safe, but
back it up anyway.

To run without Neo4j entirely, set `FOUNDER_ORACLE_MODE=oracle_v4`. You lose causal
evidence; everything else works.

---

## Part 0 — Prerequisites (once)

### 0.1 Check Ollama has the model

```bash
ollama list
```

You need a `llama3.1:8b` row. If it is missing:

```bash
ollama pull llama3.1:8b
```

That is a ~4.9 GB download. Confirm the server answers:

```bash
curl http://localhost:11434/api/tags
```

### 0.2 Python dependencies

```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 0.3 Frontend dependencies

```bash
cd frontend; npm install
```

### 0.4 Founder memory store

Founder analyses read and write `chroma_db_founder/`, a copy of the research corpus, so
live testing cannot contaminate your thesis memories. If the folder does not exist:

```bash
cp -r chroma_db chroma_db_founder
```

If you have no `chroma_db` either, skip this — an empty store is created automatically and
everything still runs, just without retrieved memories.

### 0.5 Back up the causal graph

The founder path only reads Neo4j, but the research graph is not reproducible:

```bash
venv\Scripts\python.exe neo4j_backup.py dump
```

Writes a timestamped JSON to `backups/`. After any test run, prove nothing changed:

```bash
venv\Scripts\python.exe neo4j_backup.py verify
```

Expected: `UNCHANGED - the graph was not written to.` To roll back,
`neo4j_backup.py restore backups/<file>.json`.

---

## Part 1 — Start the stack

### The fast path — one command

```bash
.\start.ps1
```

Starts Ollama if it is not already up, checks the model is pulled, starts the API, waits
until `/api/health` actually answers, starts Vite, waits for that too, and opens the
browser. **Ctrl+C stops all of it** — including uvicorn's and npm's child processes, which
`Stop-Process` alone would orphan while they kept holding the ports.

If script execution is blocked on this machine:

```bash
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

For a demo, serve everything from one port instead — no Vite, no proxy, no chance of the
two halves disagreeing:

```bash
.\start.ps1 -Prod
```

`-Prod` always rebuilds, deliberately: a stale `frontend/dist` is the worst failure mode
here because it is completely silent — the page loads, looks correct, and runs the previous
build's JavaScript. Even so, hard-reload (**Ctrl+Shift+R**) after a rebuild, because your
*browser* caches `index.html` and will keep requesting the old bundle hash.

Other flags: `-NoOllama` (skip the model and test the rules-only path from 4.2),
`-NoBrowser`, `-Reload` (restart the API on backend `.py` changes), `-ApiPort` / `-UiPort`.
Logs land in `.logs\api.out.log` and `.logs\ui.out.log`.

If a port is already in use the script refuses to start rather than half-starting — free it
or pass a different port.

### The manual path

Worth knowing, because when the script fails these are the two commands it was running.
Two terminals, both from the repository root.

**Terminal 1 — backend:**

```bash
venv\Scripts\python.exe run_app.py
```

**Terminal 2 — frontend:**

```bash
cd frontend; npm run dev
```

### Confirm both are up

```bash
Invoke-RestMethod http://127.0.0.1:8000/api/health
```

Expected:

```
status             : ok
database           : sqlite
simulation_engine  : ready
advisor_mode       : oracle_v4_causal
```

Then check the proxy path the browser actually uses:

```bash
Invoke-RestMethod http://127.0.0.1:5173/api/health
```

Same response. If this one fails but the first succeeded, the Vite proxy is the problem,
not the backend.

Open **http://127.0.0.1:5173**.

---

## What the UI does, and what it asks for

A tester's map. Eight screens; the founder is asked for **8 required numbers** and nothing
else is mandatory.

### Required input — the whole ask

| Step | Field | Why it is needed |
|---|---|---|
| 1 | Company name | labels their own screens |
| 1 | Company age (months) | retention behaves differently with age |
| 1 | Market crowdedness | maps to a competitor count |
| 2 | Monthly recurring revenue | revenue base for every ratio |
| 2 | Cash in the bank | runway numerator |
| 2 | Total monthly costs | becomes virtual headcount at $8k slots, so burn reaches the board |
| 3 | Average price per customer | ARPA — selects the churn benchmark band |
| 3 | Monthly churn | the board's primary retention signal |

### Optional enrichment — never blocking

What you sell · marketing spend last month · new customers last month · product maturity ·
acquisition cost if tracked · team size · per-segment churn (enterprise / SMB / consumer).

Marketing spend **and** new customers together unlock the derived acquisition cost. Each
field states why it is wanted, and the accordion is collapsed by default so the minimum
path stays short.

### The eight screens

| Screen | What it is for |
|---|---|
| **Welcome** | three doors: start, explore the sample company, how it works |
| **Onboarding** | 3 steps, live derived preview (CAC, LTV) before committing |
| **Analyzing** | staged progress, honest elapsed timer, leaves the seat open |
| **Home** | risk band, four KPI tiles, this month's plan, what changed |
| **Advice** | the four domains, each with "Why this number?", watch-outs, evidence, the what-if projection, assumed values |
| **History** | month list with deltas and accepted-action counts |
| **My company** | every input with a provenance chip |
| **Settings** | narratives toggle, service status, sample exit, delete all data |

### What the founder is told, and how carefully

- **Four decisions per month**: product spend, marketing spend, hiring, pricing. Each
  carries a dollar figure **and a % of MRR**, because the engine is calibrated near $50k
  MRR and percentages stay honest across sizes where dollars do not.
- **Provenance on every input**: *You provided* · *Derived* · *Estimated by the system*.
- **Evidence is graded**: observed causal edges and seeded priors are worded differently,
  and retrieved memories carry a *from simulations, not real companies* tag.
- **Failure is stated, never filled in**: unreachable backend, unreachable strategist
  (rules-only banner), and stale numbers each have their own honest state.
- **Nothing is fabricated**: if the strategist cannot be parsed, `llm_ok` is false and the
  UI says the plan came from built-in rules.

---

## Part 2 — Sample company (no backend needed)

Fastest way to see every screen populated. Good for reviewing layout and copy.

1. On the welcome screen, click **Explore a sample company**.
2. You land on **Home** for "Acme Analytics", labelled *Sample company — data is illustrative*.
3. Walk the sidebar: **Home → Advice → History → My company → Settings**.

What to check:

- **Home** — risk band, four KPI tiles (runway, MRR, churn, growth efficiency), this month's
  plan, "what changed".
- **Advice** — the four domains (product, marketing, hiring, pricing), each with a "Why this
  number?" expander, watch-outs and things working in your favour.
- **History** — three months with deltas and accepted-action counts.
- **My company** — every input with a provenance chip: *You provided*, *Derived*, or
  *Estimated by the system*. Acquisition cost should read **$91**, not $0.
- **Settings** — narratives toggle, analysis service status, **Leave sample company**.

Sample mode never touches your real workspace. Leaving it returns you to an empty state.

---

## Part 3 — Full run with a real analysis

This is the end-to-end path: onboarding → live engine call → your own dashboard.

### 3.1 Onboarding

From the welcome screen click **Get started**.

**Step 1 — Your company**

| Field | Value |
|---|---|
| Company name | `Testco` |
| What you sell | (leave blank) |
| Company age | `12` |
| Market crowdedness | **A few rivals** |

**Step 2 — Your money**

| Field | Value |
|---|---|
| Monthly recurring revenue | `12000` |
| Cash in the bank | `90000` |
| Total monthly costs | `24000` |
| Marketing spend last month | `3000` |

**Step 3 — Your customers**

| Field | Value |
|---|---|
| Average price per customer | `40` |
| Customer churn (monthly) | `5` |
| New customers last month | `33` |
| Product maturity | **Solid** |

Before continuing, check the derived line above the footer:

> Acquisition cost ≈ **$91** · Lifetime value ≈ **$800** (price ÷ churn)

Both figures must be non-zero. `$0` means the currency formatter regressed.

### 3.2 Run it

Click **Run my first analysis**.

The Analyzing screen shows staged progress and a live elapsed timer. **Expect 30–90 seconds**
on first run — that is one real LLM call, not a fake delay. The timer is honest; if it sits
past ~2 minutes, see troubleshooting.

### 3.3 What you should see

You land on **Advice**. These values are **deterministic** — the Oracle runs at
temperature 0, so identical inputs give identical output. Nine repeat runs across two
profiles produced zero variation. If you see something different, something has changed.

| Surface | Expected |
|---|---|
| Position | **Moderate risk** · *High confidence* |
| Provenance | *your first analysis · numbers from <today> · 6 estimated inputs* |
| Product & retention | **≈$4.8k** (≈40% of MRR), marked *Priority* |
| Marketing & growth | **≈$1.8k** (≈15% of MRR) |
| Hiring | wait on hiring |
| Pricing | hold pricing |

Watch-outs and opportunities are written by the strategist and vary in wording.

**Six estimated inputs, not one.** The count now comes from the server rather than being
computed in the browser. It used to read *1 estimated input* because the client could only
see the fields it had itself decided not to ask for; four macro values —
`valuation_multiple`, `unemployment`, `innovation_factor`, `months_in_depression` — were
being filled in by pydantic schema defaults that no code could enumerate. If you see *1*,
you are looking at an analysis stored before this change, or at the sample company.

**Moderate, not low, is the point.** This company has 7.5 months of runway and 5.0%
monthly churn against a published median of 3.65% for its price point. Earlier builds
called this `LOW`; the risk read only became discriminating once burn, runway and the
churn benchmark reached the prompt.

Open **Evidence — what this is based on**. Under the *From simulations, not real
companies* tag you should see one line from the Neo4j causal graph:

> The board's working assumption when cash ran tight: runway shortened, hiring was frozen
> and marketing spend was cut. This is a built-in prior, not something measured in past
> runs.

That line is the proof Neo4j is contributing. Note the wording carefully: for this profile
the graph's edges are `MAY_CAUSE`, which are **seeded priors**, not observations. A line
beginning *"In past simulated runs…"* appears only when the graph holds `CONFIRMED_CAUSE`
edges for the company's stress node. The two are deliberately worded differently — if you
ever see a seeded prior described as something that happened, that is a bug.

If no line appears at all, the graph query returned nothing — check `neo4j_backup.py
verify` shows a populated graph, and that `advisor_mode` reads `oracle_v4_causal`.

Then check **Home** — same plan, plus runway (~8 mo), MRR $12k, churn 5.0%/mo, growth
efficiency *Healthy* at ≈8.8×.

Every dollar figure carries a **% of MRR** alongside it. That is deliberate: the engine is
calibrated for a ~$50k-MRR company, and percentages stay honest across company sizes where
absolute dollars do not.

### 3.3b Check the two calibration guards fired

Not visible on screen — they live in the stored trace, and both carry their source:

```bash
venv\Scripts\python.exe -c "import sqlite3,json; c=sqlite3.connect('data/startup_society.db'); c.row_factory=sqlite3.Row; t=json.loads(c.execute('SELECT trace_json FROM analyses ORDER BY created_at DESC LIMIT 1').fetchone()[0]); print(json.dumps({'churn_benchmark':t.get('churn_benchmark'),'spend_ceiling':t.get('spend_ceiling')}, indent=1))"
```

Expect `churn_benchmark` to show the company at 5.0% against a 3.65% median for the
`arpa_25_100` band, cited to ChartMogul with the annual→monthly derivation; and
`spend_ceiling` to show `extrapolated: true`, because the spend benchmark is printed for
$3–5M ARR companies and this founder is far below that band. **`extrapolated: true` is
expected here, not a fault** — it is the system declining to pretend the benchmark was
published for a company this size.

### 3.4 Assumed values — check nothing is silently filled in

Open **Assumed values (6)** on the Advice screen. Expect a list naming every field the
founder did not supply, the value used, and why:

| Field | Value | Why |
|---|---|---|
| Interest rate | 3.0% | not asked at onboarding; typical conditions |
| Consumer confidence | 100.0 | index where 100 is neutral |
| Unemployment | 4.0% | not asked at onboarding |
| Valuation multiple | 10.0x ARR | engine default |
| Innovation factor | 1.0 | no scarring assumed at the start of an analysis |
| Lifetime value | price ÷ monthly churn | derived from your own numbers |

If the founder supplied a value, its row disappears — enter an interest rate during
onboarding and the count drops to 5. That is the check worth doing: the list is generated
from what the server actually used, so a row that persists after you supplied the value
means the input is not reaching `build_env_state`.

A **Churn split** row appears when the founder gives one blended churn figure and no
per-segment breakdown. The engine models enterprise, SMB and consumer churn separately;
the blended number fills all three, so the average it computes is exactly the figure
entered. Worth reading once — it is the least obvious mapping in the product.

### 3.5 The what-if projection — the simulator, running

This is the only screen that steps the Gymnasium environment. Everything else asks the
board a question; this one plays the answer forward.

Scroll below the four plan cards to **What happens if you follow this plan** and click
**Run the projection**.

**It should return in well under a second.** There is no LLM call — the plan comes from
the analysis already on screen. If it takes seconds, something is re-invoking the board.

| Surface | Expected |
|---|---|
| Charts | four — Monthly revenue, Cash, Churn, Rule of 40 |
| Per chart | three median lines, each inside its own shaded 25–75 band |
| Legend | *Take the board's plan* · *Keep doing what you're doing* · *Standard playbook* |
| Caveat | directly under the charts, not in a footnote |
| Table | three rows: revenue, cash, survival %, Rule of 40 |
| Footer | *Median of 50 simulated runs per plan, same 50 starting conditions for each* |

On the reference profile the board's plan should end ahead of doing nothing on revenue,
and **Standard playbook** should show a visibly lower survival rate — it spends harder and
runs some seeds out of cash. That ordering is the point of the panel: it is not there to
make the board look good, and if the board's plan loses, that is a finding to report, not
a bug to fix.

**Now click *Add a competitor shock*.** The fan re-runs with a competitor surge at month 6.
Expect two new columns, **Shock cost** and **Recovery**, an amber dashed marker at month 6
on all four charts, and a line describing the shock.

**Recovery will read "no drop", and that is correct.** `competitor_surge` adds three
competitors, cuts price 25% and lifts SMB churn 50% — it does not remove revenue. A company
still growing never falls below its pre-shock level, so there is no recovery time to
report and the panel says so instead of printing a fabricated `0 mo`. The shock's real cost
shows in the **Shock cost** column, which re-runs each plan on the same seeds without the
shock and compares. Expect small single-digit negative percentages.

That the shock is nearly inert on revenue is a property of the model worth knowing:
`compute_new_mrr` steps on competitor count only at 4 and 10, so 5→8 changes nothing, and
in the default marketing curve new revenue does not depend on price at all.

Finally open **Show what this projection assumes (4)**. Gross margin should read **83.5%**,
tagged *derived*, cited to SaaS Capital with the arithmetic shown — it is the sum of four
printed CoGS components, not a figure any source prints directly. Price should read *held
flat*, tagged **unidentified**, because price elasticity has no public dataset and the
projection refuses to move a lever it cannot justify.

### 3.6 Confirm it persisted

```bash
venv\Scripts\python.exe -c "import sqlite3,json; c=sqlite3.connect('data/startup_society.db'); c.row_factory=sqlite3.Row; r=c.execute('SELECT id,llm_ok,oracle_mode FROM analyses ORDER BY created_at DESC LIMIT 1').fetchone(); print(dict(r))"
```

Expect `llm_ok: 1` and `oracle_mode: oracle_v4_causal`. `llm_ok: 0` means the strategist call
failed and you are looking at rules-only output (see 4.2).

---

## Part 4 — Failure states

These matter as much as the happy path. The product's central claim is that it never
fabricates an analysis, and these are the paths where that claim is actually tested.

### 4.1 Backend unreachable

Stop the backend (Ctrl+C in terminal 1), then try **Update my numbers → Run analysis**.

Expected: *The analysis service couldn't be reached*, your numbers saved, with **Retry
analysis** and **Continue without analysis**. No invented brief anywhere.

Restart the backend and hit **Retry analysis** — it should complete normally.

### 4.2 Strategist unreachable (rules-only degradation)

Stop Ollama, leaving the backend running:

```bash
Stop-Process -Name ollama -Force
```

Run an analysis. It still succeeds, but the Advice screen must carry an amber banner:

> The AI strategist couldn't be reached for this analysis. This plan comes from the board's
> built-in rules — still grounded in your numbers, just without the strategist's read.

Confirm the flag reached the database — the query from 3.4 should now show `llm_ok: 0`.

This is the honesty guarantee working end to end. Before this branch, a failed strategist
call silently produced a neutral, confident-looking brief indistinguishable from a real one.

Restart Ollama afterwards (launch it from the Start menu, or `ollama serve`).

### 4.3 Stale numbers

The footer shows *Numbers from <date>* and the sidebar flags staleness once a month passes.
To exercise it without waiting, edit the stored `enteredAt` in the browser console:

```js
const s = JSON.parse(localStorage.getItem("ssom_founder_v1"));
s.months[0].enteredAt = new Date(Date.now() - 45*864e5).toISOString();
localStorage.setItem("ssom_founder_v1", JSON.stringify(s));
location.reload();
```

---

## Part 5 — Reset between runs

**Frontend data** — Settings → **Delete all my data** → **Delete permanently**. This clears
localStorage and returns you to the welcome screen.

**Backend data** — stop the backend, then:

```bash
Remove-Item data\startup_society.db*
```

Tables are recreated on next startup. This clears stored companies and analyses; it does not
touch your memory stores.

**Memory stores** — leave `chroma_db/` alone. It is the research corpus and nothing in the
founder path writes to it. To reset founder memories, delete `chroma_db_founder/` and copy
it again from `chroma_db/`.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/api/health` 500 through :5173 but fine on :8000 | Vite proxy | Restart the frontend dev server |
| Analysis hangs past ~2 min | Ollama cold-loading the model | Wait out the first call; later ones are faster |
| `llm_ok: 0` unexpectedly | Ollama down or model missing | `ollama list`, then `ollama serve` |
| Acquisition cost shows `$0` | Currency formatter regression | Check the sub-$1k branch in `money()` in `frontend/src/derive.js` |
| Advice recommends more than MRR | Calibration scaling not applied | Check `absolute_scale()` in `backend/advise_service.py` |
| No causal sentence in Evidence | Neo4j down, or graph empty | `neo4j_backup.py verify`; check `advisor_mode` is `oracle_v4_causal` |
| `[CausalGraphStore] Neo4j unavailable` in backend log | Neo4j not running | Start Neo4j, or set `FOUNDER_ORACLE_MODE=oracle_v4` |
| Clicks do nothing after a branch switch | Vite hot-reload got stuck mid-merge | Hard-reload the page |
| Blank page, console shows duplicate `createRoot` | Dev-only HMR artifact | Harmless; hard-reload clears it |
| **What-if panel missing entirely on :8000** | Browser cached the old `index.html`, which points at a bundle hash that no longer exists | **Ctrl+Shift+R.** Hit during verification of this build — the page loads and looks fine, the panel is simply absent. Confirm with `document.querySelector('script').src`: it should end in the hash printed by the last `npm run build` |
| What-if panel missing on :5173 but present on :8000 | Dev server started before the file existed | Restart the Vite dev server |
| Projection returns 503 | `run_whatif` raised | Check the backend log; the endpoint wraps any failure rather than returning a half-built chart |
| Projection takes seconds, not milliseconds | Something is re-invoking the board | It must read `trace.final_action` from the existing analysis, never call the LLM |
| **Conditions diverged between plans** warning in the footer | The three policies stopped sharing an RNG stream | Expected never at a 12-month horizon (0 of 50 rollouts). If you see it, the horizon was raised or the starting macro state was unusual — the comparison is confounded and should be reported, not ignored |

---

## Known gaps

Not defects in what you are testing — deliberate scope boundaries worth knowing before you
judge a result.

- **`/api/advise` is synchronous.** The spec (G1) calls for an async job with status. It
  currently blocks for the whole 30–90s call. Fine for one tester; it will not hold up under
  concurrent users.
- **The spend ceiling extrapolates.** The only spend benchmark that survived source
  verification is printed for $3–5M ARR companies. Applying it to a founder is
  extrapolation; the trace says so on every analysis, but the number itself is borrowed.
  A pre-revenue company can still be told to spend 51% of MRR on product and stay within
  the ceiling.
- **Macro assumptions can surface as findings.** Watch-outs like "Low Unemployment Rate"
  derive from system defaults the founder never supplied. The screen labels the count as
  *estimated input*, but the item still reads as a finding about their business.
- **"Three AI advisors" is one LLM call.** CFO, CMO and CPO are three sections of a single
  JSON response, not three agents. Two Ollama calls happen per analysis: one strategist
  brief, one batched proposal.
- **The plan can contradict its own evidence.** On the reference run above, the board
  recommended *adding* payroll while the Evidence panel said hiring was frozen — for a
  company it had itself scored as `Cash_Shortage` with ~4 months runway by its own estimate.
  The deterministic CFO guard (no hires under 24 months runway) returns zero here; the LLM
  proposal generator overrode it. Worth watching on every run.
- **Months are local-first.** Monthly snapshots live in browser localStorage; only analyses
  reach the server. Clearing browser data loses history that the database does not hold.
- **The projection holds the plan constant.** It repeats this month's plan for twelve
  months. A founder re-running the analysis monthly would get a different path, so the fan
  is "what if you did this and nothing else", not "what if you used this product".
- **The projection's do-nothing arm may be unfairly weak.** It holds the founder's current
  marketing spend, which onboarding only asks for optionally, and R&D, which it never asks
  for at all — so both can sit at zero. When marketing spend is absent the assumptions list
  says so. Enter last month's marketing spend to make the comparison fair.
- **The shock is nearly inert on revenue** — see 3.5. Do not present shock mode as evidence
  the plan confers resilience until the shock actually moves revenue.
- **Cash in the projection uses an 83.5% gross margin; the rest of the engine does not.**
  Everywhere else revenue still lands in cash at 100% margin (defect D4). The flag is
  default-off and only the projection sets it, so a cash figure on Home and a cash figure
  in the fan chart are computed differently. Recorded in `docs/data_provenance.md` §R3.1.
