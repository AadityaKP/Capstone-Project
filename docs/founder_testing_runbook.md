# Founder frontend — testing runbook

How to bring up the full stack and exercise every screen yourself, start to finish.

Branch: `founder-integration`. Everything below is verified on Windows (PowerShell) against
this branch.

---

## What you are running

Three processes, two of which you start:

| Piece | What it is | Port | You start it? |
|---|---|---|---|
| Frontend | Vite dev server (React) | 5173 | Yes |
| Backend | FastAPI + SQLite | 8000 | Yes |
| Ollama | Local LLM (`llama3.1:8b`) | 11434 | Runs as a service |

The frontend proxies `/api` to the backend. The backend calls Ollama for the one
Oracle reasoning call per analysis.

**Neo4j is not required.** The founder product runs `oracle_v4`, which builds no causal
graph store — only `oracle_v4_causal` does. You can leave Neo4j stopped.

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

---

## Part 1 — Start the stack

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
advisor_mode       : oracle_v4
```

Then check the proxy path the browser actually uses:

```bash
Invoke-RestMethod http://127.0.0.1:5173/api/health
```

Same response. If this one fails but the first succeeded, the Vite proxy is the problem,
not the backend.

Open **http://127.0.0.1:5173**.

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

You land on **Advice**. Values will be close to these but may shift slightly between runs:

- Position: **Low risk**, growth is accelerating, **High confidence**
- Provenance line: *your first analysis · numbers from <today> · 1 estimated input*
- **Product & retention** — invest ≈$6.0k (≈49% of MRR), marked *Priority*
- **Marketing & growth** — ≈$7.5k on brand building (≈62% of MRR), *up from the ≈$3.0k you reported*
- **Hiring** — wait on hiring
- **Pricing** — hold pricing
- Watch-outs and opportunities written by the strategist (these vary run to run)

Then check **Home** — same plan, plus runway (~8 mo), MRR $12k, churn 5.0%/mo, growth
efficiency *Healthy* at ≈8.8×.

Every dollar figure carries a **% of MRR** alongside it. That is deliberate: the engine is
calibrated for a ~$50k-MRR company, and percentages stay honest across company sizes where
absolute dollars do not.

### 3.4 Confirm it persisted

```bash
venv\Scripts\python.exe -c "import sqlite3,json; c=sqlite3.connect('data/startup_society.db'); c.row_factory=sqlite3.Row; r=c.execute('SELECT id,llm_ok,oracle_mode FROM analyses ORDER BY created_at DESC LIMIT 1').fetchone(); print(dict(r))"
```

Expect `llm_ok: 1` and `oracle_mode: oracle_v4`. `llm_ok: 0` means the strategist call
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
| Clicks do nothing after a branch switch | Vite hot-reload got stuck mid-merge | Hard-reload the page |
| Blank page, console shows duplicate `createRoot` | Dev-only HMR artifact | Harmless; hard-reload clears it |

---

## Known gaps

Not defects in what you are testing — deliberate scope boundaries worth knowing before you
judge a result.

- **`/api/advise` is synchronous.** The spec (G1) calls for an async job with status. It
  currently blocks for the whole 30–90s call. Fine for one tester; it will not hold up under
  concurrent users.
- **Risk read can disagree with runway.** The board has called *Low risk* on a company with
  ~8 months runway while advising ~90% of MRR in spend. That is the engine's own judgment,
  not a wiring fault — but it is the kind of advice a founder would act on.
- **Macro assumptions can surface as findings.** Watch-outs like "Low Unemployment Rate"
  derive from system defaults the founder never supplied. The screen labels the count as
  *estimated input*, but the item still reads as a finding about their business.
- **Months are local-first.** Monthly snapshots live in browser localStorage; only analyses
  reach the server. Clearing browser data loses history that the database does not hold.
