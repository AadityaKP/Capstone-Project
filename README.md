# Startup Society of Minds

An integrated React, FastAPI, and Python simulation application. The API runs
the startup environment and boardroom policies from `startup-multi`, while
SQLite persists scenarios, runs, episode metrics, monthly traces, and actions.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
cd frontend
npm install
npm run build
cd ..
python run_app.py
```

Open `http://127.0.0.1:8000`. FastAPI serves the built frontend and API as one
application. For frontend development, run `npm run dev` inside `frontend`;
Vite proxies `/api` to port 8000.

## The founder product: the OEFA loop

The product's primary flow is a multi-month cycle (`docs/oefa_loop_plan.md`,
decisions in `docs/oefa_loop_decisions.md`). Each cycle steps four months
through **Observe → Execute → Feedback → Adapt**: the board reads the company's
own memory, decides with a numeric prediction (`expected_delta`) on every
proposal, finds out what its decision did in the simulator, and adapts. The
founder then closes the month with real numbers (did / partly / didn't per
action) and the board is scored against its own prediction; only actions that
were actually taken become evidence.

- Navigation: **This month · History · My company · Settings**. This month
  carries the plan, how last month's plan held up, and one Outlook chart on
  which months land as the board deliberates (everything right of your own
  numbers is shaded "projected"); **Why this plan** holds the reasons,
  evidence, assumptions and an OEFA strip per horizon month; the close-the-month
  form is the one place the founder answers. Inventory: `docs/ui_components.md`.
- API: `POST /api/cycles` (202, months stream in), `GET /api/cycles/{id}`,
  `POST /api/cycles/{id}/feedback` (the HITL close), and `/api/health` reports
  whether the loop is actually live (`graph_store_enabled`, `memory_scope`,
  `llm_reachable`).
- Run the stack for a demo with `.\start.ps1 -Prod` (sets `SIM_PROFILE=founder`;
  the review2 research profile cannot run the loop's Feedback step).
  `docs/demo_runbook.md` is the verified run of show: what to load, the exact
  numbers to type, and what every screen shows.
- Before a demo: `venv\Scripts\python.exe experiments\loop_live_check.py`
  makes both silent failure modes loud; `experiments\seed_demo_company.py`
  seeds a company with history and two cycles (one closed) and Settings offers
  to load it; `experiments\prediction_error_harness.py <export.json>` reports
  the calibration of `expected_delta` against a company's own months.

## Data and optional services

The primary application database is SQLite at `data/startup_society.db`.
SQLite is used because this application stores structured run, episode, and
time-series records and should work without an external database server.

Oracle policies use the existing optional integrations:

- Ollama at `OLLAMA_BASE_URL` for model reasoning.
- ChromaDB at `CHROMA_PATH` for Oracle memory.
- Neo4j at `NEO4J_URI` for `oracle_v4_causal`.

The `heuristic`, `random`, and `boardroom` policies work without those services.
Copy `.env.example` to `.env` when enabling optional integrations.

## API

- `GET /api/health`
- `GET /api/config`
- `GET/POST /api/scenarios`
- `GET/POST /api/runs`
- `GET /api/runs/{run_id}?include_trace=true`

Run tests with `pytest tests/test_api.py tests/test_startup_multi_integration.py`.
