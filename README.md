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
