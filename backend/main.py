from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.database import connect, initialize_database, parse_json_fields, row_to_dict, utc_now
from backend.schemas import AdviseRequest, ScenarioCreate, SimulationCreate, WhatIfRequest
from backend import demo_fixtures
from backend.advise_service import ORACLE_MODE, run_analysis, store_analysis
from backend.whatif_service import run_whatif
from backend.simulation_service import (
    SUPPORTED_POLICIES,
    create_run,
    get_run,
    list_runs,
    start_run,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="Startup Society of Minds API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "database": "sqlite",
        "simulation_engine": "ready",
        "advisor_mode": ORACLE_MODE,
        # Whether this instance can answer without Ollama and Neo4j, and how.
        "demo_fixtures": demo_fixtures.enabled(),
    }


@app.get("/api/config")
def config() -> dict:
    return {
        "policies": SUPPORTED_POLICIES,
        "default_policy": "boardroom",
        "oracle_policies_require_llm": [
            "oracle_v1",
            "oracle_v2",
            "oracle_v3",
            "oracle_v4",
            "oracle_v4_causal",
        ],
    }


@app.post("/api/advise")
def advise(payload: AdviseRequest) -> dict:
    """One board analysis of a founder's current numbers (G1).

    Runs synchronously: a single Boardroom.decide() with one Oracle call takes
    roughly 20-90s on the local model, inside the client's 120s budget. The
    client renders an honest failure state on timeout rather than a fake brief.
    """
    now = utc_now()
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO companies (id, name, age_months, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                age_months = excluded.age_months,
                updated_at = excluded.updated_at
            """,
            (
                payload.company_id,
                payload.config.company_name,
                payload.company_age_months,
                now,
                now,
            ),
        )

    body = payload.model_dump()

    if demo_fixtures.enabled():
        # Offline demonstration. Recorded answers for the demo dataset, and a
        # real but model-free board run for anything else. The live path is
        # never reached, deliberately: run_analysis does not raise when Ollama
        # and Neo4j are down, it spends about six seconds failing to connect
        # and then degrades into its own fallback - so "try live, catch the
        # error" would have shown that degraded answer instead of the
        # recording, slowly, with a stack of connection warnings in the log.
        # Set FOUNDER_DEMO_FIXTURES=0 for the live stack.
        result = demo_fixtures.lookup(body) or demo_fixtures.offline_analysis(body)
    else:
        try:
            result = run_analysis(body)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Analysis engine unavailable: {exc}",
            ) from exc

    analysis_id = store_analysis(payload.company_id, payload.month_index, result)
    return {"analysis": {**result, "id": analysis_id}}


@app.post("/api/whatif")
def whatif(payload: WhatIfRequest) -> dict:
    """Twelve-month projection under three policies (D5).

    Pure simulation: no LLM call, no Oracle, no memory write. The board's plan
    arrives from an analysis the client already has, so this is CPU-only and
    returns in well under a second for the default 50 seeds x 3 policies.
    """
    try:
        return run_whatif(payload.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Projection engine unavailable: {exc}"
        ) from exc


@app.get("/api/companies/{company_id}/analyses")
def company_analyses(company_id: str, limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM analyses WHERE company_id = ?
            ORDER BY created_at DESC LIMIT ?
            """,
            (company_id, limit),
        ).fetchall()
    return [
        parse_json_fields(dict(row), "brief_json", "trace_json") for row in rows
    ]


@app.get("/api/scenarios")
def scenarios() -> list[dict]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM scenarios ORDER BY updated_at DESC"
        ).fetchall()
    return [parse_json_fields(dict(row), "config_json") for row in rows]


@app.post("/api/scenarios", status_code=201)
def save_scenario(payload: ScenarioCreate) -> dict:
    now = utc_now()
    with connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO scenarios (name, config_json, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (payload.name, payload.config.model_dump_json(), now, now),
        )
        row = connection.execute(
            "SELECT * FROM scenarios WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    return parse_json_fields(row_to_dict(row), "config_json")


@app.get("/api/runs")
def runs(limit: int = Query(default=20, ge=1, le=100)) -> list[dict]:
    return list_runs(limit)


@app.post("/api/runs", status_code=202)
def run_simulation(payload: SimulationCreate) -> dict:
    if payload.policy not in SUPPORTED_POLICIES:
        raise HTTPException(status_code=422, detail="Unsupported policy")
    try:
        run = create_run(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    start_run(run["id"])
    return run


@app.get("/api/runs/{run_id}")
def run_detail(run_id: str, include_trace: bool = False) -> dict:
    run = get_run(run_id, include_trace=include_trace)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        requested = FRONTEND_DIST / path
        if path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(FRONTEND_DIST / "index.html")
