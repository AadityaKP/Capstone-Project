from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.database import connect, initialize_database, parse_json_fields, row_to_dict, utc_now
from backend.schemas import ScenarioCreate, SimulationCreate
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
    return {"status": "ok", "database": "sqlite", "simulation_engine": "ready"}


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
