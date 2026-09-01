from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.database import connect, initialize_database, parse_json_fields, row_to_dict, utc_now
from backend.schemas import (
    AdviseRequest,
    BacktestRunRequest,
    CompareRequest,
    ScenarioCreate,
    SimulationCreate,
    WhatIfRequest,
)
from backend.advise_service import run_analysis, store_analysis
from backend.review_service import (
    BACKTEST_POLICIES,
    COMPARE_POLICIES,
    backtest_companies,
    dataset_meta,
    edgar_growth_band,
    figure_path,
    panel_rows,
    run_backtest_policy,
    run_compare_policy,
)
from backend.sim_profile import get_oracle_mode, get_profile
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
        "advisor_mode": get_oracle_mode(),
        "sim_profile": get_profile(),
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

    try:
        result = run_analysis(payload.model_dump())
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


@app.get("/api/review/meta")
def review_meta() -> dict:
    """EDGAR growth band + dataset-card numbers for the review demo.

    Everything numeric is read from validation/results/environment_stats.csv or
    computed from data/edgar_ratios.csv at request time (review ground rule:
    nothing hand-typed).
    """
    return {
        "edgar_growth": edgar_growth_band(),
        "dataset": dataset_meta(),
        "compare_policies": COMPARE_POLICIES,
    }


@app.get("/api/review/figures/{key}")
def review_figure(key: str):
    """The two dataset-card figures, served from validation/figures/review/."""
    path = figure_path(key)
    if path is None:
        raise HTTPException(status_code=404, detail="Unknown review figure")
    return FileResponse(path)


@app.post("/api/review/compare")
def review_compare(payload: CompareRequest) -> dict:
    """One policy x one seed x 120 months under deterministic_rng.

    Synchronous on purpose: the client fires one request per arm, renders the
    fast arm as it lands, and keeps a progress indicator up for the LLM arm.
    Runs are serialized inside review_service (shared global RNG), so the
    boardroom request submitted first also finishes first.
    """
    if payload.policy not in COMPARE_POLICIES:
        raise HTTPException(status_code=422, detail="Unsupported compare policy")
    try:
        return run_compare_policy(payload.policy, payload.seed)
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Simulation failed: {exc}"
        ) from exc


@app.get("/api/review/panel")
def review_panel(
    ticker: str | None = Query(default=None),
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=25, ge=1, le=200),
    order: str = Query(default="asc", pattern="^(asc|desc)$"),
) -> dict:
    """Raw EDGAR panel rows for the Dataset tab — as-ingested dollars, paginated."""
    return panel_rows(ticker, offset, limit, descending=(order == "desc"))


@app.get("/api/review/backtest/companies")
def review_backtest_companies() -> dict:
    """The 39 C1-mapped company states for the Run tab dropdown."""
    return backtest_companies()


@app.post("/api/review/backtest/run")
def review_backtest_run(payload: BacktestRunRequest) -> dict:
    """One policy from one company's C1-mapped state, deterministic RNG.

    Synchronous; the client fires the three rule arms first, then oracle_v3.
    Runs are serialized inside review_service (shared global RNG).
    """
    if payload.policy not in BACKTEST_POLICIES:
        raise HTTPException(status_code=422, detail="Unsupported backtest policy")
    try:
        return run_backtest_policy(
            payload.ticker, payload.policy, payload.seed, payload.horizon
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=503, detail=f"Simulation failed: {exc}"
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
