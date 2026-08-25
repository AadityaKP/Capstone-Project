from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATABASE_PATH = ROOT_DIR / "data" / "startup_society.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def database_path() -> Path:
    configured = os.getenv("DATABASE_PATH")
    return Path(configured).resolve() if configured else DEFAULT_DATABASE_PATH


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def initialize_database() -> None:
    with connect() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS simulation_runs (
                id TEXT PRIMARY KEY,
                scenario_id INTEGER,
                policy TEXT NOT NULL,
                episodes INTEGER NOT NULL,
                seed_start INTEGER NOT NULL,
                oracle_frequency INTEGER NOT NULL,
                status TEXT NOT NULL,
                error TEXT,
                summary_json TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id)
            );

            CREATE TABLE IF NOT EXISTS episode_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                episode INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS monthly_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                episode INTEGER NOT NULL,
                month INTEGER NOT NULL,
                trace_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS action_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                episode INTEGER NOT NULL,
                month INTEGER NOT NULL,
                trace_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES simulation_runs(id) ON DELETE CASCADE
            );

            -- Founder product tables (spec G2). Companies and their monthly
            -- snapshots are the founder's own record; analyses store one
            -- Boardroom.decide() result per snapshot.
            CREATE TABLE IF NOT EXISTS companies (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                what_you_sell TEXT,
                age_months INTEGER NOT NULL DEFAULT 0,
                crowdedness TEXT,
                maturity TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS company_months (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                company_id TEXT NOT NULL,
                month_index INTEGER NOT NULL,
                values_json TEXT NOT NULL,
                entered_at TEXT NOT NULL,
                UNIQUE (company_id, month_index),
                FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                company_id TEXT NOT NULL,
                month_index INTEGER NOT NULL,
                status TEXT NOT NULL,
                brief_json TEXT,
                trace_json TEXT,
                llm_ok INTEGER,
                oracle_mode TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT,
                FOREIGN KEY (company_id) REFERENCES companies(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_runs_created_at
                ON simulation_runs(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_months_company
                ON company_months(company_id, month_index DESC);
            CREATE INDEX IF NOT EXISTS idx_analyses_company
                ON analyses(company_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_monthly_run_episode
                ON monthly_traces(run_id, episode, month);
            """
        )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def parse_json_fields(record: dict[str, Any] | None, *fields: str) -> dict[str, Any] | None:
    if record is None:
        return None
    for field in fields:
        value = record.get(field)
        if value:
            record[field.removesuffix("_json")] = json.loads(value)
        record.pop(field, None)
    return record
