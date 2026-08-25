from __future__ import annotations

import json
import math
import threading
import uuid
from typing import Any

import numpy as np

from backend.database import connect, parse_json_fields, row_to_dict, utc_now
from simulation_runner import run_simulation


SUPPORTED_POLICIES = [
    "heuristic",
    "random",
    "boardroom",
    "oracle_v1",
    "oracle_v2",
    "oracle_v3",
    "oracle_v4",
    "oracle_v4_causal",
]

_active_runs: set[str] = set()
_active_runs_lock = threading.Lock()


def json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def create_run(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    now = utc_now()
    with connect() as connection:
        if payload.get("scenario_id") is not None:
            scenario = connection.execute(
                "SELECT id FROM scenarios WHERE id = ?", (payload["scenario_id"],)
            ).fetchone()
            if scenario is None:
                raise ValueError("Scenario not found")
        connection.execute(
            """
            INSERT INTO simulation_runs (
                id, scenario_id, policy, episodes, seed_start, oracle_frequency,
                status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'queued', ?)
            """,
            (
                run_id,
                payload.get("scenario_id"),
                payload["policy"],
                payload["episodes"],
                payload["seed_start"],
                payload["oracle_frequency"],
                now,
            ),
        )
    return get_run(run_id)


def start_run(run_id: str) -> None:
    with _active_runs_lock:
        if run_id in _active_runs:
            return
        _active_runs.add(run_id)
    thread = threading.Thread(target=_execute_run, args=(run_id,), daemon=True)
    thread.start()


def _execute_run(run_id: str) -> None:
    try:
        with connect() as connection:
            run = row_to_dict(
                connection.execute(
                    "SELECT * FROM simulation_runs WHERE id = ?", (run_id,)
                ).fetchone()
            )
            if run is None:
                return
            scenario_config = None
            if run["scenario_id"] is not None:
                scenario_row = connection.execute(
                    "SELECT config_json FROM scenarios WHERE id = ?",
                    (run["scenario_id"],),
                ).fetchone()
                if scenario_row is not None:
                    scenario_config = json.loads(scenario_row["config_json"])
            connection.execute(
                "UPDATE simulation_runs SET status = 'running', started_at = ? WHERE id = ?",
                (utc_now(), run_id),
            )

        frame, traces = run_simulation(
            policy=run["policy"],
            num_episodes=run["episodes"],
            seed_start=run["seed_start"],
            oracle_frequency=run["oracle_frequency"],
            return_action_trace=True,
            return_monthly_trace=True,
            environment_config=scenario_config,
        )
        episodes = json_safe(frame.to_dict(orient="records"))
        monthly = json_safe(traces["monthly_trace"])
        actions = json_safe(traces["action_trace"])
        summary = build_summary(episodes, monthly, actions)

        with connect() as connection:
            connection.executemany(
                "INSERT INTO episode_results (run_id, episode, metrics_json) VALUES (?, ?, ?)",
                [
                    (run_id, row["episode"], json.dumps(row, allow_nan=False))
                    for row in episodes
                ],
            )
            connection.executemany(
                """
                INSERT INTO monthly_traces (run_id, episode, month, trace_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        row["episode"],
                        row["month"],
                        json.dumps(row, allow_nan=False),
                    )
                    for row in monthly
                ],
            )
            connection.executemany(
                """
                INSERT INTO action_traces (run_id, episode, month, trace_json)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        row["episode"],
                        row["month"],
                        json.dumps(row, allow_nan=False),
                    )
                    for row in actions
                ],
            )
            connection.execute(
                """
                UPDATE simulation_runs
                SET status = 'completed', summary_json = ?, completed_at = ?
                WHERE id = ?
                """,
                (json.dumps(summary, allow_nan=False), utc_now(), run_id),
            )
    except Exception as exc:
        with connect() as connection:
            connection.execute(
                """
                UPDATE simulation_runs
                SET status = 'failed', error = ?, completed_at = ?
                WHERE id = ?
                """,
                (str(exc), utc_now(), run_id),
            )
    finally:
        with _active_runs_lock:
            _active_runs.discard(run_id)


def build_summary(
    episodes: list[dict[str, Any]],
    monthly: list[dict[str, Any]],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    def average(field: str) -> float | None:
        values = [row[field] for row in episodes if row.get(field) is not None]
        return sum(values) / len(values) if values else None

    shock_events = [
        {
            "episode": row["episode"],
            "month": row["month"],
            "type": row["shock_label"],
            "mrr": row.get("mrr"),
            "cash": row.get("cash"),
        }
        for row in monthly
        if row.get("shock_label") not in (None, "NO_SHOCK")
    ]
    memories = []
    for row in actions:
        decision_trace = row.get("decision_trace") or {}
        for memory in decision_trace.get("retrieved_memories") or []:
            memories.append(memory)

    return json_safe(
        {
            "final_mrr": average("final_mrr"),
            "final_cash": average("final_cash"),
            "final_ltv_cac": average("final_ltv_cac"),
            "avg_rule_40": average("avg_rule_40"),
            "post_shock_rule_40": average("post_shock_avg_rule40_25_60"),
            "recovery_time_months": average("mean_recovery_time_months"),
            "survival_rate": (
                sum(row.get("cause") == "Time Limit" for row in episodes)
                / len(episodes)
                * 100
                if episodes
                else 0
            ),
            "shock_events": shock_events,
            "memories": memories[:10],
            "latest_brief": next(
                (row.get("brief") for row in reversed(actions) if row.get("brief")),
                None,
            ),
        }
    )


def get_run(run_id: str, include_trace: bool = False) -> dict[str, Any] | None:
    with connect() as connection:
        run = parse_json_fields(
            row_to_dict(
                connection.execute(
                    "SELECT * FROM simulation_runs WHERE id = ?", (run_id,)
                ).fetchone()
            ),
            "summary_json",
        )
        if run is None:
            return None
        episodes = connection.execute(
            "SELECT metrics_json FROM episode_results WHERE run_id = ? ORDER BY episode",
            (run_id,),
        ).fetchall()
        run["episodes_results"] = [json.loads(row["metrics_json"]) for row in episodes]
        if include_trace:
            trace_rows = connection.execute(
                """
                SELECT trace_json FROM monthly_traces
                WHERE run_id = ? ORDER BY episode, month
                """,
                (run_id,),
            ).fetchall()
            run["monthly_trace"] = [json.loads(row["trace_json"]) for row in trace_rows]
        return run


def list_runs(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM simulation_runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [parse_json_fields(dict(row), "summary_json") for row in rows]
