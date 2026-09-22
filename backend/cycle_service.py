"""The loop engine: one cycle = H months of Observe -> Execute -> Feedback -> Adapt.

docs/oefa_loop_plan.md section 4. The single-month advise path runs Observe
and Execute and stops: it never steps the environment, so the board never finds
out what its own decision did. This module is the missing half, and the HITL
close (submit_feedback, section 6) is what turns the simulated half into a
real one.

Per month, in order:

  Observe   Boardroom.decide() observes the state (snapshot + trend), retrieves
            similar past months from the company's own memory and, under
            oracle_v4_causal, role-specific causal evidence. All of it is
            recorded, because the retrieval is the explanation.
  Execute   Boardroom.decide() -> final action + full decision trace, now with
            expected_delta on every proposal and on the final action. The
            founder-product guards (spend ceiling, hiring-runway guard) apply.
  Feedback  env.step(action) -> next state. kpi_delta against the month's
            opening state, prediction_error against the month's expected_delta,
            and write_causal_outcome(..., source="sim"): the board finds out
            what its decision did and how far off its prediction was.
  Adapt     the delta shapes the next month: the track record goes to every
            agent, weights move through the brief, the brief refreshes on
            events only (oracle_frequency=0). What changed is stated
            explicitly per month.

Four things this deliberately does NOT do, each from plan section 4.1:
end_episode() is never called; the cadence knob is 0 so events earn the
refresh; only the month-1 observation (the real current state) is persisted
into the company's Oracle calendar - simulated months 2..H are observed on the
in-memory Oracle only, and memory writes are suspended for them so a real
pending month can never mature against a simulated snapshot; brief freshness
travels per month rather than being implied.

Asynchronous, following simulation_service.start_run: POST /api/cycles returns
202 with the id, months land in cycle_months as they finish, and GET returns
whatever is done so far.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from copy import deepcopy
from typing import Any, Callable

from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent
from backend import founder_view, sim_profile
from backend.advise_service import (
    _apply_spend_ceiling,
    _graph_summary,
    _replay_history,
    assumed_fields,
    build_boardroom,
    build_env_state,
    build_oracle,
)
from backend.database import connect, parse_json_fields, row_to_dict, utc_now
from backend.oracle_state import load_oracle_state, save_oracle_state
from boardroom import expectation as ex
from boardroom.boardroom import Boardroom
from env import business_logic
from env.schemas import EnvState
from env.startup_env import StartupEnv

DEFAULT_HORIZON_MONTHS = 4
MAX_HORIZON_MONTHS = 6

# The cycle runs the Boardroom with the cadence branch off (plan section 4.1):
# FOUNDER_ORACLE_FREQUENCY = 1 makes months_elapsed % 1 == 0 always true, so
# under it the event triggers never get a turn. 0 leaves `initial` + `event`.
CYCLE_ORACLE_FREQUENCY = 0

# HITL evidence weights (docs/oefa_loop_decisions.md, decision 3).
FEEDBACK_WEIGHT = {"did": 1.0, "partly": 0.5}

_active: set[str] = set()
_active_lock = threading.Lock()


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------

def _upsert_company(payload: dict[str, Any]) -> None:
    now = utc_now()
    config = payload.get("config") or {}
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
                payload["company_id"],
                config.get("company_name") or "My company",
                int(payload.get("company_age_months") or 0),
                now,
                now,
            ),
        )


def create_cycle(payload: dict[str, Any]) -> dict[str, Any]:
    """Queue a cycle. Returns the stored row; nothing has run yet."""
    _upsert_company(payload)
    cycle_id = str(uuid.uuid4())
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO cycles (
                id, company_id, month_index, horizon_months, status, request_json,
                oracle_mode, use_oracle, seed, created_at
            ) VALUES (?, ?, ?, ?, 'queued', ?, ?, ?, ?, ?)
            """,
            (
                cycle_id,
                payload["company_id"],
                int(payload.get("month_index") or 0),
                int(payload.get("horizon_months") or DEFAULT_HORIZON_MONTHS),
                json.dumps(payload),
                sim_profile.get_oracle_mode() if payload.get("use_oracle", True) else "none",
                1 if payload.get("use_oracle", True) else 0,
                int(payload.get("seed") or 0),
                utc_now(),
            ),
        )
    return get_cycle(cycle_id)


def start_cycle(cycle_id: str) -> None:
    """Run the cycle on a background thread; safe to call twice."""
    with _active_lock:
        if cycle_id in _active:
            return
        _active.add(cycle_id)
    threading.Thread(target=_execute_cycle, args=(cycle_id,), daemon=True).start()


def run_cycle_sync(cycle_id: str) -> dict[str, Any]:
    """Run the cycle inline (tests, the demo seed script). Same code path."""
    with _active_lock:
        _active.add(cycle_id)
    _execute_cycle(cycle_id)
    return get_cycle(cycle_id)


def _execute_cycle(cycle_id: str) -> None:
    try:
        with connect() as connection:
            row = row_to_dict(
                connection.execute("SELECT * FROM cycles WHERE id = ?", (cycle_id,)).fetchone()
            )
            if row is None:
                return
            payload = json.loads(row["request_json"])
            connection.execute(
                "UPDATE cycles SET status = 'running', started_at = ? WHERE id = ?",
                (utc_now(), cycle_id),
            )

        def persist_month(month: dict[str, Any]) -> None:
            with connect() as connection:
                connection.execute(
                    """
                    INSERT INTO cycle_months (cycle_id, month_index, month_json, latency_s, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(cycle_id, month_index) DO UPDATE SET
                        month_json = excluded.month_json,
                        latency_s = excluded.latency_s
                    """,
                    (cycle_id, month["month_index"], json.dumps(month), month["latency_s"], utc_now()),
                )
            print(
                f"[cycle {cycle_id[:8]}] month {month['month_index']} done in "
                f"{month['latency_s']:.1f}s (brief {month['execute']['brief_source']}, "
                f"llm_ok={month['execute']['llm_ok']})"
            )

        months, summary, meta = run_months(payload, on_month=persist_month)
        with connect() as connection:
            connection.execute(
                """
                UPDATE cycles SET status = 'completed', summary_json = ?, completed_at = ?
                WHERE id = ?
                """,
                (json.dumps({"summary": summary, "meta": meta}), utc_now(), cycle_id),
            )
    except Exception as exc:  # the row must say what happened, not hang in 'running'
        with connect() as connection:
            connection.execute(
                "UPDATE cycles SET status = 'failed', error = ?, completed_at = ? WHERE id = ?",
                (f"{type(exc).__name__}: {exc}", utc_now(), cycle_id),
            )
    finally:
        with _active_lock:
            _active.discard(cycle_id)


def fail_orphaned_cycles() -> int:
    """Called at startup. A cycle runs on a thread inside the server process,
    so a restart (uvicorn --reload on a code edit, a crash, a deploy) kills
    it silently and leaves the row 'running' forever, with the client polling
    a plan that will never land. Mark those failed and say why."""
    with connect() as connection:
        cursor = connection.execute(
            """
            UPDATE cycles SET status = 'failed', completed_at = ?,
                error = 'the engine restarted while this cycle was running; re-run it from your numbers'
            WHERE status IN ('queued', 'running')
            """,
            (utc_now(),),
        )
        return cursor.rowcount


def get_cycle(cycle_id: str) -> dict[str, Any] | None:
    with connect() as connection:
        row = row_to_dict(
            connection.execute("SELECT * FROM cycles WHERE id = ?", (cycle_id,)).fetchone()
        )
        if row is None:
            return None
        month_rows = connection.execute(
            "SELECT month_json FROM cycle_months WHERE cycle_id = ? ORDER BY month_index",
            (cycle_id,),
        ).fetchall()
        feedback_rows = connection.execute(
            "SELECT month_index, feedback_json, result_json, created_at FROM cycle_feedback "
            "WHERE cycle_id = ? ORDER BY month_index",
            (cycle_id,),
        ).fetchall()
    cycle = parse_json_fields(row, "summary_json")
    stored = cycle.pop("summary", None) or {}
    cycle["summary"] = stored.get("summary")
    cycle["meta"] = stored.get("meta")
    cycle["request"] = json.loads(cycle.pop("request_json"))
    cycle["use_oracle"] = bool(cycle.get("use_oracle"))
    cycle["months"] = [json.loads(r["month_json"]) for r in month_rows]
    cycle["feedback"] = [
        {
            "month_index": r["month_index"],
            "submitted": json.loads(r["feedback_json"]),
            "result": json.loads(r["result_json"]) if r["result_json"] else None,
            "created_at": r["created_at"],
        }
        for r in feedback_rows
    ]
    return cycle


def list_cycles(company_id: str, limit: int = 20) -> list[dict[str, Any]]:
    with connect() as connection:
        rows = connection.execute(
            "SELECT id FROM cycles WHERE company_id = ? ORDER BY created_at DESC LIMIT ?",
            (company_id, limit),
        ).fetchall()
    return [get_cycle(r["id"]) for r in rows]


# --------------------------------------------------------------------------
# the loop
# --------------------------------------------------------------------------

def _display(state: EnvState, action: dict[str, Any], brief: dict[str, Any],
             assumed_count: int) -> dict[str, Any]:
    """The same founder-vocabulary block advise_service attaches, so a cycle
    month renders with exactly the words a single analysis would."""
    burn = business_logic.monthly_burn(state)
    outflow = (
        burn
        + float((action.get("marketing") or {}).get("spend", 0.0) or 0.0)
        + float((action.get("product") or {}).get("r_and_d_spend", 0.0) or 0.0)
    )
    return {
        "confidence": founder_view.confidence(brief.get("confidence"), assumed_count),
        "runway": founder_view.runway_phrase(state.cash, burn, state.mrr),
        "spend_ratio": founder_view.spend_ratio_phrase(outflow, state.mrr),
        "show_rule_of_40": founder_view.rule_of_40_is_meaningful(state.mrr),
        "monthly_burn": burn,
        "monthly_burn_supplied": state.monthly_burn is not None,
    }


def _weight_moves(previous: dict[str, float] | None, current: dict[str, float] | None) -> list[dict[str, Any]]:
    if not previous or not current:
        return []
    moves = []
    for key, value in current.items():
        delta = float(value) - float(previous.get(key, value))
        if abs(delta) >= 0.005:
            moves.append({"key": key, "from": round(float(previous[key]), 3),
                          "to": round(float(value), 3), "delta": round(delta, 3)})
    return moves


def run_months(
    payload: dict[str, Any],
    on_month: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Step H months. Returns (months, summary, meta). `on_month` is called
    with each month as it completes so a caller can persist or stream it."""
    state = build_env_state(payload)
    company_id = payload.get("company_id")
    horizon = max(1, min(MAX_HORIZON_MONTHS, int(payload.get("horizon_months") or DEFAULT_HORIZON_MONTHS)))
    use_oracle = bool(payload.get("use_oracle", True))
    seed = int(payload.get("seed") or 0)
    scale = sim_profile.get_agent_scale(state.mrr)
    env_kwargs = sim_profile.get_env_kwargs(gross_margin=sim_profile.get_applied_gross_margin())
    expectation = ex.make_expectation(env_kwargs=env_kwargs)
    assumed = assumed_fields(payload)

    oracle = None
    if use_oracle:
        oracle, _ = build_oracle(state, company_id)
        boardroom = build_boardroom(
            state, oracle, scale, oracle_frequency=CYCLE_ORACLE_FREQUENCY, expectation=expectation
        )
    else:
        boardroom = Boardroom(
            [
                CFOProposalAgent(scale=scale, expectation=expectation),
                CMOProposalAgent(scale=scale, expectation=expectation),
                CPOProposalAgent(scale=scale, expectation=expectation),
            ],
            use_oracle=False,
            expectation=expectation,
            **sim_profile.get_boardroom_kwargs(state.mrr),
        )
    boardroom.start_episode(episode_seed=seed)

    replayed = 0
    if oracle is not None:
        oracle.import_state(load_oracle_state(company_id))
        replayed = _replay_history(boardroom, state, payload.get("history") or [])

    # The previous HITL close, if the client sent it back: the board starts
    # this cycle already knowing how its last prediction went.
    boardroom.set_track_record(payload.get("previous_track_record"))

    env = StartupEnv(initial_config=env_kwargs)
    env.reset(seed=seed)
    env.state = state.model_copy(deep=True)

    months: list[dict[str, Any]] = []
    previous_weights: dict[str, float] | None = None
    persisted_oracle_state: dict[str, Any] | None = None
    stats_before = boardroom.get_episode_stats() if use_oracle else {}

    for index in range(horizon):
        started = time.perf_counter()
        opening = env.state.model_copy(deep=True)
        pending_before = len(oracle.pending_memories) if oracle is not None else 0

        # ---- Observe + Execute (decide() observes first, then decides) ----
        action = boardroom.decide(opening)
        trace = dict(boardroom.get_last_decision_trace() or {})
        final_action = deepcopy(trace.get("final_action") or action)
        spend_ceiling = _apply_spend_ceiling(final_action, opening) if sim_profile.apply_spend_ceiling() else None
        expected = trace.get("expected_delta")
        if spend_ceiling and spend_ceiling.get("applied"):
            # The guard changed what will actually be executed; the prediction
            # must be for the executed action or the score means nothing.
            expected = expectation(opening, final_action)
        trace["final_action"] = final_action
        trace["expected_delta"] = expected
        trace["spend_ceiling"] = spend_ceiling
        trace["graph_summary"] = _graph_summary(trace)
        trace["assumed_fields"] = assumed
        trace["absolute_scale"] = scale
        brief = trace.get("brief") or {}
        llm_ok = bool(brief.get("parse_ok", False)) if use_oracle else False
        trace["display"] = _display(opening, final_action, brief, len(assumed))

        if oracle is not None and index == 0:
            # The real current state is the only observation that belongs in
            # the company's persisted calendar; from here on the Oracle sees
            # simulated months and must neither persist nor mature on them.
            persisted_oracle_state = oracle.export_state()
            oracle.memory_writes_enabled = False

        pending_after = len(oracle.pending_memories) if oracle is not None else 0
        observe = {
            "state_before": {**ex.kpi_values(opening), "months_elapsed": opening.months_elapsed},
            "env_state_before": opening.model_dump(mode="json"),
            "memory_count": trace.get("memory_count", 0),
            "memories": trace.get("retrieved_memories") or [],
            "trend": (
                oracle.latest_trend_context.model_dump(mode="json") if oracle is not None else None
            ),
            "graph": {
                "stress_node": trace.get("causal_stress_node"),
                "contexts": trace.get("causal_contexts") or {},
                "summary": trace.get("graph_summary"),
                "enabled": oracle.graph_store_enabled if oracle is not None else False,
            },
            "memory_scope": oracle.memory_scope if oracle is not None else None,
            "pending_memories": pending_after,
            "matured_memories": max(0, pending_before + 1 - pending_after) if index == 0 else 0,
        }
        execute = {
            "action": final_action,
            "brief": brief,
            "llm_ok": llm_ok,
            "brief_source": trace.get("brief_source"),
            "refresh_reason": trace.get("refresh_reason"),
            "proposal_source": trace.get("proposal_source"),
            "proposals": trace.get("proposals") or [],
            "weights": trace.get("applied_weights"),
            "base_weights": trace.get("base_weights"),
            "expected_delta": expected,
            "spend_ceiling": spend_ceiling,
            "display": trace["display"],
            "trace": trace,
        }

        # ---- Feedback ----
        _, _, terminated, _, info = env.step(deepcopy(final_action))
        after = env.state.model_copy(deep=True)
        kpi_delta = ex.kpi_delta_between(opening, after)
        error = ex.prediction_error(expected, kpi_delta)
        evidence_written = False
        if oracle is not None:
            evidence_written = oracle.write_causal_outcome(
                final_action,
                {k: v for k, v in kpi_delta.items() if v is not None},
                stress_node=trace.get("causal_stress_node"),
                month=opening.months_elapsed,
                source="sim",
            )
        # The chart's uncertainty band is the spread of THIS month's end state
        # across seeds under the executed action - one month ahead, not the
        # two-month expected_delta horizon - so the band around a projected
        # point is the uncertainty of that point (plan section 5.2).
        band = ex.predict_expected_delta(
            opening, final_action, horizon_months=1, env_kwargs=env_kwargs
        )["band"]
        feedback = {
            "state_after": {**ex.kpi_values(after), "months_elapsed": after.months_elapsed,
                            "survived": not terminated},
            "kpi_delta": kpi_delta,
            "prediction_error": error,
            "projection_band": band,
            "evidence_written": evidence_written,
            "evidence_source": "sim",
            "shock_label": info.get("shock_label"),
            "basis": "simulated",
        }

        # ---- Adapt ----
        record = ex.build_track_record(
            final_action, expected, kpi_delta, month_label=f"month {index + 1}", source="simulated"
        )
        boardroom.set_track_record(record)
        weights = trace.get("applied_weights") or {}
        moves = _weight_moves(previous_weights, weights)
        adaptations = [
            {"agent": p.get("agent"), "sentence": p.get("adaptation")}
            for p in (trace.get("proposals") or []) if p.get("adaptation")
        ]
        what_changed: list[str] = []
        source = trace.get("brief_source")
        reason = trace.get("refresh_reason")
        if source == "llm":
            what_changed.append(f"brief refreshed by the strategist ({reason})")
        elif source == "cache_hit":
            what_changed.append(f"brief refreshed from a matching earlier read ({reason})")
        elif source == "reuse":
            what_changed.append("brief reused - no event moved the numbers enough to re-read them")
        for move in moves:
            what_changed.append(
                f"{move['key']} weight {'up' if move['delta'] > 0 else 'down'} "
                f"{abs(move['delta']):.3f} to {move['to']:.3f}"
            )
        for item in adaptations:
            what_changed.append(f"{item['agent']}: {item['sentence']}")
        if error and error["summary"]["kpis_scored"]:
            what_changed.append(
                f"prediction scored: {error['summary']['sign_agrees']} of "
                f"{error['summary']['kpis_scored']} KPIs moved in the predicted direction"
            )
        adapt = {
            "what_changed": what_changed,
            "refresh_reason": reason,
            "brief_source": source,
            "weight_moves": moves,
            "adaptations": adaptations,
            "track_record_for_next_month": record,
            "memory": {"pending": pending_after, "matured_this_month": observe["matured_memories"]},
        }
        previous_weights = weights

        month = {
            "month_index": index + 1,
            "months_elapsed": opening.months_elapsed,
            "projection": index > 0,
            "observe": observe,
            "execute": execute,
            "feedback": feedback,
            "adapt": adapt,
            "latency_s": round(time.perf_counter() - started, 3),
        }
        months.append(month)
        if on_month is not None:
            on_month(month)
        if terminated:
            break

    if oracle is not None:
        oracle.memory_writes_enabled = True
        if persisted_oracle_state is not None:
            save_oracle_state(company_id, persisted_oracle_state)

    stats_after = boardroom.get_episode_stats() if use_oracle else {}
    last = months[-1]
    summary = {
        "months_completed": len(months),
        "horizon_months": horizon,
        "survived": all(m["feedback"]["state_after"]["survived"] for m in months),
        "projected_state": last["feedback"]["state_after"],
        "runway_at_horizon": last["feedback"]["state_after"]["runway_months"],
        "fresh_briefs": sum(1 for m in months if m["execute"]["brief_source"] == "llm"),
        "reused_briefs": sum(1 for m in months if m["execute"]["brief_source"] in ("cache_hit", "reuse")),
        "llm_ok_months": sum(1 for m in months if m["execute"]["llm_ok"]),
        "llm_calls": int(stats_after.get("llm_calls", 0)) - int(stats_before.get("llm_calls", 0)),
        "proposal_llm_calls": int(stats_after.get("proposal_llm_calls", 0)) - int(stats_before.get("proposal_llm_calls", 0)),
        "evidence_written_months": sum(1 for m in months if m["feedback"]["evidence_written"]),
        "total_latency_s": round(sum(m["latency_s"] for m in months), 3),
        "graph_store_enabled": oracle.graph_store_enabled if oracle is not None else False,
        "memory_scope": oracle.memory_scope if oracle is not None else None,
        "prediction": {
            "kpis_scored": sum((m["feedback"]["prediction_error"] or {}).get("summary", {}).get("kpis_scored", 0) for m in months),
            "sign_agrees": sum((m["feedback"]["prediction_error"] or {}).get("summary", {}).get("sign_agrees", 0) for m in months),
            "within_tolerance": sum((m["feedback"]["prediction_error"] or {}).get("summary", {}).get("within_tolerance", 0) for m in months),
        },
    }
    meta = {
        "oracle_mode": oracle.mode if oracle is not None else "none",
        "use_oracle": use_oracle,
        "seed": seed,
        "history_months_replayed": replayed,
        "assumed_fields": assumed,
        "absolute_scale": scale,
        "sim_profile": sim_profile.get_profile(),
        "expected_delta_basis": "simulated",
        "started_from_track_record": payload.get("previous_track_record") is not None,
    }
    return months, summary, meta


# --------------------------------------------------------------------------
# HITL close (plan section 6)
# --------------------------------------------------------------------------

def _actual_state(opening: EnvState, actuals: dict[str, Any]) -> EnvState:
    """The founder's real month as an EnvState: the cycle's opening state with
    the numbers the founder reported, one month on."""
    actual = opening.model_copy(deep=True)
    actual.mrr = max(0.0, float(actuals["mrr"]))
    actual.cash = float(actuals["cash"])
    churn = min(max(float(actuals["churn"]) / 100.0, 0.0), 1.0)
    actual.churn_enterprise = actual.churn_smb = actual.churn_b2c = churn
    if actuals.get("costs") is not None and sim_profile.apply_monthly_burn():
        actual.monthly_burn = max(0.0, float(actuals["costs"]))
    actual.months_elapsed = opening.months_elapsed + 1
    return actual


def _credited_action(final_action: dict[str, Any], per_action: list[dict[str, Any]]) -> tuple[dict[str, Any], list[str], list[str], float]:
    """The bundle of actions the founder actually took. Domains marked
    "didnt" (or not mentioned) are set to no-op so the evidence is about what
    happened, not what was advised. Returns (action, credited, skipped, weight)."""
    status = {item["action_key"]: item["done"] for item in per_action}
    credited = [k for k, v in status.items() if v in FEEDBACK_WEIGHT]
    skipped = [k for k in ("marketing", "product", "hiring", "pricing") if k not in credited]
    action = deepcopy(ex.NOOP_ACTION)
    for domain in credited:
        block = final_action.get(domain)
        if isinstance(block, dict):
            action[domain].update(block)
    weight = min((FEEDBACK_WEIGHT[status[k]] for k in credited), default=0.0)
    return action, credited, skipped, weight


def submit_feedback(cycle_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Close a month with the founder's real numbers.

    1. prediction error: the month's expected_delta against the actual delta;
    2. actions marked did/partly -> write_causal_outcome(source="observed"),
       one write for the bundle actually taken, partly at half weight;
    3. actions marked didnt -> no causal edge. The month is not evidence about
       an action nobody took, and the response says so per domain;
    4. oracle.observe_state(actual_state): the real month enters episodic
       memory through the persisted per-company calendar, where it matures
       six real months later;
    5. the track record for the next cycle, so it starts from the actual
       state carrying this prediction error.
    """
    cycle = get_cycle(cycle_id)
    if cycle is None:
        raise KeyError("Cycle not found")
    month_index = int(payload.get("month_index") or 1)
    month = next((m for m in cycle["months"] if m["month_index"] == month_index), None)
    if month is None:
        raise ValueError(f"Cycle has no month {month_index} to close")

    opening = EnvState(**month["observe"]["env_state_before"])
    actuals = dict(payload["actuals"])
    actual = _actual_state(opening, actuals)
    final_action = month["execute"]["action"]
    expected = month["execute"]["expected_delta"]

    actual_delta = ex.kpi_delta_between(opening, actual)
    error = ex.prediction_error(expected, actual_delta)

    company_id = cycle["company_id"]
    per_action = [dict(item) for item in payload.get("per_action") or []]
    credited_action, credited, skipped, weight = _credited_action(final_action, per_action)

    evidence: dict[str, Any] = {
        "written": False,
        "edges": 0,
        "source": "observed",
        "weight": weight,
        "credited": credited,
        "skipped": skipped,
        "reason": None,
    }
    memory: dict[str, Any] = {"observed": False, "pending": None, "matured": 0}

    oracle = None
    if cycle["use_oracle"]:
        oracle, _ = build_oracle(actual, company_id)
        oracle.start_episode(episode_seed=None)
        oracle.import_state(load_oracle_state(company_id))
        graph_kpis = {k: v for k, v in actual_delta.items() if v is not None}
        if not credited:
            evidence["reason"] = (
                "none of the plan's actions were taken, so this month is not evidence "
                "about any of them; only the state was recorded"
            )
        elif not graph_kpis:
            evidence["reason"] = "no KPI moved measurably, so there is nothing to attribute"
        elif not oracle.graph_store_enabled:
            evidence["reason"] = (
                "the causal graph is not reachable; the state was recorded but no "
                "evidence was written"
            )
        else:
            evidence["written"] = oracle.write_causal_outcome(
                credited_action, graph_kpis,
                stress_node=month["observe"]["graph"].get("stress_node"),
                month=actual.months_elapsed, source="observed", weight=weight,
            )
            evidence["edges"] = len(graph_kpis) if evidence["written"] else 0
            evidence["reason"] = (
                f"{len(graph_kpis)} edge(s) strengthened or weakened for the actions taken"
                + (" (half weight: partly done)" if weight < 1.0 else "")
            )
        pending_before = len(oracle.pending_memories)
        oracle.observe_state(actual)
        memory = {
            "observed": True,
            "pending": len(oracle.pending_memories),
            "matured": max(0, pending_before + 1 - len(oracle.pending_memories)),
            "scope": oracle.memory_scope,
        }
        save_oracle_state(company_id, oracle.export_state())
    else:
        evidence["reason"] = "this cycle ran without the Oracle, so nothing is written back"

    track_record = ex.build_track_record(
        final_action, expected, actual_delta,
        month_label=f"month {month_index}", source="observed",
    )
    per_domain = []
    for domain in ("marketing", "product", "hiring", "pricing"):
        status = next((i["done"] for i in per_action if i["action_key"] == domain), None)
        per_domain.append({
            "action_key": domain,
            "done": status,
            "counted": domain in credited,
            "note": next((i.get("note") for i in per_action if i["action_key"] == domain), None),
            "why": (
                "counted as evidence" if domain in credited and status == "did"
                else "counted at half weight" if domain in credited
                else "you didn't do this, so this month is not counted as evidence about it"
                if status == "didnt"
                else "not part of last month's plan"
            ),
        })

    result = {
        "cycle_id": cycle_id,
        "month_index": month_index,
        "expected_delta": expected,
        "actual_delta": actual_delta,
        "prediction_error": error,
        "actual_state": {**ex.kpi_values(actual), "months_elapsed": actual.months_elapsed},
        "evidence": evidence,
        "per_action": per_domain,
        "memory": memory,
        "track_record": track_record,
        "graph_store_enabled": oracle.graph_store_enabled if oracle is not None else False,
        "closed_at": utc_now(),
    }
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO cycle_feedback (cycle_id, month_index, feedback_json, result_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(cycle_id, month_index) DO UPDATE SET
                feedback_json = excluded.feedback_json,
                result_json = excluded.result_json,
                created_at = excluded.created_at
            """,
            (cycle_id, month_index, json.dumps(payload), json.dumps(result), utc_now()),
        )
    return result
