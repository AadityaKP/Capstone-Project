"""Phase 4 of docs/oefa_loop_plan.md: the cycle and the HITL close.

Every item from the plan's section 7 list is here:

  - a cycle produces exactly H months
  - a cycle with use_oracle=False runs deterministically end to end
  - same seed -> same actions -> same projected states
  - feedback that is all-"didn't" writes zero causal edges
  - feedback that is "done" with a matching delta writes one edge per KPI in
    the delta and promotes nothing on the first observation
  - expected_delta is present on every proposal in the product path
  - per-month latency is in the record

Ollama and Neo4j are both absent here; the Oracle used by the cycle is built
with a canned LLM, an in-memory memory collection and an in-memory causal
graph that applies the real store's edge and promotion rules (tests/loop_fakes).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from backend import cycle_service, sim_profile
from backend.database import initialize_database
from backend.oracle_state import load_oracle_state
from oracle.memory import OracleMemoryStore
from oracle.oracle import Oracle
from tests.loop_fakes import FakeCollection, FakeLLM, InMemoryGraphStore


def request(company_id="co_test", horizon=4, use_oracle=False, seed=0, **overrides):
    payload = {
        "company_id": company_id,
        "company_age_months": 8,
        "month_index": 2,
        "config": {
            "company_name": "Acme Analytics",
            "initial_mrr": 30_000, "initial_cash": 220_000, "average_price": 85,
            "cac": 90, "churn_enterprise": 0.05, "churn_smb": 0.05, "churn_b2c": 0.05,
            "competitors": 9, "product_quality": 0.5, "monthly_costs": 48_000,
            "initial_headcount": 4,
        },
        "history": [{"mrr": 28_800, "churn": 0.056}, {"mrr": 30_000, "churn": 0.052}],
        "horizon_months": horizon,
        "use_oracle": use_oracle,
        "seed": seed,
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "cycles.db"))
    initialize_database()


@pytest.fixture
def fake_oracle(monkeypatch):
    """Route build_oracle to a hermetic Oracle: canned LLM, shared in-memory
    memory collection, in-memory causal graph. Returns the shared graph."""
    graph = InMemoryGraphStore()
    collection = FakeCollection()
    llm = FakeLLM()

    def build(state, company_id):
        scope = sim_profile.get_memory_scope(company_id)
        oracle = Oracle(
            mode="oracle_v4_causal", run_id=scope, dedupe_months=True,
            memory_store=OracleMemoryStore(run_id=scope, collection=collection),
            graph_store=graph, llm=llm, include_burn_context=True,
        )
        return oracle, None

    monkeypatch.setattr(cycle_service, "build_oracle", build)
    # The causal generator would try Ollama; without graph context it falls
    # back anyway, but keep the test on the rule-based proposal path.
    monkeypatch.setattr(sim_profile, "use_causal_proposals", lambda: False)
    return {"graph": graph, "collection": collection, "llm": llm}


def wait_for(client, cycle_id, timeout=120):
    deadline = time.time() + timeout
    while time.time() < deadline:
        cycle = client.get(f"/api/cycles/{cycle_id}").json()
        if cycle["status"] in ("completed", "failed"):
            return cycle
        time.sleep(0.2)
    raise AssertionError("cycle did not finish in time")


# --------------------------------------------------------------------------
# shape and determinism (no Oracle)
# --------------------------------------------------------------------------

def test_a_cycle_produces_exactly_h_months(db):
    for horizon in (1, 3, 4):
        months, summary, meta = cycle_service.run_months(request(horizon=horizon))
        assert len(months) == horizon
        assert summary["months_completed"] == horizon
        assert [m["month_index"] for m in months] == list(range(1, horizon + 1))
        assert months[0]["projection"] is False
        assert all(m["projection"] for m in months[1:])
        assert meta["use_oracle"] is False


def test_every_month_has_the_four_beats_and_a_latency(db):
    months, _, _ = cycle_service.run_months(request(horizon=2))
    for month in months:
        assert set(month) >= {"observe", "execute", "feedback", "adapt", "latency_s"}
        assert month["latency_s"] >= 0
        assert month["observe"]["state_before"]["mrr"] > 0
        assert month["execute"]["action"]["marketing"]["spend"] >= 0
        assert month["execute"]["expected_delta"]["horizon_months"] == 2
        assert set(month["feedback"]["kpi_delta"]) == {"mrr_pct", "cash_pct", "churn_pp", "runway_months"}
        assert month["feedback"]["prediction_error"]["summary"]["kpis_scored"] >= 3
        assert month["feedback"]["basis"] == "simulated"
        assert isinstance(month["adapt"]["what_changed"], list)
    # Month 2 knows what month 1 predicted and saw.
    assert months[1]["execute"]["trace"]["track_record"]["month"] == "month 1"
    assert months[0]["execute"]["trace"]["track_record"] is None


def test_use_oracle_false_is_deterministic_end_to_end(db):
    a, sa, _ = cycle_service.run_months(request(seed=3))
    b, sb, _ = cycle_service.run_months(request(seed=3))
    strip = lambda months: [(m["execute"]["action"], m["feedback"]["state_after"]) for m in months]
    assert strip(a) == strip(b)
    assert sa["projected_state"] == sb["projected_state"]


def test_the_seed_holds_the_world_fixed_and_changes_it_when_changed(db):
    a, _, _ = cycle_service.run_months(request(seed=1, horizon=4))
    b, _, _ = cycle_service.run_months(request(seed=2, horizon=4))
    # Same first decision (same opening state), different worlds afterwards.
    assert a[0]["execute"]["action"] == b[0]["execute"]["action"]
    assert a[-1]["feedback"]["state_after"] != b[-1]["feedback"]["state_after"]


def test_expected_delta_is_on_every_proposal_and_the_final_action(db):
    months, _, _ = cycle_service.run_months(request(horizon=2))
    for month in months:
        assert all(p["expected_delta"] is not None for p in month["execute"]["proposals"])
        assert month["execute"]["expected_delta"]["basis"] == "simulated"


def test_no_oracle_means_no_evidence_and_says_so(db):
    months, summary, _ = cycle_service.run_months(request(horizon=1))
    assert months[0]["feedback"]["evidence_written"] is False
    assert summary["graph_store_enabled"] is False
    assert summary["memory_scope"] is None


# --------------------------------------------------------------------------
# with the Oracle: observe, remember, write simulated evidence
# --------------------------------------------------------------------------

def test_oracle_cycle_writes_sim_evidence_and_persists_only_the_real_month(db, fake_oracle):
    months, summary, meta = cycle_service.run_months(request(use_oracle=True, horizon=3))
    graph = fake_oracle["graph"]
    assert summary["graph_store_enabled"] is True
    assert summary["memory_scope"] == "company:co_test"
    assert summary["evidence_written_months"] == 3
    assert all(w["source"] == "sim" for w in graph.writes)
    assert graph.edges_of("MAY_CAUSE") == []
    assert len(graph.edges_of("MAY_CAUSE_SIM")) >= 1
    assert graph.confirmed == set()
    # Brief freshness travels per month; with oracle_frequency=0 only the
    # first month is guaranteed fresh.
    assert months[0]["execute"]["brief_source"] == "llm"
    assert months[0]["execute"]["llm_ok"] is True
    assert all(m["execute"]["brief_source"] in ("llm", "cache_hit", "reuse") for m in months)
    # Persisted calendar = history (2) + the real current month, not the two
    # simulated ones the cycle stepped through.
    saved = load_oracle_state("co_test")
    assert saved["global_month"] == 3
    assert saved["state_history"][-1]["source_month"] == 8  # company_age_months, the current month
    assert meta["history_months_replayed"] == 2


def test_a_second_cycle_starts_where_the_first_left_off(db, fake_oracle):
    cycle_service.run_months(request(use_oracle=True, horizon=2))
    months, _, meta = cycle_service.run_months(request(use_oracle=True, horizon=1))
    # Same current month observed again: replaced, not duplicated.
    assert load_oracle_state("co_test")["global_month"] == 3
    assert months[0]["observe"]["pending_memories"] == 3


def test_previous_track_record_reaches_the_first_month(db, fake_oracle):
    from boardroom import expectation as ex
    record = ex.build_track_record(
        {"marketing": {"spend": 6000, "channel": "ppc"}, "product": {"r_and_d_spend": 9000},
         "hiring": {"hires": 0, "cost_per_employee": 10000}, "pricing": {"price_change_pct": 0}},
        {"mrr_pct": 5.0, "cash_pct": -4.0, "churn_pp": -0.2, "runway_months": None, "horizon_months": 2},
        {"mrr_pct": -2.0, "cash_pct": -4.5, "churn_pp": -0.1, "runway_months": None},
        month_label="last real month", source="observed",
    )
    months, _, meta = cycle_service.run_months(
        request(use_oracle=True, horizon=1, previous_track_record=record)
    )
    assert meta["started_from_track_record"] is True
    trace = months[0]["execute"]["trace"]
    assert trace["track_record"]["source"] == "observed"
    cmo = next(p for p in trace["proposals"] if p["agent"] == "CMO")
    assert cmo["adaptation"] is not None and "held back" in cmo["adaptation"]
    assert any("CMO:" in line for line in months[0]["adapt"]["what_changed"])


# --------------------------------------------------------------------------
# the API: async, streamed, honest on failure
# --------------------------------------------------------------------------

def test_api_runs_a_cycle_in_the_background_and_streams_months(db):
    from backend.main import app
    with TestClient(app) as client:
        created = client.post("/api/cycles", json=request(horizon=2))
        assert created.status_code == 202
        cycle_id = created.json()["id"]
        assert created.json()["status"] in ("queued", "running")
        done = wait_for(client, cycle_id)
        assert done["status"] == "completed"
        assert len(done["months"]) == 2
        assert done["summary"]["months_completed"] == 2
        assert done["months"][0]["latency_s"] >= 0
        listed = client.get("/api/companies/co_test/cycles").json()
        assert listed[0]["id"] == cycle_id
        assert client.get("/api/cycles/nope").status_code == 404


def test_api_rejects_a_horizon_outside_the_bounds(db):
    from backend.main import app
    with TestClient(app) as client:
        assert client.post("/api/cycles", json=request(horizon=0)).status_code == 422
        assert client.post("/api/cycles", json=request(horizon=7)).status_code == 422


def test_a_restart_fails_the_cycles_it_killed(db):
    """The row must never stay 'running' after the thread that owned it is gone."""
    from backend.main import app
    queued = cycle_service.create_cycle(request(horizon=1))
    assert queued["status"] == "queued"
    assert cycle_service.fail_orphaned_cycles() == 1
    with TestClient(app) as client:  # lifespan runs it again: nothing left to fail
        cycle = client.get(f"/api/cycles/{queued['id']}").json()
    assert cycle["status"] == "failed"
    assert "restarted" in cycle["error"]
    assert cycle_service.fail_orphaned_cycles() == 0


def test_a_failing_cycle_is_marked_failed_not_left_running(db, monkeypatch):
    def boom(*_, **__):
        raise RuntimeError("engine exploded")
    monkeypatch.setattr(cycle_service, "run_months", boom)
    cycle = cycle_service.create_cycle(request(horizon=1))
    done = cycle_service.run_cycle_sync(cycle["id"])
    assert done["status"] == "failed"
    assert "engine exploded" in done["error"]


# --------------------------------------------------------------------------
# the HITL close
# --------------------------------------------------------------------------

def close_with(client_or_none, cycle_id, per_action, mrr=31_900, cash=208_000, churn=4.6):
    return cycle_service.submit_feedback(cycle_id, {
        "month_index": 1,
        "per_action": per_action,
        "actuals": {"mrr": mrr, "cash": cash, "churn": churn, "costs": 47_000},
    })


def test_all_didnt_writes_zero_causal_edges_and_says_why(db, fake_oracle):
    cycle = cycle_service.create_cycle(request(use_oracle=True, horizon=1))
    cycle_service.run_cycle_sync(cycle["id"])
    graph = fake_oracle["graph"]
    writes_before = len(graph.writes)
    result = close_with(None, cycle["id"], [
        {"action_key": "marketing", "done": "didnt"},
        {"action_key": "product", "done": "didnt"},
    ])
    assert result["evidence"]["written"] is False
    assert result["evidence"]["edges"] == 0
    assert "not evidence" in result["evidence"]["reason"]
    assert len(graph.writes) == writes_before
    assert graph.edges_of("MAY_CAUSE") == []
    # The state observation still happened.
    assert result["memory"]["observed"] is True
    marketing = next(d for d in result["per_action"] if d["action_key"] == "marketing")
    assert marketing["counted"] is False and "didn't do this" in marketing["why"]


def test_done_writes_one_edge_per_kpi_and_promotes_nothing(db, fake_oracle):
    cycle = cycle_service.create_cycle(request(use_oracle=True, horizon=1))
    cycle_service.run_cycle_sync(cycle["id"])
    graph = fake_oracle["graph"]
    result = close_with(None, cycle["id"], [
        {"action_key": "marketing", "done": "did"},
        {"action_key": "product", "done": "did"},
    ])
    assert result["evidence"]["written"] is True
    scored = {k: v for k, v in result["actual_delta"].items() if v is not None}
    assert result["evidence"]["edges"] == len(scored)
    observed = graph.edges_of("MAY_CAUSE")
    assert len(observed) == len(scored)
    assert all(edge["observations"] == 1 for _, edge in observed)
    assert all(edge["source"] == "observed" for _, edge in observed)
    assert graph.confirmed == set(), "one confirmed month must not promote an edge"
    # Prediction error is computed against the real numbers and carried forward.
    assert result["prediction_error"]["mrr_pct"]["realized"] == pytest.approx(
        (31_900 - 30_000) / 30_000 * 100, abs=0.01)
    assert result["track_record"]["source"] == "observed"
    assert result["track_record"]["realized_delta"]["churn_pp"] == pytest.approx(4.6 - 5.0, abs=0.01)


def test_partly_halves_the_increment(db, fake_oracle):
    cycle = cycle_service.create_cycle(request(use_oracle=True, horizon=1))
    cycle_service.run_cycle_sync(cycle["id"])
    result = close_with(None, cycle["id"], [
        {"action_key": "marketing", "done": "did"},
        {"action_key": "product", "done": "partly", "note": "shipped half of it"},
    ])
    assert result["evidence"]["weight"] == 0.5
    assert "half weight" in result["evidence"]["reason"]
    assert fake_oracle["graph"].writes[-1]["weight"] == 0.5
    product = next(d for d in result["per_action"] if d["action_key"] == "product")
    assert product["note"] == "shipped half of it"


def test_the_close_puts_the_real_month_into_the_persisted_calendar(db, fake_oracle):
    cycle = cycle_service.create_cycle(request(use_oracle=True, horizon=4))
    cycle_service.run_cycle_sync(cycle["id"])
    assert load_oracle_state("co_test")["global_month"] == 3
    close_with(None, cycle["id"], [{"action_key": "marketing", "done": "did"}])
    saved = load_oracle_state("co_test")
    assert saved["global_month"] == 4
    assert saved["state_history"][-1]["source_month"] == 9
    assert saved["state_history"][-1]["mrr"] == 31_900
    stored = cycle_service.get_cycle(cycle["id"])
    assert stored["feedback"][0]["result"]["evidence"]["written"] is True


def test_feedback_api_round_trip_and_errors(db, fake_oracle):
    from backend.main import app
    cycle = cycle_service.create_cycle(request(use_oracle=True, horizon=1))
    cycle_service.run_cycle_sync(cycle["id"])
    with TestClient(app) as client:
        body = {"month_index": 1,
                "per_action": [{"action_key": "marketing", "done": "did"}],
                "actuals": {"mrr": 31_000, "cash": 210_000, "churn": 4.8}}
        ok = client.post(f"/api/cycles/{cycle['id']}/feedback", json=body)
        assert ok.status_code == 200
        assert ok.json()["prediction_error"]["mrr_pct"]["realized"] > 0
        assert client.post("/api/cycles/missing/feedback", json=body).status_code == 404
        bad = {**body, "month_index": 9}
        assert client.post(f"/api/cycles/{cycle['id']}/feedback", json=bad).status_code == 422
        bad_status = {**body, "per_action": [{"action_key": "marketing", "done": "maybe"}]}
        assert client.post(f"/api/cycles/{cycle['id']}/feedback", json=bad_status).status_code == 422
