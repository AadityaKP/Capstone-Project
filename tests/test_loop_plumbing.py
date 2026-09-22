"""Phase 0.5 of docs/oefa_loop_plan.md: the loop's plumbing is real.

Three things were silently not happening before this phase, and each gets a
test that fails if it stops happening again:

  1. founder memory was write-only (a fresh run_id per request);
  2. the causal write was a no-op with no signal that it was;
  3. simulated and observed evidence shared one edge and one counter.

Plus the state-ownership decision (2.4a): the Oracle's per-company calendar
survives between requests.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from backend import sim_profile
from backend.database import initialize_database
from env.schemas import EnvState
from oracle.memory import OracleMemoryStore
from oracle.oracle import Oracle
from tests.loop_fakes import FakeCollection, FakeLLM, InMemoryGraphStore, recording_graph_store


def state(mrr=30_000, cash=200_000, costs=40_000, churn=0.05, month=8):
    return EnvState(
        mrr=mrr, cash=cash, cac=90, ltv=85 / churn,
        churn_enterprise=churn, churn_smb=churn, churn_b2c=churn,
        interest_rate=3, consumer_confidence=100, competitors=5,
        product_quality=0.5, price=85, months_elapsed=month,
        headcount=4, monthly_burn=costs,
    )


ACTION = {
    "marketing": {"spend": 6_000.0, "channel": "ppc"},
    "hiring": {"hires": 0, "cost_per_employee": 10_000},
    "product": {"r_and_d_spend": 9_000.0},
    "pricing": {"price_change_pct": 0.0},
}


# --------------------------------------------------------------------------
# 2.1 memory scope
# --------------------------------------------------------------------------

def test_memory_scope_is_stable_per_company():
    assert sim_profile.get_memory_scope("co_1") == "company:co_1"
    assert sim_profile.get_memory_scope("co_1") == sim_profile.get_memory_scope("co_1")
    assert sim_profile.get_memory_scope("co_1") != sim_profile.get_memory_scope("co_2")


def test_no_company_keeps_the_old_isolation_rather_than_pooling():
    assert sim_profile.get_memory_scope(None) != sim_profile.get_memory_scope(None)


def test_founder_oracle_kwargs_scope_store_and_oracle_to_the_company(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM_PROFILE", "founder")
    monkeypatch.setattr(sim_profile, "FOUNDER_CHROMA_PATH", str(tmp_path / "chroma"))
    kwargs = sim_profile.get_oracle_kwargs(company_id="co_9")
    assert kwargs["run_id"] == "company:co_9"
    assert kwargs["memory_store"].run_id == "company:co_9"
    assert kwargs["dedupe_months"] is True


def test_review2_oracle_kwargs_scope_but_keep_the_research_store(monkeypatch):
    monkeypatch.setenv("SIM_PROFILE", "review2")
    kwargs = sim_profile.get_oracle_kwargs(company_id="co_9")
    assert kwargs == {"run_id": "company:co_9", "dedupe_months": True}


def test_a_second_analysis_can_read_what_the_first_matured():
    """The five-minute check from plan section 2.1, hermetically: one company,
    one shared collection, two Oracle instances that persist through
    export/import. The second retrieves a memory the first wrote."""
    collection = FakeCollection()
    scope = "company:co_1"

    first = Oracle(mode="oracle_v3", run_id=scope, dedupe_months=True,
                   memory_store=OracleMemoryStore(run_id=scope, collection=collection),
                   llm=FakeLLM())
    # Seven months of history: the oldest matures at the 6-month horizon and
    # is written with source_month >= 3, which store_memory requires.
    for month in range(3, 10):
        first.observe_state(state(mrr=20_000 + month * 1_000, month=month))
    assert collection.count() >= 1
    assert all(meta["run_id"] == scope for meta in collection.metadatas)

    second = Oracle(mode="oracle_v3", run_id=scope, dedupe_months=True,
                    memory_store=OracleMemoryStore(run_id=scope, collection=collection),
                    llm=FakeLLM())
    second.import_state(first.export_state())
    second.observe_state(state(mrr=31_000, month=10))
    _, memories, _, _ = second.get_context(state(mrr=31_000, month=10))
    assert len(memories) >= 1

    other = Oracle(mode="oracle_v3", run_id="company:someone_else",
                   memory_store=OracleMemoryStore(run_id="company:someone_else", collection=collection),
                   llm=FakeLLM())
    other.observe_state(state(mrr=31_000, month=10))
    _, foreign, _, _ = other.get_context(state(mrr=31_000, month=10))
    assert foreign == [], "one company must never read another company's memory"


# --------------------------------------------------------------------------
# 2.4 / 4.1 persisted Oracle calendar
# --------------------------------------------------------------------------

def test_export_import_round_trip_keeps_the_pending_queue():
    oracle = Oracle(mode="oracle_v3", run_id="company:x", enable_memory_retrieval=False, llm=FakeLLM())
    for month in range(5, 9):
        oracle.observe_state(state(month=month))
    exported = oracle.export_state()
    assert exported["global_month"] == 4
    assert len(exported["pending_memories"]) == 4

    fresh = Oracle(mode="oracle_v3", run_id="company:x", enable_memory_retrieval=False, llm=FakeLLM())
    fresh.import_state(exported)
    assert fresh.global_month == 4
    assert len(fresh.pending_memories) == 4
    assert fresh.latest_snapshot.source_month == 8
    assert fresh.export_state() == exported


def test_same_month_observed_twice_replaces_rather_than_duplicates():
    oracle = Oracle(mode="oracle_v3", run_id="company:x", enable_memory_retrieval=False,
                    dedupe_months=True, llm=FakeLLM())
    oracle.observe_state(state(mrr=30_000, month=8))
    oracle.observe_state(state(mrr=31_000, month=8))
    assert oracle.global_month == 1
    assert len(oracle.state_history) == 1
    assert len(oracle.pending_memories) == 1
    assert oracle.latest_snapshot.mrr == 31_000


def test_research_oracle_still_appends_duplicates_by_default():
    oracle = Oracle(mode="oracle_v3", enable_memory_retrieval=False, llm=FakeLLM())
    oracle.observe_state(state(month=8))
    oracle.observe_state(state(month=8))
    assert oracle.global_month == 2
    assert len(oracle.state_history) == 2


# --------------------------------------------------------------------------
# 2.2 the causal write says whether it happened
# --------------------------------------------------------------------------

def test_causal_write_reports_false_when_no_graph_store():
    oracle = Oracle(mode="oracle_v3", enable_memory_retrieval=False, llm=FakeLLM())
    assert oracle.graph_store_enabled is False
    assert oracle.write_causal_outcome(ACTION, {"mrr_pct": 2.0}, source="sim") is False


def test_causal_write_reports_true_when_it_lands():
    graph = InMemoryGraphStore()
    oracle = Oracle(mode="oracle_v4_causal", graph_store=graph,
                    enable_memory_retrieval=False, llm=FakeLLM())
    assert oracle.graph_store_enabled is True
    assert oracle.write_causal_outcome(ACTION, {"mrr_pct": 2.0, "churn_pp": -0.1}, source="observed") is True
    assert len(graph.writes) == 1


def test_health_reports_the_loop_capabilities(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
    monkeypatch.setenv("SIM_PROFILE", "review2")
    initialize_database()
    from backend import loop_status
    loop_status._CACHE["value"] = None
    from backend.main import app

    with TestClient(app) as client:
        body = client.get("/api/health").json()
    loop = body["loop"]
    for key in ("graph_store_enabled", "graph_store_reason", "memory_scope",
                "memory_store_enabled", "llm_reachable", "loop_live"):
        assert key in loop
    # Under review2 the mode is oracle_v3, so the graph is off by construction
    # and the reason says why rather than blaming Neo4j.
    assert loop["graph_store_enabled"] is False
    assert "oracle_v3" in loop["graph_store_reason"]
    assert loop["memory_scope"] == "company:<company_id>"


# --------------------------------------------------------------------------
# 2.3 evidence provenance
# --------------------------------------------------------------------------

def _cypher_for(store, **kwargs):
    store.write_action_outcome(action=ACTION, kpi_delta={"mrr_pct": 3.0}, **kwargs)
    return store.driver.log


def test_simulated_evidence_goes_to_its_own_edge_at_reduced_weight_and_never_promotes():
    log = _cypher_for(recording_graph_store(), source="sim")
    assert len(log) == 1, "no promotion query for simulated evidence"
    cypher, params = log[0]
    assert "[r:MAY_CAUSE_SIM]" in cypher
    assert "MAY_CAUSE]" not in cypher
    assert params["source"] == "sim"
    assert params["confidence_increment"] == pytest.approx(0.02)


def test_observed_evidence_keeps_the_full_increment_and_can_promote():
    log = _cypher_for(recording_graph_store(), source="observed")
    assert len(log) == 2
    write, promote = log
    assert "[r:MAY_CAUSE]" in write[0]
    assert write[1]["source"] == "observed"
    assert write[1]["confidence_increment"] == pytest.approx(0.05)
    assert "CONFIRMED_CAUSE" in promote[0]


def test_partial_credit_scales_the_increment_not_the_edge():
    log = _cypher_for(recording_graph_store(), source="observed", weight=0.5)
    assert "[r:MAY_CAUSE]" in log[0][0]
    assert log[0][1]["confidence_increment"] == pytest.approx(0.025)


def test_legacy_research_write_is_byte_identical():
    """No source: MAY_CAUSE, 0.05/-0.03, no provenance property, promotion
    query present - exactly what every recorded run wrote."""
    log = _cypher_for(recording_graph_store())
    assert len(log) == 2
    assert "[r:MAY_CAUSE]" in log[0][0]
    assert "r.source" not in log[0][0]
    assert "source" not in log[0][1]
    assert log[0][1]["confidence_increment"] == pytest.approx(0.05)
    negative = recording_graph_store()
    negative.write_action_outcome(action=ACTION, kpi_delta={"mrr_pct": -3.0})
    assert negative.driver.log[0][1]["confidence_increment"] == pytest.approx(-0.03)


def test_unknown_source_is_refused_loudly():
    with pytest.raises(ValueError):
        recording_graph_store().write_action_outcome(action=ACTION, kpi_delta={"mrr_pct": 1.0}, source="guess")


def test_one_call_writes_one_edge_per_kpi_and_promotes_nothing_on_first_observation():
    graph = InMemoryGraphStore()
    graph.write_action_outcome(ACTION, {"mrr_pct": 3.0, "churn_pp": -0.2, "cash_pct": -4.0}, source="observed")
    assert len(graph.edges_of("MAY_CAUSE")) == 3
    assert graph.confirmed == set()
    # Five positive observations minimum: 0.6 base + 5 x 0.05 = 0.85.
    for _ in range(4):
        graph.write_action_outcome(ACTION, {"mrr_pct": 3.0}, source="observed")
    assert ("marketing_mid|rd_mid|hires_0|price_flat", "mrr_pct_Up") in graph.confirmed


def test_simulated_evidence_never_reaches_confirmed_cause_however_often_it_repeats():
    graph = InMemoryGraphStore()
    for _ in range(40):
        graph.write_action_outcome(ACTION, {"mrr_pct": 3.0}, source="sim")
    assert graph.confirmed == set()
    assert graph.edges_of("MAY_CAUSE") == []
    assert len(graph.edges_of("MAY_CAUSE_SIM")) == 1
