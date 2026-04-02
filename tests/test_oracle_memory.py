import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.schemas import EnvState
from oracle.memory import OracleMemoryStore
from oracle.oracle import Oracle
from oracle.schemas import ExpectedOutcome, PendingMemoryEntry, TrendContext


class FakeCollection:
    def __init__(self, query_result=None):
        self.query_result = query_result or {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        self.add_calls = []
        self.query_calls = []

    def add(self, documents, metadatas, ids):
        self.add_calls.append(
            {
                "documents": documents,
                "metadatas": metadatas,
                "ids": ids,
            }
        )

    def query(self, query_texts, n_results, where, include):
        self.query_calls.append(
            {
                "query_texts": query_texts,
                "n_results": n_results,
                "where": where,
                "include": include,
            }
        )
        return self.query_result


class RecordingMemoryStore:
    def __init__(self):
        self.records = []

    def store_memory(self, pending_entry, stored_global_month, realized_outcome):
        self.records.append(
            {
                "pending_entry": pending_entry,
                "stored_global_month": stored_global_month,
                "realized_outcome": realized_outcome,
            }
        )

    def retrieve_similar(self, state, trend_context, current_global_month):
        return []


class StaticLLM:
    def complete(self, system_prompt, user_prompt):
        return (
            '{"risk_level":"MEDIUM","growth_outlook":"STABLE","efficiency_pressure":"MEDIUM",'
            '"innovation_urgency":"MEDIUM","macro_condition":"NEUTRAL","confidence":0.8}'
        )


def make_state(months_elapsed: int, mrr: float) -> EnvState:
    return EnvState(
        mrr=mrr,
        cash=1_000_000.0,
        cac=100.0,
        ltv=700.0,
        churn_enterprise=0.02,
        churn_smb=0.03,
        churn_b2c=0.04,
        interest_rate=3.0,
        consumer_confidence=100.0,
        competitors=5,
        product_quality=0.5,
        price=50.0,
        months_elapsed=months_elapsed,
        headcount=5,
        valuation_multiple=10.0,
        unemployment=4.0,
        innovation_factor=0.6,
        months_in_depression=0,
    )


def test_store_memory_adds_run_scoped_metadata():
    collection = FakeCollection()
    store = OracleMemoryStore(run_id="run-123", collection=collection)
    pending_entry = PendingMemoryEntry(
        snapshot={
            "global_month": 2,
            "source_month": 2,
            "episode_seed": 9,
            "mrr": 120_000,
            "avg_churn": 0.031,
            "innovation": 0.55,
        },
        trend_context=TrendContext(),
    )

    store.store_memory(pending_entry, stored_global_month=8, realized_outcome=ExpectedOutcome.DECLINE)

    metadata = collection.add_calls[0]["metadatas"][0]
    assert metadata["run_id"] == "run-123"
    assert metadata["stored_global_month"] == 8
    assert metadata["source_month"] == 2
    assert metadata["episode_seed"] == 9
    assert metadata["realized_outcome"] == "DECLINE"


def test_retrieve_similar_reranks_by_similarity_and_recency():
    collection = FakeCollection(
        query_result={
            "documents": [[
                "recent-similar",
                "old-equal-similarity",
                "older-but-more-similar",
                "weak-recent",
            ]],
            "metadatas": [[
                {"stored_global_month": 100, "run_id": "run-123"},
                {"stored_global_month": 10, "run_id": "run-123"},
                {"stored_global_month": 50, "run_id": "run-123"},
                {"stored_global_month": 108, "run_id": "run-123"},
            ]],
            "distances": [[0.2, 0.2, 0.01, 0.8]],
        }
    )
    store = OracleMemoryStore(run_id="run-123", collection=collection)

    memories = store.retrieve_similar(make_state(4, 150_000), TrendContext(), current_global_month=110)

    assert collection.query_calls[0]["where"] == {"run_id": "run-123"}
    assert len(memories) == 3
    assert [memory.document for memory in memories] == [
        "recent-similar",
        "older-but-more-similar",
        "weak-recent",
    ]
    assert memories[1].memory_weight > memories[2].memory_weight


def test_oracle_delays_memory_until_horizon_and_labels_realized_outcome():
    recording_store = RecordingMemoryStore()
    oracle = Oracle(mode="oracle_v3", memory_store=recording_store, llm=StaticLLM(), run_id="run-xyz")
    oracle.start_episode(episode_seed=42)

    for month, mrr in enumerate([100_000, 101_000, 102_000, 103_000, 104_000, 105_000]):
        oracle.observe_state(make_state(month, mrr))

    assert recording_store.records == []

    oracle.observe_state(make_state(6, 120_000))

    assert len(recording_store.records) == 1
    record = recording_store.records[0]
    assert record["stored_global_month"] == 6
    assert record["realized_outcome"] == ExpectedOutcome.GROWTH
    assert record["pending_entry"].snapshot.episode_seed == 42


def test_end_episode_force_matures_remaining_pending_memories():
    recording_store = RecordingMemoryStore()
    oracle = Oracle(mode="oracle_v3", memory_store=recording_store, llm=StaticLLM(), run_id="run-xyz")
    oracle.start_episode(episode_seed=9)

    oracle.observe_state(make_state(0, 100_000))
    oracle.observe_state(make_state(1, 95_000))
    oracle.end_episode()

    assert len(recording_store.records) == 2
    assert len(oracle.pending_memories) == 0
