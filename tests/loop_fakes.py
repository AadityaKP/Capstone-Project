"""Test doubles for the OEFA loop: an in-memory memory collection, an
in-memory causal graph that mirrors CausalGraphStore's edge and promotion
rules, a recording Neo4j driver, and a canned LLM.

Ollama and Neo4j are both optional services; the loop's tests must prove the
plumbing with neither running, so every external surface has a double here.
"""

from __future__ import annotations

import json
from typing import Any


class FakeLLM:
    """Returns one valid brief every time, and counts calls."""

    def __init__(self, brief: dict[str, Any] | None = None):
        self.calls = 0
        self.brief = brief or {
            "risk_level": "HIGH", "growth_outlook": "DECLINING",
            "efficiency_pressure": "MEDIUM", "innovation_urgency": "HIGH",
            "macro_condition": "NEUTRAL", "expected_outcome": "STAGNATION",
            "key_risks": ["Churn is high for this stage"],
            "key_opportunities": ["Retention work has room"],
            "recommended_focus": ["Protect retention"],
            "confidence": 0.7,
        }

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        return json.dumps(self.brief)


class FakeCollection:
    """Enough of a chromadb collection for OracleMemoryStore: add() and a
    query() that filters on the `where` metadata and returns everything
    (distance 0.5) in insertion order."""

    def __init__(self):
        self.documents: list[str] = []
        self.metadatas: list[dict[str, Any]] = []
        self.ids: list[str] = []

    def add(self, documents, metadatas, ids):
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

    def count(self) -> int:
        return len(self.documents)

    def query(self, query_texts, n_results, where=None, include=None):
        hits = [
            (doc, meta)
            for doc, meta in zip(self.documents, self.metadatas)
            if not where or all(meta.get(k) == v for k, v in where.items())
        ][:n_results]
        return {
            "documents": [[doc for doc, _ in hits]],
            "metadatas": [[meta for _, meta in hits]],
            "distances": [[0.5 for _ in hits]],
        }


class InMemoryGraphStore:
    """A causal graph store that keeps edges in a dict and applies exactly the
    rules CausalGraphStore.write_action_outcome applies in Cypher: one edge per
    KPI, source-routed relationship type and increment, confidence clamp, and
    promotion to CONFIRMED_CAUSE at >= 0.85 confidence and >= 3 positive
    observations - observed evidence only."""

    enabled = True

    def __init__(self):
        from oracle.graph_store import CausalGraphStore, evidence_rule

        self._name = CausalGraphStore._action_pattern_name
        self._delta_name = CausalGraphStore._kpi_delta_name
        self._positive = CausalGraphStore._is_positive_delta
        self._rule = evidence_rule
        self.edges: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.confirmed: set[tuple[str, str]] = set()
        self.writes: list[dict[str, Any]] = []

    # --- the surface Oracle uses ---
    def write_action_outcome(self, action, kpi_delta, confidence=0.6, stress_node=None,
                             episode_id=None, month=None, source=None, weight=1.0):
        rule = self._rule(source)
        self.writes.append({"action": action, "kpi_delta": dict(kpi_delta),
                            "source": source, "weight": weight, "stress_node": stress_node})
        action_name = self._name(action)
        for metric, delta in kpi_delta.items():
            if delta is None:
                continue
            delta_value = float(delta)
            key = (rule["relationship"], action_name, self._delta_name(metric, delta_value))
            positive = self._positive(metric, delta_value)
            increment = (rule["positive"] if positive else rule["negative"]) * max(0.0, float(weight))
            edge = self.edges.setdefault(key, {"observations": 0, "positive_observations": 0,
                                               "confidence": float(confidence), "source": source})
            edge["observations"] += 1
            edge["positive_observations"] += int(positive)
            edge["confidence"] = min(0.95, max(0.05, edge["confidence"] + increment))
            edge["last_delta"] = delta_value
            if rule["promotable"] and edge["confidence"] >= 0.85 and edge["positive_observations"] >= 3:
                self.confirmed.add((action_name, key[2]))

    def query_role_causal_context(self, stress_node, role, limit=3):
        from oracle.schemas import CausalGraphContext

        return CausalGraphContext(role=role, stress_node=stress_node, chain_summary="", confidence=0.0)

    def build_graph_context(self, shock_type=None, mrr_tier=None):
        from oracle.schemas import GraphContext

        return GraphContext()

    def write_shock_event(self, **kwargs):
        pass

    def write_outcome(self, **kwargs):
        pass

    def write_episode(self, episode_metrics):
        pass

    # --- inspection ---
    def edges_of(self, relationship: str) -> list[tuple[tuple[str, str, str], dict[str, Any]]]:
        return [(k, v) for k, v in self.edges.items() if k[0] == relationship]


class RecordingSession:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, cypher, params=None):
        self.log.append((cypher, dict(params or {})))
        return []


class RecordingDriver:
    """Captures every Cypher statement CausalGraphStore would send to Neo4j."""

    def __init__(self):
        self.log: list[tuple[str, dict[str, Any]]] = []

    def session(self):
        return RecordingSession(self.log)

    def close(self):
        pass


def recording_graph_store():
    """A real CausalGraphStore whose driver records instead of connecting."""
    from oracle.graph_store import CausalGraphStore

    store = CausalGraphStore.__new__(CausalGraphStore)
    store.enabled = True
    store.driver = RecordingDriver()
    return store
