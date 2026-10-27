"""Is the learning loop actually live? (plan section 2.2)

`Oracle.write_causal_outcome` returns silently unless a graph store is enabled,
and the graph store only exists when the advisor mode is oracle_v4_causal AND
Neo4j answers. Memory retrieval is only useful when it is scoped per company.
Neither failure raises. A demo that claims a learning loop while the graph is
unreachable is the worst outcome this project can have, so /api/health reports
each capability explicitly and the Settings page shows them.

The checks here are cheap probes (a TCP connect with a one-second timeout), not
full client handshakes, so the health endpoint stays fast when a service is
down. Each cycle additionally records the real `graph_store_enabled` flag from
the Oracle instance that ran it, which is the authoritative per-run answer.
"""

from __future__ import annotations

import os
import socket
import time
from typing import Any
from urllib.parse import urlparse

from backend import sim_profile

_CACHE: dict[str, Any] = {"at": 0.0, "value": None}
_CACHE_TTL_S = 15.0


def _tcp_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _neo4j_target() -> tuple[str, int]:
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    parsed = urlparse(uri)
    return parsed.hostname or "localhost", parsed.port or 7687


def _ollama_target() -> tuple[str, int]:
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    if "://" not in host:
        host = f"http://{host}"
    parsed = urlparse(host)
    return parsed.hostname or "127.0.0.1", parsed.port or 11434


def graph_store_status() -> dict[str, Any]:
    mode = sim_profile.get_oracle_mode()
    if mode != "oracle_v4_causal":
        return {
            "enabled": False,
            "reason": (
                f"advisor mode is {mode}; the causal graph only runs under "
                "oracle_v4_causal (SIM_PROFILE=founder)"
            ),
        }
    host, port = _neo4j_target()
    if not _tcp_open(host, port):
        return {"enabled": False, "reason": f"Neo4j is not answering at {host}:{port}"}
    return {"enabled": True, "reason": f"Neo4j reachable at {host}:{port}"}


def llm_status() -> dict[str, Any]:
    host, port = _ollama_target()
    if not _tcp_open(host, port):
        return {"reachable": False, "reason": f"Ollama is not answering at {host}:{port}"}
    return {"reachable": True, "reason": f"Ollama reachable at {host}:{port}"}


def memory_status() -> dict[str, Any]:
    try:
        import chromadb  # noqa: F401
        available = True
    except ImportError:
        available = False
    path = (
        sim_profile.FOUNDER_CHROMA_PATH
        if sim_profile.is_founder()
        else os.getenv("CHROMA_PATH", "./chroma_db")
    )
    return {
        "enabled": available,
        "path": path,
        # The pattern, not a specific company: health has no company in scope.
        # A per-request UUID here would mean founder memory is write-only.
        "scope": "company:<company_id>",
        "reason": (
            "one memory scope per company; a second analysis can read what the first matured"
            if available
            else "chromadb is not installed; memories are neither stored nor retrieved"
        ),
    }


def loop_status(force: bool = False) -> dict[str, Any]:
    now = time.monotonic()
    if not force and _CACHE["value"] is not None and now - _CACHE["at"] < _CACHE_TTL_S:
        return _CACHE["value"]
    graph = graph_store_status()
    llm = llm_status()
    memory = memory_status()
    value = {
        "advisor_mode": sim_profile.get_oracle_mode(),
        "sim_profile": sim_profile.get_profile(),
        # Which marketing-response curve the predictions run (decision 9):
        # "v2" is the CAL-fitted rate, "scale_aware" the assumed one the
        # product shipped with. Stated so the claim on stage matches the code.
        "marketing_curve": (
            sim_profile.FOUNDER_MARKETING_CURVE if sim_profile.is_founder() else "legacy"
        ),
        "graph_store_enabled": graph["enabled"],
        "graph_store_reason": graph["reason"],
        "memory_store_enabled": memory["enabled"],
        "memory_scope": memory["scope"],
        "memory_path": memory["path"],
        "memory_reason": memory["reason"],
        "llm_reachable": llm["reachable"],
        "llm_reason": llm["reason"],
        # Everything the loop needs to be honest about: Feedback writes need the
        # graph, Adapt needs memory, Execute needs the LLM for a fresh brief.
        "loop_live": bool(graph["enabled"] and memory["enabled"] and llm["reachable"]),
    }
    _CACHE["at"] = now
    _CACHE["value"] = value
    return value
