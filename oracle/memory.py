import os
import uuid
from typing import Any, Dict, List, Optional

from env.schemas import EnvState
from oracle.schemas import (
    ExpectedOutcome,
    PendingMemoryEntry,
    RetrievedMemoryCandidate,
    StateSnapshot,
    TrendContext,
)

try:
    import chromadb
except ImportError:
    chromadb = None

MEMORY_COLLECTION_NAME = "oracle_live_memories"
MEMORY_HORIZON_MONTHS = 6
MEMORY_QUERY_CANDIDATES = 10
MEMORY_PROMPT_LIMIT = 3
RECENCY_DECAY_MONTHS = 120.0
OUTCOME_GROWTH_THRESHOLD = 0.10
OUTCOME_DECLINE_THRESHOLD = -0.10


def classify_realized_outcome(source_mrr: float, current_mrr: float) -> ExpectedOutcome:
    baseline = max(abs(source_mrr), 1.0)
    pct_change = (current_mrr - source_mrr) / baseline
    if pct_change > OUTCOME_GROWTH_THRESHOLD:
        return ExpectedOutcome.GROWTH
    if pct_change < OUTCOME_DECLINE_THRESHOLD:
        return ExpectedOutcome.DECLINE
    return ExpectedOutcome.STAGNATION


def format_memory_document(
    snapshot: StateSnapshot,
    trend_context: TrendContext,
    realized_outcome: ExpectedOutcome,
) -> str:
    return (
        f"Episode month {snapshot.source_month}: MRR {snapshot.mrr:,.0f}, "
        f"avg churn {snapshot.avg_churn:.3f}, innovation {snapshot.innovation:.3f}. "
        f"Trends were MRR {trend_context.mrr_trend.value}, innovation {trend_context.innovation_trend.value}, "
        f"churn {trend_context.churn_trend.value}. "
        f"After 6 months the realized outcome was {realized_outcome.value}."
    )


def build_memory_query(state: EnvState, trend_context: TrendContext) -> str:
    avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0
    return (
        f"MRR {state.mrr:,.0f}; avg churn {avg_churn:.3f}; innovation {state.innovation_factor:.3f}; "
        f"MRR trend {trend_context.mrr_trend.value}; innovation trend {trend_context.innovation_trend.value}; "
        f"churn trend {trend_context.churn_trend.value}"
    )


class OracleMemoryStore:
    def __init__(
        self,
        run_id: str,
        chroma_path: Optional[str] = None,
        collection: Any = None,
    ):
        self.run_id = run_id
        self.collection = collection
        self.enabled = collection is not None

        if self.collection is not None:
            return

        if chromadb is None:
            return

        chroma_path = chroma_path or os.getenv("CHROMA_PATH", "./chroma_db")
        try:
            client = chromadb.PersistentClient(path=chroma_path)
            self.collection = client.get_or_create_collection(name=MEMORY_COLLECTION_NAME)
            self.enabled = True
        except Exception as exc:
            print(f"[OracleMemoryStore] Chroma disabled: {exc}")
            self.collection = None
            self.enabled = False

    def store_memory(
        self,
        pending_entry: PendingMemoryEntry,
        stored_global_month: int,
        realized_outcome: ExpectedOutcome,
    ) -> None:
        if not self.enabled or self.collection is None:
            return

        metadata: Dict[str, Any] = {
            "run_id": self.run_id,
            "stored_global_month": stored_global_month,
            "source_month": pending_entry.snapshot.source_month,
            "realized_outcome": realized_outcome.value,
        }
        if pending_entry.snapshot.episode_seed is not None:
            metadata["episode_seed"] = pending_entry.snapshot.episode_seed

        document = format_memory_document(
            pending_entry.snapshot,
            pending_entry.trend_context,
            realized_outcome,
        )

        try:
            self.collection.add(
                documents=[document],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())],
            )
        except Exception as exc:
            print(f"[OracleMemoryStore] Failed to store memory: {exc}")

    def retrieve_similar(
        self,
        state: EnvState,
        trend_context: TrendContext,
        current_global_month: int,
        limit: int = MEMORY_PROMPT_LIMIT,
    ) -> List[RetrievedMemoryCandidate]:
        if not self.enabled or self.collection is None:
            return []

        query_text = build_memory_query(state, trend_context)
        try:
            query_result = self.collection.query(
                query_texts=[query_text],
                n_results=MEMORY_QUERY_CANDIDATES,
                where={"run_id": self.run_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            print(f"[OracleMemoryStore] Failed to query memory: {exc}")
            return []

        candidates = self._rerank_candidates(query_result, current_global_month)
        return candidates[:limit]

    def _rerank_candidates(
        self,
        query_result: Dict[str, Any],
        current_global_month: int,
    ) -> List[RetrievedMemoryCandidate]:
        documents = (query_result.get("documents") or [[]])[0]
        metadatas = (query_result.get("metadatas") or [[]])[0]
        distances = (query_result.get("distances") or [[]])[0]

        candidates: List[RetrievedMemoryCandidate] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] if index < len(metadatas) and metadatas[index] is not None else {}
            distance = float(distances[index]) if index < len(distances) else 1.0
            similarity_score = 1.0 / (1.0 + max(distance, 0.0))
            stored_global_month = int(metadata.get("stored_global_month", current_global_month))
            age_months = max(0, current_global_month - stored_global_month)
            recency_factor = 1.0 / (1.0 + (age_months / RECENCY_DECAY_MONTHS))
            memory_weight = similarity_score * recency_factor

            candidates.append(
                RetrievedMemoryCandidate(
                    document=document,
                    metadata=metadata,
                    distance=distance,
                    similarity_score=similarity_score,
                    recency_factor=recency_factor,
                    memory_weight=memory_weight,
                )
            )

        candidates.sort(key=lambda item: item.memory_weight, reverse=True)
        return candidates
