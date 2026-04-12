from collections import deque
import uuid

from env.schemas import EnvState
from oracle.context import compute_trend_context, snapshot_state
from oracle.memory import MEMORY_HORIZON_MONTHS, OracleMemoryStore, classify_realized_outcome
from oracle.prompt_builder import build_prompt
from oracle.parser import parse_llm_response
from oracle.schemas import OracleBrief, PendingMemoryEntry, RetrievedMemoryCandidate, TrendContext

class DummyLLMClient:
    """Fallback structural placeholder until proper LLM is wired."""
    def generate(self, prompt: str) -> str:
        print("[WARNING] DummyLLMClient used! `ollama` package might not be installed, yielding identical metrics.")
        return '{"risk_level":"MEDIUM","growth_outlook":"STABLE","efficiency_pressure":"MEDIUM","innovation_urgency":"MEDIUM","macro_condition":"NEUTRAL","key_risks":[],"key_opportunities":[],"recommended_focus":[],"confidence":0.5}'

try:
    from agents.llm_client import LLMClient
except ImportError:
    LLMClient = DummyLLMClient

class Oracle:
    def __init__(
        self,
        mode: str = "oracle_v1",
        run_id: str | None = None,
        memory_store: OracleMemoryStore | None = None,
        llm=None,
        enable_memory_retrieval: bool = True,
    ):
        self.mode = mode
        self.run_id = run_id or str(uuid.uuid4())
        self.llm = llm or LLMClient()
        self.enable_memory_retrieval = enable_memory_retrieval
        self.state_history = deque(maxlen=5)
        self.pending_memories = deque()
        self.global_month = 0
        self.current_episode_seed = None
        self.latest_snapshot = None
        self.latest_trend_context = TrendContext()
        self.memory_store = memory_store if memory_store is not None else None
        if self.memory_store is None and self.mode == "oracle_v3" and self.enable_memory_retrieval:
            self.memory_store = OracleMemoryStore(run_id=self.run_id)

    def start_episode(self, episode_seed: int | None = None) -> None:
        self.current_episode_seed = episode_seed
        self.state_history.clear()
        self.pending_memories.clear()
        self.latest_snapshot = None
        self.latest_trend_context = TrendContext()

    def observe_state(self, state: EnvState, episode_seed: int | None = None) -> None:
        if episode_seed is not None:
            self.current_episode_seed = episode_seed

        snapshot = snapshot_state(
            state,
            global_month=self.global_month,
            episode_seed=self.current_episode_seed,
        )
        self.state_history.append(snapshot)
        self.latest_snapshot = snapshot
        self.latest_trend_context = compute_trend_context(list(self.state_history))

        self._mature_pending_memories(snapshot)
        self.pending_memories.append(
            PendingMemoryEntry(
                snapshot=snapshot,
                trend_context=self.latest_trend_context,
            )
        )
        self.global_month += 1

    def get_context(
        self,
        state: EnvState,
    ) -> tuple[TrendContext, list[RetrievedMemoryCandidate], int]:
        trend_context = self.latest_trend_context
        current_global_month = self.latest_snapshot.global_month if self.latest_snapshot is not None else self.global_month

        if self.latest_snapshot is None or self.latest_snapshot.source_month != state.months_elapsed:
            temp_snapshot = snapshot_state(state, current_global_month, self.current_episode_seed)
            history = list(self.state_history)
            if not history or history[-1].global_month != temp_snapshot.global_month:
                history.append(temp_snapshot)
            trend_context = compute_trend_context(history)

        memories = []
        if self.mode == "oracle_v3" and self.enable_memory_retrieval and self.memory_store is not None:
            memories = self.memory_store.retrieve_similar(
                state=state,
                trend_context=trend_context,
                current_global_month=current_global_month,
            )

        return trend_context, memories, current_global_month
        
    def generate_brief(
        self,
        state: EnvState,
        trend_context: TrendContext | None = None,
        memories: list[RetrievedMemoryCandidate] | None = None,
        shock_label: str | None = None,
    ) -> OracleBrief:
        """
        Pure function: interprets state, calls LLM, and parses into OracleBrief.
        Does NOT store to memory here.
        """
        if trend_context is None or memories is None:
            resolved_trend_context, resolved_memories, _ = self.get_context(state)
            if trend_context is None:
                trend_context = resolved_trend_context
            if memories is None:
                memories = resolved_memories

        prompt = build_prompt(
            state,
            mode=self.mode,
            trend_context=trend_context,
            memories=memories,
            shock_label=shock_label,
        )
        
        # Safely invoke LLM regardless of actual class method structure implementations
        if hasattr(self.llm, 'complete'):
            raw_output = self.llm.complete("You are a strategic SaaS oracle. Only output perfect JSON.", prompt)
        elif hasattr(self.llm, 'generate'):
            raw_output = self.llm.generate(prompt)
        elif hasattr(self.llm, 'call'):
            raw_output = self.llm.call(prompt)
        else:
            raw_output = DummyLLMClient().generate(prompt)
            
        if not raw_output:
            print("[WARNING] LLMClient returned completely empty output. Ollama might be offline. Using fallback.")
            
        brief = parse_llm_response(str(raw_output))
        return brief

    def build_cache_key(
        self,
        state: EnvState,
        trend_context: TrendContext | None = None,
        memories: list[RetrievedMemoryCandidate] | None = None,
    ) -> tuple[str, ...]:
        if trend_context is None or memories is None:
            resolved_trend_context, resolved_memories, _ = self.get_context(state)
            if trend_context is None:
                trend_context = resolved_trend_context
            if memories is None:
                memories = resolved_memories

        mrr_bracket = int(state.mrr / 50_000)
        runway_bracket = int(self._estimate_runway_months(state) / 3)
        competitor_count = state.competitors
        confidence_bracket = int(state.consumer_confidence / 10)
        shock_flag = self._detect_shock(state)

        return (
            self.mode,
            str(mrr_bracket),
            str(runway_bracket),
            str(competitor_count),
            str(confidence_bracket),
            trend_context.mrr_trend.value,
            shock_flag,
            self._build_memory_signature(memories),
        )

    def end_episode(self) -> None:
        """
        Called at the end of each episode to force-mature all remaining memories.
        """
        if self.memory_store is None or self.latest_snapshot is None:
            return

        while self.pending_memories:
            entry = self.pending_memories.popleft()
            realized = classify_realized_outcome(
                source_mrr=entry.snapshot.mrr,
                current_mrr=self.latest_snapshot.mrr,
            )
            self.memory_store.store_memory(
                pending_entry=entry,
                stored_global_month=self.latest_snapshot.global_month,
                realized_outcome=realized,
            )

    def _detect_shock(self, state: EnvState) -> str:
        flags = []
        if state.consumer_confidence < 70:
            flags.append("LOW_CONF")
        if state.interest_rate > 7.0:
            flags.append("HIGH_RATES")
        if state.competitors > 8:
            flags.append("CROWDED_MKT")
        if state.months_in_depression > 3:
            flags.append("DEPRESSION")
        return "|".join(flags) if flags else "NORMAL"

    def _mature_pending_memories(self, current_snapshot) -> None:
        if self.memory_store is None:
            return

        while self.pending_memories:
            oldest_entry = self.pending_memories[0]
            age_months = current_snapshot.global_month - oldest_entry.snapshot.global_month
            if age_months < MEMORY_HORIZON_MONTHS:
                break

            pending_entry = self.pending_memories.popleft()
            realized_outcome = classify_realized_outcome(
                source_mrr=pending_entry.snapshot.mrr,
                current_mrr=current_snapshot.mrr,
            )
            self.memory_store.store_memory(
                pending_entry=pending_entry,
                stored_global_month=current_snapshot.global_month,
                realized_outcome=realized_outcome,
            )

    @staticmethod
    def _estimate_runway_months(state: EnvState) -> float:
        monthly_burn_estimate = max(1.0, float(state.headcount * 8000.0))
        return state.cash / monthly_burn_estimate

    def _build_memory_signature(self, memories: list[RetrievedMemoryCandidate] | None) -> str:
        if self.mode != "oracle_v3" or not memories:
            return "none"

        signature_parts = []
        for memory in memories[:2]:
            metadata = memory.metadata or {}
            source_month = metadata.get("source_month", "na")
            realized_outcome = metadata.get("realized_outcome", "UNKNOWN")
            signature_parts.append(f"{source_month}:{realized_outcome}")

        return "|".join(signature_parts) if signature_parts else "none"
