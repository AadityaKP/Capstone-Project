from collections import deque
import uuid

from env.schemas import EnvState
from oracle.context import compute_trend_context, get_mrr_tier, snapshot_state
from oracle.memory import MEMORY_HORIZON_MONTHS, OracleMemoryStore, classify_realized_outcome
from oracle.prompt_builder import build_prompt
from oracle.parser import parse_llm_response
from oracle.schemas import (
    CausalGraphContext,
    GraphContext,
    OracleBrief,
    PendingMemoryEntry,
    RetrievedMemoryCandidate,
    TrendContext,
)


class DummyLLMClient:
    """Fallback structural placeholder until proper LLM is wired."""

    def generate(self, prompt: str) -> str:
        print("[WARNING] DummyLLMClient used!")
        return (
            '{"risk_level":"MEDIUM","growth_outlook":"STABLE",'
            '"efficiency_pressure":"MEDIUM","innovation_urgency":"MEDIUM",'
            '"macro_condition":"NEUTRAL","key_risks":[],"key_opportunities":[],'
            '"recommended_focus":[],"confidence":0.5}'
        )


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
        graph_store=None,
        llm=None,
        enable_memory_retrieval: bool = True,
        include_burn_context: bool = False,
        churn_benchmark_pct: float | None = None,
    ):
        self.mode = mode
        self.run_id = run_id or str(uuid.uuid4())
        self.llm = llm or LLMClient()
        self.enable_memory_retrieval = enable_memory_retrieval
        # Product surfaces set this so the brief can reason about runway; the
        # research prompt stays unchanged by default (see prompt_builder).
        self.include_burn_context = include_burn_context
        # Published median monthly churn for this company's ARPA band, or None
        # when no source covers it. Never inferred - see calibration/bands.json.
        self.churn_benchmark_pct = churn_benchmark_pct
        self.state_history = deque(maxlen=5)
        self.pending_memories = deque()
        self.global_month = 0
        self.episode_global_start = 0
        self.current_episode_seed = None
        self.latest_snapshot = None
        self.latest_trend_context = TrendContext()

        self.memory_store = memory_store
        if (
            self.memory_store is None
            and self.mode in {"oracle_v3", "oracle_v4", "oracle_v4_causal"}
            and self.enable_memory_retrieval
        ):
            self.memory_store = OracleMemoryStore(run_id=self.run_id)

        self.graph_store = graph_store
        if self.graph_store is None and self.mode == "oracle_v4_causal":
            try:
                from oracle.graph_store import CausalGraphStore

                self.graph_store = CausalGraphStore()
            except Exception as exc:
                print(f"[Oracle] CausalGraphStore init failed: {exc}")
                self.graph_store = None

    def start_episode(self, episode_seed: int | None = None) -> None:
        self.current_episode_seed = episode_seed
        self.state_history.clear()
        self.pending_memories.clear()
        self.latest_snapshot = None
        self.latest_trend_context = TrendContext()
        self.episode_global_start = self.global_month

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
        active_shock_label: str | None = None,
    ) -> tuple[TrendContext, list[RetrievedMemoryCandidate], int, GraphContext]:
        """
        Returns (trend_context, memories, current_global_month, graph_context).
        Graph context is empty unless mode == oracle_v4_causal and a shock is active.
        """

        trend_context = self.latest_trend_context
        current_global_month = (
            self.latest_snapshot.global_month
            if self.latest_snapshot is not None
            else self.global_month
        )

        if (
            self.latest_snapshot is None
            or self.latest_snapshot.source_month != state.months_elapsed
        ):
            temp_snapshot = snapshot_state(
                state,
                current_global_month,
                self.current_episode_seed,
            )
            history = list(self.state_history)
            if not history or history[-1].global_month != temp_snapshot.global_month:
                history.append(temp_snapshot)
            trend_context = compute_trend_context(history)

        memories: list[RetrievedMemoryCandidate] = []
        if (
            self.mode in {"oracle_v3", "oracle_v4", "oracle_v4_causal"}
            and self.enable_memory_retrieval
            and self.memory_store is not None
        ):
            memories = self.memory_store.retrieve_similar(
                state=state,
                trend_context=trend_context,
                current_global_month=current_global_month,
                episode_global_start=self.episode_global_start,
                mrr_trend=trend_context.mrr_trend,
            )

        graph_context = GraphContext()
        if self.mode == "oracle_v4_causal" and self.graph_store is not None:
            shock_type = self._parse_shock_type(active_shock_label)
            mrr_tier = get_mrr_tier(state.mrr)
            graph_context = self.graph_store.build_graph_context(
                shock_type=shock_type,
                mrr_tier=mrr_tier,
            )

        return trend_context, memories, current_global_month, graph_context

    def get_causal_graph_context(
        self,
        state: EnvState,
    ) -> dict[str, CausalGraphContext]:
        """Return role-specific causal evidence for v4 proposal grounding."""

        if not self.mode.startswith("oracle_v4"):
            return {}
        if (
            self.graph_store is None
            or not getattr(self.graph_store, "enabled", False)
            or not hasattr(self.graph_store, "query_role_causal_context")
        ):
            return {}

        stress_node = self._identify_stress_node(state)
        contexts: dict[str, CausalGraphContext] = {}
        for role in ("CFO", "CMO", "CPO"):
            try:
                context = self.graph_store.query_role_causal_context(
                    stress_node=stress_node,
                    role=role,
                    limit=3,
                )
            except Exception as exc:
                print(f"[Oracle] Causal graph context query failed for {role}: {exc}")
                continue
            if context and context.raw_triples:
                contexts[role] = context
        return contexts

    def write_causal_outcome(
        self,
        action: dict,
        kpi_delta: dict[str, float],
        confidence: float = 0.6,
        stress_node: str | None = None,
        episode_id: int | None = None,
        month: int | None = None,
    ) -> None:
        """Persist action-to-KPI evidence when causal graph storage is active."""

        if (
            self.graph_store is None
            or not getattr(self.graph_store, "enabled", False)
            or not hasattr(self.graph_store, "write_action_outcome")
        ):
            return
        try:
            self.graph_store.write_action_outcome(
                action=action,
                kpi_delta=kpi_delta,
                confidence=confidence,
                stress_node=stress_node,
                episode_id=episode_id,
                month=month,
            )
        except Exception as exc:
            print(f"[Oracle] Causal outcome write failed: {exc}")

    def generate_brief(
        self,
        state: EnvState,
        trend_context: TrendContext | None = None,
        memories: list[RetrievedMemoryCandidate] | None = None,
        shock_label: str | None = None,
        graph_context: GraphContext | None = None,
    ) -> OracleBrief:
        if trend_context is None or memories is None:
            resolved_trend, resolved_memories, _, resolved_graph = self.get_context(
                state,
                active_shock_label=shock_label,
            )
            if trend_context is None:
                trend_context = resolved_trend
            if memories is None:
                memories = resolved_memories
            if graph_context is None:
                graph_context = resolved_graph

        prompt = build_prompt(
            state,
            mode=self.mode,
            trend_context=trend_context,
            memories=memories,
            shock_label=shock_label,
            graph_context=graph_context,
            include_burn_context=self.include_burn_context,
            churn_benchmark_pct=self.churn_benchmark_pct,
        )

        if hasattr(self.llm, "complete"):
            raw_output = self.llm.complete(
                "You are a strategic SaaS oracle. Only output perfect JSON.",
                prompt,
            )
        elif hasattr(self.llm, "generate"):
            raw_output = self.llm.generate(prompt)
        else:
            raw_output = DummyLLMClient().generate(prompt)

        if not raw_output:
            print("[WARNING] LLMClient returned empty output. Using fallback.")

        return parse_llm_response(str(raw_output))

    def build_cache_key(
        self,
        state: EnvState,
        trend_context: TrendContext | None = None,
        memories: list[RetrievedMemoryCandidate] | None = None,
    ) -> tuple[str, ...]:
        if trend_context is None:
            trend_context, _, _, _ = self.get_context(state)

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
        )

    def end_episode(self, episode_metrics: dict | None = None) -> None:
        """
        Force-mature all remaining pending memories.
        Also writes Episode node to Neo4j if graph store is active.
        """

        if self.latest_snapshot is not None:
            while self.pending_memories:
                entry = self.pending_memories.popleft()
                realized = classify_realized_outcome(
                    source_mrr=entry.snapshot.mrr,
                    current_mrr=self.latest_snapshot.mrr,
                )
                if self.memory_store is not None:
                    self.memory_store.store_memory(
                        pending_entry=entry,
                        stored_global_month=self.latest_snapshot.global_month,
                        realized_outcome=realized,
                    )

        if self.graph_store is not None and episode_metrics:
            self.graph_store.write_episode(episode_metrics)

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

    def _identify_stress_node(self, state: EnvState) -> str:
        avg_churn = (
            state.churn_enterprise + state.churn_smb + state.churn_b2c
        ) / 3.0
        runway = self._estimate_runway_months(state)
        ltv_cac = state.ltv / max(state.cac, 1.0)

        if runway < 6.0:
            return "Cash_Shortage"
        if avg_churn > 0.04:
            return "Churn_Spike"
        if ltv_cac < 3.0:
            return "CAC_Pressure"
        if state.mrr < 50_000:
            return "Growth_Stall"
        return "Steady_State"

    @staticmethod
    def _parse_shock_type(shock_label: str | None) -> str | None:
        if not shock_label or shock_label == "NO_SHOCK":
            return None
        return shock_label.split(":")[0].strip()
