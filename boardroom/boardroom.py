from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import inspect
import math
from typing import List
from boardroom.schemas import Proposal, NegotiationState, ScoreVector
from env import business_logic
from env.schemas import EnvState
from oracle.action_modifier import ActionModifier
from oracle.oracle import Oracle
from oracle.schemas import OracleEpisodeStats, OracleRefreshSnapshot
from oracle.weight_adapter import WeightAdapter


class Boardroom:
    def __init__(
        self,
        agents: List,
        use_oracle: bool = False,
        oracle_frequency: int = 3,
        oracle_mode: str | None = None,
        oracle_instance: Oracle | None = None,
        action_modifier_instance=None,
        oracle_cache_max_size: int = 5000,
        enable_action_modifier: bool = True,
        enable_memory_retrieval: bool = True,
        proposal_generator=None,
        scale_absolutes: float = 1.0,
        hiring_runway_guard_months: float | None = None,
    ):
        # Spec G11. The absolute spend floors below are calibrated for a ~$50k
        # MRR company; applied unscaled to a $12k-MRR founder they demand more
        # than the company earns. Product surfaces pass mrr/50k here; research
        # runs leave it at 1.0, which reproduces the original constants exactly.
        self.scale_absolutes = max(0.01, float(scale_absolutes))
        # CFOAgent refuses to hire under 24 months of runway, but that guard
        # lives in the proposal and an LLM proposal generator can simply not
        # apply it. Set this (product surfaces do) to re-assert the rule on the
        # final action, after every proposal and modifier has had its say.
        # None keeps research behaviour untouched.
        self.hiring_runway_guard_months = hiring_runway_guard_months
        self.agents = agents
        self.proposal_generator = proposal_generator
        self.oracle_mode = oracle_mode or ("oracle_v1" if use_oracle else "none")
        self.use_oracle = self.oracle_mode != "none"
        self.oracle_frequency = oracle_frequency
        self.oracle_cache_max_size = oracle_cache_max_size
        self.enable_action_modifier = enable_action_modifier
        self.episode_oracle_stats = OracleEpisodeStats()
        self.last_brief = None
        self.last_oracle_state = None
        self.last_refresh_snapshot = None
        self.active_shock_label = None
        self.last_context_memories = []
        self.last_decision_trace = None
        self.last_proposals: list[Proposal] = []
        self.last_causal_contexts = {}
        self.last_causal_stress_node = None
        self.last_stress_node_for_persistence = None
        self.consecutive_stress_node_months = 0
        self.last_final_action_snapshot = None
        self._pending_outcome_writes: list[dict] = []
        self.proposal_cache = OrderedDict()
        if self.use_oracle:
            self.oracle = oracle_instance or Oracle(
                mode=self.oracle_mode,
                enable_memory_retrieval=enable_memory_retrieval,
            )
            if hasattr(self.oracle, "enable_memory_retrieval"):
                self.oracle.enable_memory_retrieval = enable_memory_retrieval
            self.action_modifier = action_modifier_instance or ActionModifier()
            self.weight_adapter = WeightAdapter()
            self.oracle_cache = OrderedDict()

    def start_episode(self, episode_seed: int | None = None) -> None:
        self.episode_oracle_stats = OracleEpisodeStats()
        self.last_brief = None
        self.last_oracle_state = None
        self.last_refresh_snapshot = None
        self.active_shock_label = None
        self.last_context_memories = []
        self.last_decision_trace = None
        self.last_proposals = []
        self.last_causal_contexts = {}
        self.last_causal_stress_node = None
        self.last_stress_node_for_persistence = None
        self.consecutive_stress_node_months = 0
        self.last_final_action_snapshot = None
        self._pending_outcome_writes = []
        if not self.use_oracle:
            return
        self.oracle.start_episode(episode_seed=episode_seed)

    def get_episode_stats(self) -> dict:
        return self.episode_oracle_stats.model_dump()

    def set_shock_label(self, shock_label: str | None) -> None:
        if not self.use_oracle:
            return
        self.active_shock_label = shock_label

    def get_last_brief(self):
        return getattr(self, "last_brief", None)

    def get_last_decision_trace(self):
        return self.last_decision_trace

    def decide(self, state: EnvState) -> dict:
        use_batched_proposals = self.proposal_generator is not None
        proposals = [] if use_batched_proposals else self._collect_proposals(state)
        proposal_source = "agents"
        proposal_error = None
        causal_contexts = dict(self.last_causal_contexts)
        causal_stress_node = self.last_causal_stress_node
        previous_final_action_snapshot = deepcopy(self.last_final_action_snapshot)
        
        base_weights = self._compute_weights(state)
        refresh_reason = None
        brief_source = "none"
        cache_key = None
        context_memories = list(self.last_context_memories)
        
        if self.use_oracle:
            self.oracle.observe_state(state)
            refresh_reason = self._get_oracle_refresh_reason(state)
            if refresh_reason is None and self.last_brief is None:
                refresh_reason = "initial"

            if refresh_reason is not None:
                trend_context, memories, _, graph_context = self.oracle.get_context(
                    state,
                    active_shock_label=self.active_shock_label,
                )
                self._record_refresh_request(refresh_reason)
                cache_key = self.oracle.build_cache_key(
                    state,
                    trend_context=trend_context,
                    memories=memories,
                )
                self.last_context_memories = list(memories)
                context_memories = list(memories)
                cached_brief = self._get_cached_brief(cache_key)
                if cached_brief is not None:
                    print(f"[Boardroom Oracle] Reusing cached brief at Month {state.months_elapsed} ({refresh_reason})...")
                    self.last_brief = cached_brief
                    self.episode_oracle_stats.cache_hits += 1
                    brief_source = "cache_hit"
                else:
                    print(f"[Boardroom Oracle] Triggering LLM reasoning at Month {state.months_elapsed} ({refresh_reason})...")
                    self.last_brief = self._generate_oracle_brief(
                        state,
                        trend_context=trend_context,
                        memories=memories,
                        graph_context=graph_context,
                    )
                    self.episode_oracle_stats.llm_calls += 1
                    self._cache_brief(cache_key, self.last_brief)
                    brief_source = "llm"

                self.last_refresh_snapshot = self._make_refresh_snapshot(state)
                self.last_oracle_state = self.last_refresh_snapshot
            elif self.last_brief is not None:
                brief_source = "reuse"
                
            weights = self.weight_adapter.adjust_weights(base_weights, self.last_brief, oracle_mode=self.oracle_mode)
        else:
            weights = base_weights

        if use_batched_proposals:
            (
                proposals,
                proposal_source,
                proposal_error,
                causal_contexts,
                causal_stress_node,
            ) = self._resolve_batched_proposals(
                state=state,
                refresh_reason=refresh_reason,
                cache_key=cache_key,
            )
        
        proposal_base_scores = {}
        for p in proposals:
            p.score_vector = self._evaluate_proposal(p, state)
            base_score = (
                p.score_vector.efficiency * weights["efficiency"] +
                p.score_vector.growth * weights["growth"] +
                p.score_vector.innovation * weights["innovation"] +
                p.score_vector.macro * weights["macro"]
            )
            proposal_base_scores[p.agent] = base_score
            p.confidence = self._apply_causal_confidence_score(base_score, p)

        negotiation = NegotiationState(proposals=proposals, round_number=1)

        cfo_prop = next((p for p in proposals if p.agent == "CFO"), None)
        cmo_prop = next((p for p in proposals if p.agent == "CMO"), None)
        cpo_prop = next((p for p in proposals if p.agent == "CPO"), None)
        
        # Grab the global systemic innovation urgency
        global_innov_score = proposals[0].score_vector.innovation if proposals else 0.0

        base_rd = cpo_prop.actions.get("product", {}).get("r_and_d_spend", 0) if cpo_prop else 0
        
        # 1. SCALE R&D aggressively based on system size and innovation deficit
        innovation_deficit = max(0.0, 1.0 - state.innovation_factor)
        if state.innovation_factor < 0.3:
            # Nonlinear escalation: Massive push when innovation is failing
            aggressive_rd = state.mrr * innovation_deficit * 0.15 # Up to 15% of MRR
            scaled_rd_spend = max(base_rd, aggressive_rd)
        else:
            scaled_rd_spend = base_rd * (1.0 + (global_innov_score * 2.0))
        
        raw_action = {
            "marketing": cmo_prop.actions.get("marketing", {"spend": 0, "channel": "ppc"}) if cmo_prop else {"spend": 0, "channel": "ppc"},
            "hiring": cfo_prop.actions.get("hiring", {"hires": 0, "cost_per_employee": 10000}) if cfo_prop else {"hires": 0, "cost_per_employee": 10000},
            "pricing": cfo_prop.actions.get("pricing", {"price_change_pct": 0.0}) if cfo_prop else {"price_change_pct": 0.0},
            "product": {"r_and_d_spend": scaled_rd_spend}
        }

        pre_modifier_action = deepcopy(raw_action)
        modifier_applied = self.use_oracle and self.enable_action_modifier and self.last_brief is not None
        if modifier_applied:
            raw_action = self.action_modifier.modify(raw_action, self.last_brief)
        post_modifier_action = deepcopy(raw_action)

        # Apply structural sanity checks
        raw_action = self._apply_sanity_bounds(raw_action, state)
        # Apply strict minimum guarantees
        raw_action = self._apply_dynamic_minimums(raw_action, state, global_innov_score)
        # Keep v4 causal R&D floors from re-inflating beyond cash-aware safety caps.
        raw_action = self._apply_v4_causal_rd_cap(raw_action, state)
        # Sequence conflict resolutions
        final_action = self._resolve_conflicts(
            raw_action,
            state,
            global_innov_score,
            stress_node=causal_stress_node,
        )
        final_action_snapshot = deepcopy(final_action)
        self.last_final_action_snapshot = deepcopy(final_action_snapshot)

        negotiation.final_action = final_action
        negotiation.consensus_reached = True
        if (
            self.use_oracle
            and self.oracle_mode == "oracle_v4_causal"
            and hasattr(self.oracle, "graph_store")
            and self.oracle.graph_store is not None
            and self.oracle.graph_store.enabled
        ):
            current_month = state.months_elapsed

            still_pending = []
            for pending in self._pending_outcome_writes:
                if current_month - pending["shock_month"] >= 6:
                    mrr_at_shock = pending["mrr_at_shock"]
                    mrr_change_pct = (state.mrr - mrr_at_shock) / max(abs(mrr_at_shock), 1.0)
                    self.oracle.graph_store.write_outcome(
                        episode_id=pending["episode_id"],
                        shock_month=pending["shock_month"],
                        outcome_metrics={
                            "mrr_change_pct": mrr_change_pct,
                            "recovered": state.mrr >= mrr_at_shock,
                            "recovery_months": current_month - pending["shock_month"],
                            "post_shock_rule40": 0.0,
                        },
                    )
                else:
                    still_pending.append(pending)
            self._pending_outcome_writes = still_pending

            if self.active_shock_label and self.active_shock_label != "NO_SHOCK":
                if self.oracle.latest_snapshot is not None:
                    self.oracle.graph_store.write_shock_event(
                        episode_id=self.oracle.current_episode_seed or 0,
                        shock_label=self.active_shock_label,
                        month=current_month,
                        pre_state=self.oracle.latest_snapshot,
                        decision=final_action,
                        brief=self.last_brief,
                    )
                    self._pending_outcome_writes.append(
                        {
                            "episode_id": self.oracle.current_episode_seed or 0,
                            "shock_month": current_month,
                            "mrr_at_shock": state.mrr,
                        }
                    )
        self.last_decision_trace = {
            "month": state.months_elapsed,
            "oracle_mode": self.oracle_mode,
            "used_oracle": self.use_oracle,
            "refresh_reason": refresh_reason,
            "brief_source": brief_source,
            "cache_key": list(cache_key) if cache_key is not None else None,
            "shock_label": self.active_shock_label,
            "base_weights": deepcopy(base_weights),
            "applied_weights": deepcopy(weights),
            "brief": self._brief_to_dict(self.last_brief),
            "memory_count": len(context_memories),
            "retrieved_memories": self._serialize_memories(context_memories),
            "proposal_source": proposal_source,
            "proposal_error": proposal_error,
            "causal_stress_node": causal_stress_node,
            "stress_persistence_months": self.consecutive_stress_node_months,
            "previous_final_action": previous_final_action_snapshot,
            "causal_contexts": self._serialize_causal_contexts(causal_contexts),
            "proposals": self._serialize_proposals(proposals, proposal_base_scores),
            "pre_modifier_action": pre_modifier_action,
            "post_modifier_action": post_modifier_action,
            "final_action": final_action_snapshot,
            "action_modifier_applied": modifier_applied,
            "marketing_spend_change_pct": self._pct_change(
                pre_modifier_action.get("marketing", {}).get("spend", 0.0),
                post_modifier_action.get("marketing", {}).get("spend", 0.0),
            ),
            "rd_spend_change_pct": self._pct_change(
                pre_modifier_action.get("product", {}).get("r_and_d_spend", 0.0),
                post_modifier_action.get("product", {}).get("r_and_d_spend", 0.0),
            ),
            "hires_change": post_modifier_action.get("hiring", {}).get("hires", 0) - pre_modifier_action.get("hiring", {}).get("hires", 0),
        }

        return final_action

    def _collect_proposals(self, state: EnvState) -> list[Proposal]:
        should_parallelize = (
            len(self.agents) > 1
            and any(getattr(agent, "use_llm", False) for agent in self.agents)
        )
        if not should_parallelize:
            return [agent.propose(state) for agent in self.agents]

        with ThreadPoolExecutor(max_workers=min(3, len(self.agents))) as pool:
            return list(pool.map(lambda agent: agent.propose(state), self.agents))

    def _resolve_batched_proposals(
        self,
        state: EnvState,
        refresh_reason: str | None,
        cache_key: tuple[str, ...] | None,
    ) -> tuple[list[Proposal], str, str | None, dict, str | None]:
        if not self.use_oracle:
            proposals = self._collect_proposals(state)
            self.last_proposals = deepcopy(proposals)
            return proposals, "agents", None, {}, None

        if refresh_reason is not None:
            if cache_key is not None:
                cached = self._get_cached_proposals(cache_key)
                if cached is not None:
                    proposals, contexts, stress_node = cached
                    stress_node = self._extract_stress_node(state, contexts) or stress_node
                    self._update_stress_persistence(stress_node)
                    self.last_proposals = deepcopy(proposals)
                    self.last_causal_contexts = dict(contexts)
                    self.last_causal_stress_node = stress_node
                    self.episode_oracle_stats.proposal_cache_hits += 1
                    return proposals, "cache_hit", None, contexts, stress_node

            contexts = self._get_causal_contexts_for_proposals(state)
            stress_node = self._extract_stress_node(state, contexts)
            stress_persistence_months = self._update_stress_persistence(stress_node)
            before_calls = getattr(self.proposal_generator, "llm_calls", 0)
            proposals = self._call_proposal_generator(
                state,
                contexts,
                stress_persistence_months=stress_persistence_months,
                recent_action_pattern=self.last_final_action_snapshot,
            )
            after_calls = getattr(self.proposal_generator, "llm_calls", before_calls)
            self.episode_oracle_stats.proposal_llm_calls += max(0, after_calls - before_calls)

            source = getattr(self.proposal_generator, "last_source", "llm")
            error = getattr(self.proposal_generator, "last_error", None)
            if source.startswith("fallback"):
                self.episode_oracle_stats.proposal_fallbacks += 1

            self.last_proposals = deepcopy(proposals)
            self.last_causal_contexts = dict(contexts)
            self.last_causal_stress_node = stress_node
            if cache_key is not None and source == "llm":
                self._cache_proposals(cache_key, proposals, contexts, stress_node)
            return proposals, source, error, contexts, stress_node

        if self.last_proposals:
            stress_node = self._extract_stress_node(state, {}) or self.last_causal_stress_node
            self._update_stress_persistence(stress_node)
            self.last_causal_stress_node = stress_node
            return (
                deepcopy(self.last_proposals),
                "reuse",
                None,
                dict(self.last_causal_contexts),
                stress_node,
            )

        proposals = self._collect_proposals(state)
        stress_node = self._extract_stress_node(state, {})
        self._update_stress_persistence(stress_node)
        self.last_causal_stress_node = stress_node
        self.last_proposals = deepcopy(proposals)
        return proposals, "fallback_no_cached_proposals", None, {}, stress_node

    def _call_proposal_generator(
        self,
        state: EnvState,
        contexts: dict,
        stress_persistence_months: int,
        recent_action_pattern: dict | None,
    ) -> list[Proposal]:
        signature = inspect.signature(self.proposal_generator.propose_all)
        kwargs = {}
        if "stress_persistence_months" in signature.parameters:
            kwargs["stress_persistence_months"] = stress_persistence_months
        if "recent_action_pattern" in signature.parameters:
            kwargs["recent_action_pattern"] = deepcopy(recent_action_pattern)
        return self.proposal_generator.propose_all(state, contexts, **kwargs)

    def _update_stress_persistence(self, stress_node: str | None) -> int:
        if not stress_node:
            self.last_stress_node_for_persistence = None
            self.consecutive_stress_node_months = 0
            return self.consecutive_stress_node_months

        if stress_node == self.last_stress_node_for_persistence:
            self.consecutive_stress_node_months += 1
        else:
            self.last_stress_node_for_persistence = stress_node
            self.consecutive_stress_node_months = 1
        return self.consecutive_stress_node_months

    def _get_causal_contexts_for_proposals(self, state: EnvState) -> dict:
        if not hasattr(self.oracle, "get_causal_graph_context"):
            return {}
        return self.oracle.get_causal_graph_context(state)

    def _extract_stress_node(self, state: EnvState, contexts: dict) -> str | None:
        for context in contexts.values():
            stress_node = getattr(context, "stress_node", None)
            if stress_node:
                return stress_node
        if hasattr(self.oracle, "_identify_stress_node"):
            return self.oracle._identify_stress_node(state)
        return None

    def _get_cached_proposals(self, cache_key: tuple[str, ...]):
        entry = self.proposal_cache.get(cache_key)
        if entry is None:
            return None
        return (
            deepcopy(entry["proposals"]),
            deepcopy(entry.get("contexts", {})),
            entry.get("stress_node"),
        )

    def _cache_proposals(
        self,
        cache_key: tuple[str, ...],
        proposals: list[Proposal],
        contexts: dict,
        stress_node: str | None,
    ) -> None:
        self.proposal_cache[cache_key] = {
            "proposals": deepcopy(proposals),
            "contexts": deepcopy(contexts),
            "stress_node": stress_node,
        }
        while len(self.proposal_cache) > self.oracle_cache_max_size:
            self.proposal_cache.popitem(last=False)

    def _apply_causal_confidence_score(self, base_score: float, proposal: Proposal) -> float:
        final_score = base_score
        if (
            self.oracle_mode.startswith("oracle_v4")
            and proposal.causal_confidence is not None
        ):
            causal_boost = 0.85 + (proposal.causal_confidence * 0.30)
            final_score = base_score * causal_boost
        return max(0.0, min(1.0, final_score))

    # -----------------------------
    # Evaluation & Weights
    # -----------------------------
    def _evaluate_proposal(self, proposal: Proposal, state: EnvState) -> ScoreVector:
        # Efficiency
        burn = max(1.0, float(state.headcount * 10000))
        efficiency = min(1.0, max(0.0, state.cash / (burn * 12)))
        
        # Growth
        growth = 0.0
        if state.cac > 0:
            growth = min(1.0, max(0.0, (state.ltv / state.cac) / 5.0))
            
        # Innovation (Precision Formula)
        innovation_deficit = max(0.0, 1.0 - state.innovation_factor)
        avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0
        churn_pressure = min(1.0, max(0.0, avg_churn / 0.10))
        depression_pressure = min(1.0, state.months_in_depression / 12.0)
        
        innovation = (0.5 * innovation_deficit) + (0.3 * churn_pressure) + (0.2 * depression_pressure)
        
        # Macro
        macro = min(1.0, max(0.0, 1.0 - (state.unemployment / 30.0)))

        return ScoreVector(
            efficiency=efficiency,
            growth=growth,
            innovation=innovation,
            macro=macro
        )

    def _compute_weights(self, state: EnvState) -> dict:
        weights = {
            "efficiency": 0.30,
            "growth": 0.20,
            "innovation": 0.40,  # Base weight vastly increased to force strategic shifts
            "macro": 0.10
        }
        
        weights["innovation"] += (state.months_in_depression * 0.02)
        if state.unemployment > 10.0:
            weights["growth"] += (state.unemployment - 10.0) * 0.02
        
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _get_oracle_refresh_reason(self, state: EnvState) -> str | None:
        if self.last_brief is None:
            return "initial"

        if self.oracle_frequency > 0 and state.months_elapsed > 0 and (state.months_elapsed % self.oracle_frequency == 0):
            return "cadence"

        if self._has_event_trigger(state):
            return "event"

        return None

    def _has_event_trigger(self, state: EnvState) -> bool:
        current_runway = self._estimate_runway_months(state)
        if current_runway < 12.0:
            return True

        if self.active_shock_label and self.active_shock_label != "NO_SHOCK":
            return True

        if self.last_refresh_snapshot is None:
            return False

        current_avg_churn = self._average_churn(state)
        if state.mrr <= (self.last_refresh_snapshot.mrr * 0.85):
            return True
        if (current_avg_churn - self.last_refresh_snapshot.avg_churn) >= 0.015:
            return True
        if (self.last_refresh_snapshot.consumer_confidence - state.consumer_confidence) >= 15.0:
            return True
        if (state.unemployment - self.last_refresh_snapshot.unemployment) >= 2.0:
            return True

        return False

    @staticmethod
    def _average_churn(state: EnvState) -> float:
        return (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0

    @staticmethod
    def _net_runway_months(state: EnvState) -> float:
        """Runway against NET burn, so revenue counts.

        _estimate_runway_months divides cash by gross salary burn and ignores
        MRR entirely: a company earning $200k against $220k of costs reads as
        8.9 months there when it actually has 100. That understatement is
        tolerable for a refresh trigger but not for a guard that blocks hiring,
        so the guard uses this instead. Left separate rather than fixing the
        original, which research runs depend on.
        """
        burn = max(1.0, business_logic.monthly_burn(state))
        net_burn = burn - state.mrr
        if net_burn <= 0:
            return float("inf")
        return state.cash / net_burn

    @staticmethod
    def _estimate_runway_months(state: EnvState) -> float:
        monthly_burn_estimate = max(1.0, business_logic.monthly_burn(state))
        return state.cash / monthly_burn_estimate

    def _make_refresh_snapshot(self, state: EnvState) -> OracleRefreshSnapshot:
        return OracleRefreshSnapshot(
            months_elapsed=state.months_elapsed,
            mrr=state.mrr,
            avg_churn=self._average_churn(state),
            consumer_confidence=state.consumer_confidence,
            unemployment=state.unemployment,
            runway_months=self._estimate_runway_months(state),
        )

    def _record_refresh_request(self, refresh_reason: str) -> None:
        self.episode_oracle_stats.oracle_refresh_requests += 1
        if refresh_reason == "cadence":
            self.episode_oracle_stats.cadence_refreshes += 1
        elif refresh_reason == "event":
            self.episode_oracle_stats.event_refreshes += 1

    def _generate_oracle_brief(self, state: EnvState, trend_context, memories, graph_context=None):
        signature = inspect.signature(self.oracle.generate_brief)
        kwargs = {
            "trend_context": trend_context,
            "memories": memories,
        }
        if "shock_label" in signature.parameters:
            kwargs["shock_label"] = self.active_shock_label
        if "graph_context" in signature.parameters and graph_context is not None:
            kwargs["graph_context"] = graph_context
        return self.oracle.generate_brief(state, **kwargs)

    def _get_cached_brief(self, cache_key: tuple[str, ...]):
        return self.oracle_cache.get(cache_key)

    def _cache_brief(self, cache_key: tuple[str, ...], brief) -> None:
        if cache_key in self.oracle_cache:
            self.oracle_cache[cache_key] = brief
            return

        self.oracle_cache[cache_key] = brief
        while len(self.oracle_cache) > self.oracle_cache_max_size:
            self.oracle_cache.popitem(last=False)

    @staticmethod
    def _pct_change(before: float, after: float) -> float:
        baseline = max(abs(before), 1.0)
        return ((after - before) / baseline) * 100.0

    @staticmethod
    def _brief_to_dict(brief):
        if brief is None:
            return None
        if hasattr(brief, "model_dump"):
            return brief.model_dump(mode="json")
        if hasattr(brief, "dict"):
            return brief.dict()
        return {
            "risk_level": getattr(brief, "risk_level", None),
            "growth_outlook": getattr(brief, "growth_outlook", None),
            "efficiency_pressure": getattr(brief, "efficiency_pressure", None),
            "innovation_urgency": getattr(brief, "innovation_urgency", None),
            "macro_condition": getattr(brief, "macro_condition", None),
            "confidence": getattr(brief, "confidence", None),
        }

    @staticmethod
    def _serialize_memories(memories) -> list[dict]:
        serialized = []
        for memory in memories or []:
            if hasattr(memory, "model_dump"):
                serialized.append(memory.model_dump(mode="json"))
            else:
                serialized.append(
                    {
                        "document": getattr(memory, "document", None),
                        "metadata": getattr(memory, "metadata", {}),
                        "distance": getattr(memory, "distance", None),
                        "similarity_score": getattr(memory, "similarity_score", None),
                        "recency_factor": getattr(memory, "recency_factor", None),
                        "memory_weight": getattr(memory, "memory_weight", None),
                    }
                )
        return serialized

    @staticmethod
    def _serialize_causal_contexts(contexts) -> dict:
        serialized = {}
        for role, context in (contexts or {}).items():
            if hasattr(context, "model_dump"):
                serialized[role] = context.model_dump(mode="json")
            else:
                serialized[role] = {
                    "role": getattr(context, "role", role),
                    "stress_node": getattr(context, "stress_node", None),
                    "chain_summary": getattr(context, "chain_summary", None),
                    "root_cause_node": getattr(context, "root_cause_node", None),
                    "confidence": getattr(context, "confidence", None),
                    "raw_triples": getattr(context, "raw_triples", []),
                }
        return serialized

    @staticmethod
    def _serialize_proposals(
        proposals: list[Proposal],
        proposal_base_scores: dict[str, float],
    ) -> list[dict]:
        serialized = []
        for proposal in proposals:
            score_vector = proposal.score_vector
            if hasattr(score_vector, "model_dump"):
                score_vector = score_vector.model_dump(mode="json")
            serialized.append(
                {
                    "agent": proposal.agent,
                    "objective": proposal.objective,
                    "actions": deepcopy(proposal.actions),
                    "expected_impact": proposal.expected_impact,
                    "risks": list(proposal.risks),
                    "rationale": proposal.rationale,
                    "causal_confidence": proposal.causal_confidence,
                    "base_score": proposal_base_scores.get(proposal.agent),
                    "final_confidence": proposal.confidence,
                    "score_vector": score_vector,
                }
            )
        return serialized

    # -----------------------------
    # Safeguards & Conflicts
    # -----------------------------
    def _apply_sanity_bounds(self, action: dict, state: EnvState) -> dict:
        max_mkt = max(state.cash * 0.3, 20000 * self.scale_absolutes)
        action["marketing"]["spend"] = min(action["marketing"].get("spend", 0), max_mkt)
        action["hiring"]["hires"] = min(action["hiring"].get("hires", 0), 10)
        if (
            self.hiring_runway_guard_months is not None
            and action["hiring"].get("hires", 0) > 0
            and self._net_runway_months(state) < self.hiring_runway_guard_months
        ):
            action["hiring"]["hires"] = 0
        action = self._apply_v4_causal_rd_cap(action, state)
        return action

    def _apply_v4_causal_rd_cap(self, action: dict, state: EnvState) -> dict:
        if self.oracle_mode != "oracle_v4_causal":
            return action
        action.setdefault("product", {})
        rd_spend = max(0.0, float(action["product"].get("r_and_d_spend", 0.0)))
        rd_cap = max(float(state.cash) * 0.25, 30_000.0 * self.scale_absolutes)
        action["product"]["r_and_d_spend"] = min(rd_spend, rd_cap)
        return action

    def _apply_dynamic_minimums(self, action: dict, state: EnvState, innov_score: float) -> dict:
        innovation_deficit = max(0.0, 1.0 - state.innovation_factor)
        
        # Dynamic R&D floor: strictly tied to % of MRR + deficit scaling
        # E.g. up to 10% of MRR floor when innovation deficit is huge
        rd_floor_mrr = state.mrr * (innovation_deficit * 0.10)
        rd_floor_abs = (20000 + (innovation_deficit * 50000)) * self.scale_absolutes
        rd_floor = max(rd_floor_mrr, rd_floor_abs)
        
        # Ensure we always hit the floor minimum
        if action["product"].get("r_and_d_spend", 0) < rd_floor:
            action["product"]["r_and_d_spend"] = rd_floor
            
        mkt_floor = max(5000.0 * self.scale_absolutes, state.mrr * 0.02)
        if action["marketing"].get("spend", 0) < mkt_floor:
            action["marketing"]["spend"] = mkt_floor
            
        return action

    def _resolve_conflicts(
        self,
        action: dict,
        state: EnvState,
        innov_score: float,
        stress_node: str | None = None,
    ) -> dict:
        mkt_spend = action["marketing"].get("spend", 0)
        rd_spend = action["product"].get("r_and_d_spend", 0)
        cost_per_employee = max(1.0, action["hiring"].get("cost_per_employee", 10000))
        hiring_spend = action["hiring"].get("hires", 0) * cost_per_employee
        
        # The eighth place the engine reimplemented "burn = headcount x a
        # per-head constant", and the one a grep for `headcount * 8000` misses
        # because the constant is a variable here - and the wrong variable.
        # cost_per_employee is the ONE-TIME recruiting cost of a new hire
        # ($10,000 by default, and used as such three lines above in
        # hiring_spend); using it again as a monthly salary charged a
        # founder-only company $10,000 a month.
        #
        # Measured on a founder at $2,500 MRR, $5,000 cash and $500/month of
        # real costs: total_needed came to $21,100 against $5,000, a $16,100
        # shortfall that does not exist, and the resolver zeroed the entire
        # plan to cover it - $500 of marketing, $600 of R&D and the hire. The
        # founder was shown "hold / wait / hold" and told nothing.
        #
        # Research runs have no monthly_burn and keep the original expression
        # exactly, because their recorded trajectories depend on it.
        base_burn = (
            business_logic.monthly_burn(state)
            if state.monthly_burn is not None
            else state.headcount * cost_per_employee
        )
        total_needed = mkt_spend + rd_spend + hiring_spend + base_burn
        
        shortfall = total_needed - state.cash
        if shortfall <= 0:
            return action
            
        # Strong protection layer: R&D cannot be slashed entirely in a single round if innov_score is high
        rd_protection_ratio = 1.0  # Under typical scenario, can cut down to 0
        if innov_score > 0.6:
            rd_protection_ratio = 0.2  # Max allowable cut is 20% of proposed R&D spend ensuring 80% survival capability
        if self.oracle_mode == "oracle_v4_causal" and stress_node == "Cash_Shortage":
            rd_protection_ratio = 1.0
            
        # 1. Cut Marketing
        mkt_cut = min(mkt_spend, shortfall)
        action["marketing"]["spend"] -= mkt_cut
        shortfall -= mkt_cut
        
        if shortfall <= 0: return action
        
        # 2. Cut Hiring
        hires_value = action["hiring"].get("hires", 0) * cost_per_employee
        hiring_cut_value = min(hires_value, shortfall)
        # A hire is lumpy, and rounding the cut DOWN means a headcount whose
        # cost exceeds the remaining shortfall can never be cut at all:
        # floor($6,100 / $10,000) is zero, so the plan sheds every dollar of
        # marketing and R&D to protect a hire it still cannot afford. Rounding
        # up removes the hire instead, which is the honest reading of "you do
        # not have the money for this person".
        #
        # Only on the real-burn path. Research runs keep the original rounding,
        # because their recorded trajectories were produced by it.
        if state.monthly_burn is not None:
            hires_to_cut = min(
                action["hiring"].get("hires", 0),
                math.ceil(hiring_cut_value / cost_per_employee),
            )
        else:
            hires_to_cut = math.floor(hiring_cut_value / cost_per_employee)
        action["hiring"]["hires"] -= hires_to_cut
        shortfall -= (hires_to_cut * cost_per_employee)
        
        if shortfall <= 0:
            # Cutting a whole hire usually overshoots, and the overshoot was
            # paid for by the marketing budget that was cut first. Give it back
            # rather than banking a saving nobody asked for: the founder above
            # was told to spend nothing on a month where $500 of marketing and
            # $600 of R&D were comfortably affordable.
            if state.monthly_burn is not None and shortfall < 0:
                action["marketing"]["spend"] += min(mkt_cut, -shortfall)
            return action
        
        # 3. Cut R&D (last priority)
        max_allowed_rd_cut = action["product"]["r_and_d_spend"] * rd_protection_ratio
        rd_cut = min(max_allowed_rd_cut, shortfall)
        action["product"]["r_and_d_spend"] -= rd_cut
        
        return action
