from collections import OrderedDict
from copy import deepcopy
import inspect
import math
from typing import List
from boardroom.schemas import Proposal, NegotiationState, ScoreVector
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
    ):
        self.agents = agents
        self.oracle_mode = oracle_mode or ("oracle_v1" if use_oracle else "none")
        self.use_oracle = self.oracle_mode != "none"
        self.oracle_frequency = oracle_frequency
        self.enable_action_modifier = enable_action_modifier
        self.episode_oracle_stats = OracleEpisodeStats()
        self.last_brief = None
        self.last_oracle_state = None
        self.last_refresh_snapshot = None
        self.active_shock_label = None
        self.last_context_memories = []
        self.last_decision_trace = None
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
            self.oracle_cache_max_size = oracle_cache_max_size

    def start_episode(self, episode_seed: int | None = None) -> None:
        self.episode_oracle_stats = OracleEpisodeStats()
        if not self.use_oracle:
            return
        self.oracle.start_episode(episode_seed=episode_seed)
        self.last_brief = None
        self.last_oracle_state = None
        self.last_refresh_snapshot = None
        self.active_shock_label = None
        self.last_context_memories = []
        self.last_decision_trace = None

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
        proposals = [agent.propose(state) for agent in self.agents]
        
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
                trend_context, memories, _ = self.oracle.get_context(state)
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
        
        for p in proposals:
            p.score_vector = self._evaluate_proposal(p, state)
            p.confidence = (
                p.score_vector.efficiency * weights["efficiency"] +
                p.score_vector.growth * weights["growth"] +
                p.score_vector.innovation * weights["innovation"] +
                p.score_vector.macro * weights["macro"]
            )

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
        # Sequence conflict resolutions
        final_action = self._resolve_conflicts(raw_action, state, global_innov_score)
        final_action_snapshot = deepcopy(final_action)

        negotiation.final_action = final_action
        negotiation.consensus_reached = True
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
    def _estimate_runway_months(state: EnvState) -> float:
        monthly_burn_estimate = max(1.0, float(state.headcount * 8000.0))
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

    def _generate_oracle_brief(self, state: EnvState, trend_context, memories):
        signature = inspect.signature(self.oracle.generate_brief)
        kwargs = {
            "trend_context": trend_context,
            "memories": memories,
        }
        if "shock_label" in signature.parameters:
            kwargs["shock_label"] = self.active_shock_label
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

    # -----------------------------
    # Safeguards & Conflicts
    # -----------------------------
    def _apply_sanity_bounds(self, action: dict, state: EnvState) -> dict:
        max_mkt = max(state.cash * 0.3, 20000)
        action["marketing"]["spend"] = min(action["marketing"].get("spend", 0), max_mkt)
        action["hiring"]["hires"] = min(action["hiring"].get("hires", 0), 10)
        return action

    def _apply_dynamic_minimums(self, action: dict, state: EnvState, innov_score: float) -> dict:
        innovation_deficit = max(0.0, 1.0 - state.innovation_factor)
        
        # Dynamic R&D floor: strictly tied to % of MRR + deficit scaling
        # E.g. up to 10% of MRR floor when innovation deficit is huge
        rd_floor_mrr = state.mrr * (innovation_deficit * 0.10)
        rd_floor_abs = 20000 + (innovation_deficit * 50000)
        rd_floor = max(rd_floor_mrr, rd_floor_abs)
        
        # Ensure we always hit the floor minimum
        if action["product"].get("r_and_d_spend", 0) < rd_floor:
            action["product"]["r_and_d_spend"] = rd_floor
            
        mkt_floor = max(5000.0, state.mrr * 0.02)
        if action["marketing"].get("spend", 0) < mkt_floor:
            action["marketing"]["spend"] = mkt_floor
            
        return action

    def _resolve_conflicts(self, action: dict, state: EnvState, innov_score: float) -> dict:
        mkt_spend = action["marketing"].get("spend", 0)
        rd_spend = action["product"].get("r_and_d_spend", 0)
        cost_per_employee = max(1.0, action["hiring"].get("cost_per_employee", 10000))
        hiring_spend = action["hiring"].get("hires", 0) * cost_per_employee
        
        base_burn = state.headcount * cost_per_employee
        total_needed = mkt_spend + rd_spend + hiring_spend + base_burn
        
        shortfall = total_needed - state.cash
        if shortfall <= 0:
            return action
            
        # Strong protection layer: R&D cannot be slashed entirely in a single round if innov_score is high
        rd_protection_ratio = 1.0  # Under typical scenario, can cut down to 0
        if innov_score > 0.6:
            rd_protection_ratio = 0.2  # Max allowable cut is 20% of proposed R&D spend ensuring 80% survival capability
            
        # 1. Cut Marketing
        mkt_cut = min(mkt_spend, shortfall)
        action["marketing"]["spend"] -= mkt_cut
        shortfall -= mkt_cut
        
        if shortfall <= 0: return action
        
        # 2. Cut Hiring
        hires_value = action["hiring"].get("hires", 0) * cost_per_employee
        hiring_cut_value = min(hires_value, shortfall)
        hires_to_cut = math.floor(hiring_cut_value / cost_per_employee)
        action["hiring"]["hires"] -= hires_to_cut
        shortfall -= (hires_to_cut * cost_per_employee)
        
        if shortfall <= 0: return action
        
        # 3. Cut R&D (last priority)
        max_allowed_rd_cut = action["product"]["r_and_d_spend"] * rd_protection_ratio
        rd_cut = min(max_allowed_rd_cut, shortfall)
        action["product"]["r_and_d_spend"] -= rd_cut
        
        return action
