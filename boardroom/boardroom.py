import math
from typing import List
from boardroom.schemas import Proposal, NegotiationState, ScoreVector
from env.schemas import EnvState


class Boardroom:
    def __init__(self, agents: List):
        self.agents = agents

    def decide(self, state: EnvState) -> dict:
        # 1. Agents generate proposals
        proposals = [agent.propose(state) for agent in self.agents]

        # 2 & 3. Evaluate proposals across 4 objectives & compute weighted scores
        weights = self._compute_weights(state)
        for p in proposals:
            p.score_vector = self._evaluate_proposal(p, state)
            p.confidence = (
                p.score_vector.efficiency * weights["efficiency"] +
                p.score_vector.growth * weights["growth"] +
                p.score_vector.innovation * weights["innovation"] +
                p.score_vector.macro * weights["macro"]
            )

        negotiation = NegotiationState(proposals=proposals, round_number=1)

        # 4. Combine proposals into unified action (Domain Extraction)
        cfo_prop = next((p for p in proposals if p.agent == "CFO"), None)
        cmo_prop = next((p for p in proposals if p.agent == "CMO"), None)
        cpo_prop = next((p for p in proposals if p.agent == "CPO"), None)
        
        # Default fallback if agents are missing for some reason
        raw_action = {
            "marketing": cmo_prop.actions.get("marketing", {"spend": 0, "channel": "ppc"}) if cmo_prop else {"spend": 0, "channel": "ppc"},
            "hiring": cfo_prop.actions.get("hiring", {"hires": 0, "cost_per_employee": 10000}) if cfo_prop else {"hires": 0, "cost_per_employee": 10000},
            "pricing": cfo_prop.actions.get("pricing", {"price_change_pct": 0.0}) if cfo_prop else {"price_change_pct": 0.0},
            "product": cpo_prop.actions.get("product", {"r_and_d_spend": 0}) if cpo_prop else {"r_and_d_spend": 0}
        }

        # Apply domain constraints
        raw_action = self._apply_sanity_bounds(raw_action, state)
        
        # 5. Apply minimum guarantees and macro nudges
        raw_action = self._apply_dynamic_minimums(raw_action, state)

        # 6. Apply conflict resolution (Constraint-based adjustment preserving intent hierarchy)
        final_action = self._resolve_conflicts(raw_action, state)

        negotiation.final_action = final_action
        negotiation.consensus_reached = True

        return final_action

    # -----------------------------
    # Evaluation & Weights
    # -----------------------------
    def _evaluate_proposal(self, proposal: Proposal, state: EnvState) -> ScoreVector:
        # Normalized metrics (0.0 to 1.0)
        
        # Efficiency (Cash runway health)
        burn = max(1.0, float(state.headcount * 10000))  # Base burn roughly headcount * 10k
        efficiency = min(1.0, max(0.0, state.cash / (burn * 12)))
        
        # Growth (LTV to CAC health)
        growth = 0.0
        if state.cac > 0:
            growth = min(1.0, max(0.0, (state.ltv / state.cac) / 5.0))
            
        # Innovation (R&D health relative to targets)
        innovation = min(1.0, max(0.0, state.innovation_factor))
        
        # Macro (Systemic recovery)
        macro = min(1.0, max(0.0, 1.0 - (state.unemployment / 30.0)))

        return ScoreVector(
            efficiency=efficiency,
            growth=growth,
            innovation=innovation,
            macro=macro
        )

    def _compute_weights(self, state: EnvState) -> dict:
        weights = {
            "efficiency": 0.40,
            "growth": 0.20,
            "innovation": 0.20,
            "macro": 0.20
        }
        
        # Gradually shift weight to innovation based on depression duration
        weights["innovation"] += (state.months_in_depression * 0.02)
        
        # Gradually shift weight to growth if unemployment is severely high
        if state.unemployment > 10.0:
            weights["growth"] += (state.unemployment - 10.0) * 0.02
        
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    # -----------------------------
    # Safeguards & Conflicts
    # -----------------------------
    def _apply_sanity_bounds(self, action: dict, state: EnvState) -> dict:
        """Domain-level sanity bounds to reign in extreme proposals."""
        # Marketing: cap spend ratio
        max_mkt = max(state.cash * 0.3, 20000)
        action["marketing"]["spend"] = min(action["marketing"].get("spend", 0), max_mkt)
        
        # Hiring: cap growth velocity
        action["hiring"]["hires"] = min(action["hiring"].get("hires", 0), 10)
        
        return action

    def _apply_dynamic_minimums(self, action: dict, state: EnvState) -> dict:
        """Dynamic Minimum Guarantees tied directly to environment state."""
        # R&D floor proportional to innovation deficit
        innovation_deficit = max(0.0, 1.0 - state.innovation_factor)
        rd_floor = 2000 + (innovation_deficit * 20000) # Scales safely
        if action["product"].get("r_and_d_spend", 0) < rd_floor:
            action["product"]["r_and_d_spend"] = rd_floor
            
        # Marketing floor proportional to MRR stagnation risks
        # ensuring we don't zero out completely and hit a hysteresis growth trap
        mkt_floor = max(5000.0, state.mrr * 0.02)
        if action["marketing"].get("spend", 0) < mkt_floor:
            action["marketing"]["spend"] = mkt_floor
            
        return action

    def _resolve_conflicts(self, action: dict, state: EnvState) -> dict:
        """Priority-aware constraint reduction. Cut order: Marketing -> Hiring -> R&D (last)."""
        mkt_spend = action["marketing"].get("spend", 0)
        rd_spend = action["product"].get("r_and_d_spend", 0)
        cost_per_employee = max(1.0, action["hiring"].get("cost_per_employee", 10000))
        hiring_spend = action["hiring"].get("hires", 0) * cost_per_employee
        
        base_burn = state.headcount * cost_per_employee
        total_needed = mkt_spend + rd_spend + hiring_spend + base_burn
        
        shortfall = total_needed - state.cash
        if shortfall <= 0:
            return action  # Approved, enough cash
            
        # 1. Cut Marketing first
        mkt_cut = min(mkt_spend, shortfall)
        action["marketing"]["spend"] -= mkt_cut
        shortfall -= mkt_cut
        
        if shortfall <= 0: return action
        
        # 2. Cut Hiring second
        hires_value = action["hiring"].get("hires", 0) * cost_per_employee
        hiring_cut_value = min(hires_value, shortfall)
        hires_to_cut = math.floor(hiring_cut_value / cost_per_employee)
        action["hiring"]["hires"] -= hires_to_cut
        shortfall -= (hires_to_cut * cost_per_employee)
        
        if shortfall <= 0: return action
        
        # 3. Cut R&D last (protect long-term)
        rd_cut = min(action["product"]["r_and_d_spend"], shortfall)
        action["product"]["r_and_d_spend"] -= rd_cut
        
        return action
