import math
from typing import List
from boardroom.schemas import Proposal, NegotiationState, ScoreVector
from env.schemas import EnvState
from oracle.oracle import Oracle
from oracle.weight_adapter import WeightAdapter


class Boardroom:
    def __init__(self, agents: List, use_oracle: bool = False, oracle_frequency: int = 3):
        self.agents = agents
        self.use_oracle = use_oracle
        self.oracle_frequency = oracle_frequency
        if self.use_oracle:
            self.oracle = Oracle()
            self.weight_adapter = WeightAdapter()
            self.last_brief = None
            self.last_oracle_state = None

    def decide(self, state: EnvState) -> dict:
        proposals = [agent.propose(state) for agent in self.agents]
        
        base_weights = self._compute_weights(state)
        
        if self.use_oracle:
            if (state.months_elapsed % self.oracle_frequency == 0) or (self.last_brief is None):
                print(f"[Boardroom Oracle] Triggering LLM reasoning at Month {state.months_elapsed}...")
                self.last_brief = self.oracle.generate_brief(state)
                
            weights = self.weight_adapter.adjust_weights(base_weights, self.last_brief)
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

        # Apply structural sanity checks
        raw_action = self._apply_sanity_bounds(raw_action, state)
        # Apply strict minimum guarantees
        raw_action = self._apply_dynamic_minimums(raw_action, state, global_innov_score)
        # Sequence conflict resolutions
        final_action = self._resolve_conflicts(raw_action, state, global_innov_score)

        negotiation.final_action = final_action
        negotiation.consensus_reached = True

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
