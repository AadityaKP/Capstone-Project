import random
import math
from env.schemas import EnvState, MarketingAction, ProductAction, PricingAction, HiringAction

def interest_rate_shock(state: EnvState, prob: float = 0.1) -> None:
    if random.random() < prob:
        state.interest_rate += 1.5  
        state.valuation_multiple *= 0.85
        state.churn_smb *= 1.2

def consumer_confidence_shock(state: EnvState, prob: float = 0.1) -> None:
    if random.random() < prob:
        state.consumer_confidence -= 20
        state.unemployment += 1.0

def competitive_entry_shock(state: EnvState, prob: float = 0.1) -> None:
    market_attractiveness = (state.mrr - 50_000) / 50_000 
    dynamic_prob = 1 / (1 + math.exp(-market_attractiveness)) 
    actual_prob = prob * (2 * dynamic_prob)

    if random.random() < actual_prob:
        state.competitors += 1
        state.price *= 0.9

def apply_recession_cascade(state: EnvState) -> None:
    """
    Credit-Bankruptcy Loop.
    If Unemployment High + Rates High -> Confidence Crash.
    """
    if state.unemployment > 8.0 and state.interest_rate > 7.0:
        if random.random() < 0.2: 
            state.consumer_confidence -= 10
            state.valuation_multiple *= 0.8
            state.unemployment += 0.5 

def apply_hysteresis(state: EnvState) -> None:
    """
    Growth Hysteresis.
    Long depressions permanently scar innovation.
    """
    if state.consumer_confidence < 50:
        state.months_in_depression += 1
    else:
        state.months_in_depression = max(0, state.months_in_depression - 1)

    if state.months_in_depression >= 6:
        state.innovation_factor *= 0.95

def apply_recovery(state: EnvState) -> None:
    """
    Mean-reversion mechanics.
    """
    if state.innovation_factor < 1.0:
        state.innovation_factor += 0.001 

    if state.valuation_multiple < 10.0:
        state.valuation_multiple += 0.05
    elif state.valuation_multiple > 10.0:
        state.valuation_multiple -= 0.05

    if state.consumer_confidence < 100 and state.unemployment < 8.0:
        state.consumer_confidence += 2.0

def hill_response(spend: float, alpha: float, beta: float, gamma: float) -> float:
    """
    Hill Function for Marketing Response.
    alpha: Shape parameter (S-curve steepness)
    beta: Max potential capacity (Saturation point)
    gamma: Half-saturation point (Spend needed to reach 50% of beta)
    """
    if spend <= 0: return 0.0
    return beta * (spend ** alpha) / (gamma ** alpha + spend ** alpha)

def compute_new_mrr(state: EnvState, action: MarketingAction) -> float:
    if action.channel == "ppc":
        alpha = random.uniform(0.5, 1.0)
        gamma = random.uniform(15_000, 50_000)
        beta = random.uniform(10_000, 50_000)
    else: 
        alpha = random.uniform(1.5, 3.0)
        gamma = random.uniform(15_000, 50_000)
        beta = random.uniform(50_000, 100_000)

    response = hill_response(action.spend, alpha, beta, gamma)

    if state.consumer_confidence < 80:
        response *= 0.85
    elif state.consumer_confidence > 120:
        response *= 1.08

    if state.competitors >= 10:
        response *= 0.6
    elif state.competitors >= 4:
        response *= 0.8

    return response

def compute_churn_rate(state: EnvState) -> float:
    base = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3

    quality_factor = 1.0 - (state.product_quality * 0.5) 

    macro_multiplier = 1.0
    if state.consumer_confidence < 80:
        macro_multiplier *= 1.3
        
    avg_tenure_proxy = max(1, state.months_elapsed * 0.4)
    tenure_decay = math.exp(-0.15 * avg_tenure_proxy)
    
    decay_multiplier = max(0.3, tenure_decay)

    return base * quality_factor * macro_multiplier * decay_multiplier

def compute_expansion_mrr(state: EnvState, action: ProductAction) -> float:
    effective_rnd = action.r_and_d_spend * state.innovation_factor
    upsell_factor = 1 + min(effective_rnd / 50_000, 0.5)
    return state.mrr * 0.02 * upsell_factor

def apply_pricing_effect(state: EnvState, action: PricingAction) -> None:
    elasticity = random.uniform(-0.9, -0.2)
    demand_change = elasticity * action.price_change_pct
    
    state.price *= (1 + action.price_change_pct)
    
    state.mrr *= (1 + action.price_change_pct) * (1 + demand_change)

def apply_hiring_cost(state: EnvState, action: HiringAction) -> None:
    total_cost = action.hires * action.cost_per_employee
    state.cash -= total_cost

def compute_cac(marketing_spend: float, new_users: float) -> float:
    if new_users <= 0: return 0.0 
    raw_cac = marketing_spend / new_users
    return raw_cac

def scale_cac_by_macro(raw_cac: float, state: EnvState) -> float:
    modifier = 1.0
    
    if state.interest_rate > 5.0:
        modifier *= 1.2
        
    if state.consumer_confidence < 80:
        modifier *= 1.3
    elif state.consumer_confidence > 120:
        modifier *= 0.8
        
    if state.competitors > 5:
        modifier *= 1.15
        
    if state.competitors >= 8:
         modifier *= 1.3
        
    return raw_cac * modifier

def compute_ltv(mrr_per_user: float, churn_rate: float, discount_rate: float = 0.0) -> float:
    if churn_rate <= 0.001: churn_rate = 0.001 
    return mrr_per_user / churn_rate

def compute_rule_of_40(prev_mrr: float, new_mrr: float, burn: float) -> float:
    if prev_mrr <= 0: prev_mrr = 1.0 
    if new_mrr <= 0: new_mrr = 1.0
    
    growth_pct = ((new_mrr - prev_mrr) / prev_mrr) * 100
    margin_pct = (-burn / new_mrr) * 100
    return growth_pct + margin_pct

def compute_reward(state: EnvState, rule_of_40: float) -> float:
    reward = state.mrr / 1_000_000 

    if rule_of_40 < 15:
        reward -= 2
    if rule_of_40 < 0:
        reward -= 5

    if state.cac > 0 and state.ltv > 0:
        ratio = state.ltv / state.cac
        if ratio < 3.0:
            reward -= 5.0 
            if ratio < 1.0:
                reward -= 10.0 

    if state.cash <= 0:
        reward -= 20

    if state.innovation_factor < 0.8:
        reward -= 5

    if state.valuation_multiple < 5.0:
        reward -= 2

    return reward
