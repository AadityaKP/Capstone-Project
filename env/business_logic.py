import random
import math
from env.schemas import EnvState, MarketingAction, ProductAction, PricingAction, HiringAction

# =============================
# SHOCK GENERATORS
# =============================

def interest_rate_shock(state: EnvState, prob: float = 0.1) -> None:
    if random.random() < prob:
        state.interest_rate += random.uniform(0.5, 1.5)  # +50–150 bps

def consumer_confidence_shock(state: EnvState, prob: float = 0.1) -> None:
    if random.random() < prob:
        state.consumer_confidence -= random.uniform(10, 25)

def competitive_entry_shock(state: EnvState, prob: float = 0.1) -> None:
    if random.random() < prob:
        state.competitors += random.randint(1, 3)

# =============================
# TRANSITION PHYSICS
# =============================

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
    else: # brand
        alpha = random.uniform(1.5, 3.0)
        gamma = random.uniform(15_000, 50_000)
        beta = random.uniform(50_000, 100_000)

    response = hill_response(action.spend, alpha, beta, gamma)

    # Consumer confidence modifier
    if state.consumer_confidence < 80:
        response *= 0.85
    elif state.consumer_confidence > 120:
        response *= 1.08

    # Competition modifier
    if state.competitors >= 10:
        response *= 0.6
    elif state.competitors >= 4:
        response *= 0.8

    return response

def compute_churn_rate(state: EnvState) -> float:
    base = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3

    quality_factor = 1.0 - (state.product_quality * 0.5) # Better quality reduces churn

    macro_multiplier = 1.0
    if state.consumer_confidence < 80:
        macro_multiplier *= 1.3
        
    # Tenure-based Decay: P_churn = Base * exp(-0.15 * tenure)
    # Proxy: Using simulation month as proxy for "avg tenure" of the base (simplified)
    # As the company matures, the blended churn rate should drop.
    tenure_factor = math.exp(-0.05 * state.months_elapsed) # Tuned to 0.05 to be less aggressive than 0.15 for aggregate
    # Wait, user specified -0.15. If I use aggregate months_elapsed, 10 months = exp(-1.5) = 0.22 multiplier.
    # That might be too strong if we assume strictly "user tenure".
    # But for "company maturity" effect, let's stick closer to the user's requested physics but maybe cap it?
    # Let's use the user's 0.15 but applied to an estimated "avg user tenure".
    # Est Avg Tenure approx = months_elapsed * 0.5 (if growing linearly)
    avg_tenure_proxy = max(1, state.months_elapsed * 0.4)
    tenure_decay = math.exp(-0.15 * avg_tenure_proxy)
    
    # Clamp decay to not go below 0.3 (30% of base) to avoid zero churn
    decay_multiplier = max(0.3, tenure_decay)

    return base * quality_factor * macro_multiplier * decay_multiplier

def compute_expansion_mrr(state: EnvState, action: ProductAction) -> float:
    upsell_factor = 1 + min(action.r_and_d_spend / 50_000, 0.5)
    return state.mrr * 0.02 * upsell_factor

def apply_pricing_effect(state: EnvState, action: PricingAction) -> None:
    elasticity = random.uniform(-0.9, -0.2)
    demand_change = elasticity * action.price_change_pct
    
    # Update actual price attribute
    state.price *= (1 + action.price_change_pct)
    
    # Update MRR based on price change and demand reaction
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
    
    # Interest Rate: High rates = expensive capital / lower ad efficiency? 
    # Actually high rates usually mean lower demand -> higher CAC.
    if state.interest_rate > 5.0:
        modifier *= 1.2
        
    # Consumer Confidence: Low confidence = hesitant buyers = Harder to sell = High CAC
    if state.consumer_confidence < 80:
        modifier *= 1.3
    elif state.consumer_confidence > 120:
        modifier *= 0.8
        
    # Competitors: Bidding wars for ads
    if state.competitors > 5:
        modifier *= 1.15
        
    return raw_cac * modifier

def compute_ltv(mrr_per_user: float, churn_rate: float, discount_rate: float = 0.0) -> float:
    # LTV = ARPU / Churn
    # Simple perpetuity formula.
    if churn_rate <= 0.001: churn_rate = 0.001 # Cap max LTV
    return mrr_per_user / churn_rate

# =============================
# RULE OF 40 & REWARD
# =============================

def compute_rule_of_40(prev_mrr: float, new_mrr: float, burn: float) -> float:
    if prev_mrr <= 0: prev_mrr = 1.0 # Avoid div by zero
    if new_mrr <= 0: new_mrr = 1.0
    
    growth_pct = ((new_mrr - prev_mrr) / prev_mrr) * 100
    margin_pct = (-burn / new_mrr) * 100
    return growth_pct + margin_pct

def compute_reward(state: EnvState, rule_of_40: float) -> float:
    reward = state.mrr / 1_000_000  # scale

    if rule_of_40 < 15:
        reward -= 2
    if rule_of_40 < 0:
        reward -= 5

    # CAC:LTV constraint penalty
    # Research requires: CAC:LTV >= 3:1
    if state.cac > 0 and state.ltv > 0:
        ratio = state.ltv / state.cac
        if ratio < 3.0:
            reward -= 5.0 # Significant penalty for inefficient growth
            if ratio < 1.0:
                reward -= 10.0 # Crisis penalty (spending more than user worth)

    if state.cash <= 0:
        reward -= 20

    return reward
