import numpy as np
from config import sim_config

# ==========================================
# Business Logic Layer
# ==========================================
# This module contains the "Laws of Physics" for the market.
# It defines purely functional transitions: Inputs -> Outputs.
# Ideally, this file should be stateless. All state is held by the Env.

def apply_marketing_effect(current_users: int, current_brand: float, spend: float, channel_efficiency: float = 1.0) -> tuple[int, float]:
    """
    Calculates the impact of Marketing Spend on User Acquisition and Brand.
    
    The Logic:
    1. Marketing is NOT linear. Spending $1M is not 10x better than $100k due to saturation.
    2. We use a Power Law / Diminishing Returns model: (Spend ^ 0.85).
    3. Brand Strength acts as a Multiplier. A known brand acquires users cheaper.
    
    Args:
        current_users: Existing user count.
        current_brand: Current brand score (0.0 - 1.0+).
        spend: Cash spent on marketing this step.
        channel_efficiency: Multiplier for different channels (default 1.0).
        
    Returns:
        (new_total_users, new_brand_strength)
    """
    if spend <= 0:
        return current_users, current_brand
    
    # Brand Multiplier: 
    # If Brand is 0.0 -> Multiplier is 1.0x (Base efficiency).
    # If Brand is 1.0 -> Multiplier is 2.0x (Double efficiency).
    brand_multiplier = 1.0 + current_brand
    
    # Effective Spend:
    # Adjust raw cash by the efficiency of the channel (placeholder for future channel selection).
    effective_spend = spend * channel_efficiency
    
    # User Acquisition Formula:
    # NewUsers = (EffectiveSpend ^ 0.85) * BrandMult * (10 / BaseCAC)
    # The power 0.85 ensures diminishing returns.
    raw_new_users = (effective_spend ** 0.85) * brand_multiplier * (10.0 / sim_config.BASE_CAC)
    new_users = int(raw_new_users)
    
    # Brand Growth:
    # Brand grows with spend, but very slowly.
    # It follows a Sigmoid-like saturation via (x / (x + k)).
    # Spending $50k adds ~0.025 to brand.
    brand_delta = (spend / (spend + 50000.0)) * 0.05
    
    # Cap Brand at 1.0? 
    # logic below uses min(..., 1.0) but brand can arguably go higher. 
    # For Phase 2, we cap at 1.0 for simplicity.
    return current_users + new_users, min(current_brand + brand_delta, 1.0)

def calculate_churn(product_quality: float, price: float, competition_price: float = 20.0) -> float:
    """
    Calculates the % of users leaving the platform this week.
    
    Drivers:
    1. Product Quality: Low quality = High Churn.
    2. Price: Price significantly higher than competition = High Churn.
    """
    base_churn = sim_config.MIN_CHURN
    
    # Quality Effect:
    # If Quality is 0.1 (terrible) -> Penalty is 0.18 (Huge churn increase).
    # If Quality is 1.0 (perfect) -> Penalty is 0.0.
    quality_penalty = 0.2 * (1.0 - product_quality)
    
    # Price Effect (Elasticity):
    # We compare our Price vs "Market Standard" ($20).
    # If Price > $20, churn increases linearly with the ratio.
    price_ratio = price / max(competition_price, 0.01)
    price_penalty = 0.0
    if price_ratio > 1.0:
        # For every 100% price increase over competitor, add 15% churn.
        price_penalty = 0.15 * (price_ratio - 1.0)
        
    raw_churn = base_churn + quality_penalty + price_penalty
    
    # Hard Clip: Churn cannot exceed MAX_CHURN (30%) or be negative.
    return np.clip(raw_churn, 0.0, sim_config.MAX_CHURN)

def apply_product_investment(current_quality: float, spend: float) -> float:
    """
    Improves Product Quality based on R&D Spend.
    
    Logic:
    - It is harder to improve a product that is already good (Asymptotic to 1.0).
    - Improvement = Spend * Efficiency * (1 - Current)
    - The (1 - Current) term ensures we never exceed 1.0.
    """
    if spend <= 0:
        return current_quality
        
    improvement_potential = 1.0 - current_quality
    efficiency = 0.000001 # Tuned constant. $10k spend -> ~0.01 gain.
    
    delta = spend * efficiency * improvement_potential
    
    # "Big Bet" Bonus:
    # Spending > $20k gets a 1.2x multiplier to simulate economy of scale / breakthrough potential.
    if spend > 20000:
        delta *= 1.2
        
    return min(current_quality + delta, 1.0)

def calculate_burn(headcount: int, infrastructure_cost: float) -> float:
    """
    Calculates total weekly burn rate.
    
    Components:
    1. Salaries: Headcount * Avg Rate ($2k/week).
    2. Infrastructure: Variable costs (server bills).
    3. Fixed Overhead: MIN_BURN_RATE (Office, Legal).
    """
    salary_burn = headcount * 2000.0 # ~$100k/year per employee
    return salary_burn + infrastructure_cost + sim_config.MIN_BURN_RATE

def calculate_revenue(users: int, price: float) -> float:
    """
    Simple Revenue = Volume * Price.
    """
    return users * price

def apply_stochastic_shock(value: float, volatility: float = sim_config.MARKET_VOLATILITY) -> float:
    """
    Simulates Market Randomness.
    Multiplies the input value by a random factor drawn from a Normal Distribution N(1.0, volatility).
    
    This represents:
    - Unexpected viral growth.
    - Server outages (negative shock).
    - Macroeconomic shifts.
    """
    if value == 0: return 0.0
    shock = np.random.normal(1.0, volatility)
    return value * shock
