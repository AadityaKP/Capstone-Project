import pytest
import math
import sys
import os
# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unittest.mock import patch
from env.schemas import EnvState
from env import business_logic

@pytest.fixture
def base_state():
    return EnvState(
        mrr=50000.0,
        cash=1000000.0,
        cac=100.0,
        ltv=500.0,
        churn_enterprise=0.01,
        churn_smb=0.03,
        churn_b2c=0.05,
        interest_rate=5.0,
        consumer_confidence=100.0,
        competitors=5,
        product_quality=0.8,
        price=50.0,
        months_elapsed=0,
        valuation_multiple=10.0,
        unemployment=4.0,
        innovation_factor=1.0,
        months_in_depression=0
    )

def test_interest_rate_shock_tier1(base_state):
    # Test that interest rate shock affects valuation and churn_smb
    with patch('env.business_logic.random.random', return_value=0.0): # Force trigger
        business_logic.interest_rate_shock(base_state, prob=0.5)
    
    assert base_state.interest_rate == 6.5 # +1.5
    assert base_state.valuation_multiple == 8.5 # 10.0 * 0.85
    assert base_state.churn_smb == 0.03 * 1.2

def test_consumer_confidence_shock_tier1(base_state):
    # Test feedback to unemployment
    with patch('env.business_logic.random.random', return_value=0.0):
        business_logic.consumer_confidence_shock(base_state, prob=0.5)
    
    assert base_state.consumer_confidence == 80.0 # 100 - 20
    assert base_state.unemployment == 5.0 # 4.0 + 1.0

def test_competitive_entry_shock_dynamic(base_state):
    # Test dynamic probability
    # Case 1: Low MRR (50k) -> Dynamic Prob near 0.5 (sigmoid(0) = 0.5)
    # Actual prob = prob * 2 * 0.5 = prob
    
    # Case 2: High MRR behavior
    base_state.mrr = 100_000 # (100k-50k)/50k = 1.0. sigmoid(1) ~= 0.73. 
    # Actual prob = prob * 2 * 0.73 = 1.46 * prob
    
    with patch('env.business_logic.random.random', return_value=0.0):
        business_logic.competitive_entry_shock(base_state, prob=0.1)
    
    assert base_state.competitors == 6
    assert base_state.price == 50.0 * 0.9

def test_recession_cascade_tier2(base_state):
    # Setup trigger conditions
    base_state.unemployment = 9.0 # > 8.0
    base_state.interest_rate = 8.0 # > 7.0
    
    # Force stochastic trigger (random < 0.2)
    with patch('env.business_logic.random.random', return_value=0.1):
        business_logic.apply_recession_cascade(base_state)
    
    assert base_state.consumer_confidence == 90.0 # 100 - 10
    assert base_state.valuation_multiple == 8.0 # 10.0 * 0.8
    assert base_state.unemployment == 9.5 # +0.5

def test_hysteresis_scarring(base_state):
    # Set depression condition
    base_state.consumer_confidence = 40.0
    base_state.months_in_depression = 5
    
    # Step 1: Should increment month, and immediately trigger scar if it becomes 6
    business_logic.apply_hysteresis(base_state)
    assert base_state.months_in_depression == 6
    assert base_state.innovation_factor == 0.95

    # Step 2: Next month, it continues to scar
    business_logic.apply_hysteresis(base_state)
    assert base_state.months_in_depression == 7
    # 0.95 * 0.95 = 0.9025
    assert base_state.innovation_factor == 0.9025

def test_recovery_dynamics(base_state):
    base_state.innovation_factor = 0.9
    base_state.valuation_multiple = 8.0
    base_state.consumer_confidence = 90.0
    base_state.unemployment = 5.0
    
    business_logic.apply_recovery(base_state)
    
    assert base_state.innovation_factor == 0.901
    assert base_state.valuation_multiple == 8.05
    assert base_state.consumer_confidence == 92.0

def test_reward_penalties(base_state):
    # Test innovation penalty
    base_state.innovation_factor = 0.7 
    # Test valuation penalty
    base_state.valuation_multiple = 4.0
    
    reward = business_logic.compute_reward(base_state, rule_of_40=20.0)
    # Base reward = 50k/1m = 0.05
    # Innovation penalty (<0.8) = -5
    # Valuation penalty (<5.0) = -2
    # Total = 0.05 - 7
    
    assert reward < -6.0
