import pytest
import math
import sys
import os

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
    with patch('env.business_logic.random.random', return_value=0.0): 
        business_logic.interest_rate_shock(base_state, prob=0.5)
    
    assert base_state.interest_rate == 6.5 
    assert base_state.valuation_multiple == 8.5 
    assert base_state.churn_smb == 0.03 * 1.2

def test_consumer_confidence_shock_tier1(base_state):
    with patch('env.business_logic.random.random', return_value=0.0):
        business_logic.consumer_confidence_shock(base_state, prob=0.5)
    
    assert base_state.consumer_confidence == 80.0 
    assert base_state.unemployment == 5.0 

def test_competitive_entry_shock_dynamic(base_state):
    base_state.mrr = 100_000 
    
    with patch('env.business_logic.random.random', return_value=0.0):
        business_logic.competitive_entry_shock(base_state, prob=0.1)
    
    assert base_state.competitors == 6
    assert base_state.price == 50.0 * 0.9

def test_recession_cascade_tier2(base_state):
    base_state.unemployment = 9.0 
    base_state.interest_rate = 8.0 
    
    with patch('env.business_logic.random.random', return_value=0.1):
        business_logic.apply_recession_cascade(base_state)
    
    assert base_state.consumer_confidence == 90.0 
    assert base_state.valuation_multiple == 8.0 
    assert base_state.unemployment == 9.5 

def test_hysteresis_scarring(base_state):
    base_state.consumer_confidence = 40.0
    base_state.months_in_depression = 5
    
    business_logic.apply_hysteresis(base_state)
    assert base_state.months_in_depression == 6
    assert base_state.innovation_factor == 0.95

    business_logic.apply_hysteresis(base_state)
    assert base_state.months_in_depression == 7
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
    base_state.innovation_factor = 0.7 
    base_state.valuation_multiple = 4.0
    
    reward = business_logic.compute_reward(base_state, rule_of_40=20.0)
    
    assert reward < -6.0
