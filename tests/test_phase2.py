import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.startup_env import StartupEnv

def test_pricing_action():
    print("Testing Pricing Action...")
    env = StartupEnv()
    env.reset()
    
    action = {"type": "pricing", "params": {"price": 50.0}}
    obs, _, _, _, info = env.step(action)
    
    state = info["state"]
    print(f"State Price: {state['price']}")
    assert state['price'] == 50.0, "Price should update to 50.0"
    print("Pricing Action Passed!")

def test_marketing_diminishing_returns():
    print("Testing Marketing Diminishing Returns...")
    env = StartupEnv()
    
    env.reset()
    start_users = 100
    spend_small = 1000.0
    env.step({"type": "marketing", "params": {"amount": spend_small}})
    users_small_delta = env.current_state.users - start_users
    print(f"Spend {spend_small} -> +{users_small_delta} users")
    
    env.reset()
    spend_large = 100000.0
    env.step({"type": "marketing", "params": {"amount": spend_large}})
    users_large_delta = env.current_state.users - start_users
    print(f"Spend {spend_large} -> +{users_large_delta} users")
    
    eff_small = users_small_delta / spend_small
    eff_large = users_large_delta / spend_large
    
    print(f"Efficiency Small: {eff_small:.4f}, Large: {eff_large:.4f}")
    assert eff_large < eff_small, "Efficiency should drop with large spend (diminishing returns)"
    print("Marketing Diminishing Returns Passed!")

def test_churn_price_sensitivity():
    print("Testing Churn Price Sensitivity...")
    env = StartupEnv()
    
    env.reset()
    env.step({"type": "pricing", "params": {"price": 10.0}})
    churn_low = env.current_state.churn
    
    env.reset()
    env.step({"type": "pricing", "params": {"price": 100.0}})
    churn_high = env.current_state.churn
    
    print(f"Churn Low Price: {churn_low}, High Price: {churn_high}")
    assert churn_high > churn_low, "Higher price should lead to higher churn"
    print("Churn Price Sensitivity Passed!")

if __name__ == "__main__":
    test_pricing_action()
    test_marketing_diminishing_returns()
    test_churn_price_sensitivity()
