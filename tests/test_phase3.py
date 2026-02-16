import sys
import os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.startup_env import StartupEnv

def test_hiring_action():
    print("Testing Hiring Action...")
    env = StartupEnv()
    env.reset()
    
    initial_burn = env.current_state.burn_rate
    initial_count = env.headcount
    
    action = {"type": "hiring", "params": {"count": 2}}
    obs, _, _, _, info = env.step(action)
    
    state = info["state"]
    new_burn = state["burn_rate"]
    new_count = env.headcount
    
    print(f"Headcount: {initial_count} -> {new_count}")
    print(f"Burn: {initial_burn} -> {new_burn}")
    
    assert new_count == initial_count + 2, "Headcount should increase by 2"
    
    assert new_burn > initial_burn, "Burn rate should increase"
    env.step({"type": "skip", "params": {}})
    recurring_burn = env.current_state.burn_rate
    print(f"Recurring Burn: {recurring_burn}")
    assert recurring_burn == 11000.0, "Burn should normalize to recurring salary"
    
    print("Hiring Action Passed!")

def test_product_investment():
    print("Testing Product Investment...")
    env = StartupEnv()
    env.reset()
    
    initial_quality = env.current_state.product_quality
    
    action = {"type": "product", "params": {"amount": 50000.0}}
    env.step(action)
    
    new_quality = env.current_state.product_quality
    print(f"Quality: {initial_quality} -> {new_quality}")
    
    assert new_quality > initial_quality, "Quality should improve with investment"
    assert new_quality <= 1.0, "Quality should be capped at 1.0"
    print("Product Investment Passed!")

if __name__ == "__main__":
    test_hiring_action()
    test_product_investment()
