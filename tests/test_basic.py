import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.startup_env import StartupEnv

def test_initialization():
    print("Testing Initialization...")
    env = StartupEnv()
    obs, info = env.reset()
    
    assert obs[0] == 1000000.0, f"Initial cash should be 1M, got {obs[0]}"
    assert obs[1] == 100, f"Initial users should be 100, got {obs[1]}"
    print("Initialization Passed!")

def test_step():
    print("Testing Step...")
    env = StartupEnv()
    env.reset()
    
    action = {
        "type": "marketing",
        "params": {"amount": 10000.0}
    }
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    state = info["state"]
    print(f"Post-Step State: {state}")
    
    assert state["users"] > 100, "Marketing should increase users"
    assert state["cash"] < 1000000.0, "Spending should decrease cash"
    assert not terminated, "Should not terminate after one step"
    print("Step Mechanic Passed!")

if __name__ == "__main__":
    test_initialization()
    test_step()
