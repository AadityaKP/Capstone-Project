import sys
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add project root to python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.startup_env import StartupEnv
from agents.adapter import ActionAdapter
from config import sim_config

# ==========================================
# Simulation Runner & Baseline Agent
# ==========================================
# This script is the "Final Exam" for the simulator.
# It runs many episodes to ensure the system is stable (doesn't crash) 
# and produces plausible results (e.g., bankruptcy is possible but not guaranteed).

class RandomAgent:
    """
    A 'Dumb' Agent that makes random choices.
    Used to stress-test the Environment and Adapter.
    """
    def get_action(self, state):
        # Pick a random strategy
        action_type = random.choice(["marketing", "hiring", "product", "pricing", "skip"])
        
        params = {}
        if action_type == "marketing":
            # Random spend between $1k and $50k
            params["amount"] = random.uniform(1000, 50000)
        elif action_type == "hiring":
            # Randomly hire 1-3 people
            params["count"] = random.randint(1, 3)
        elif action_type == "product":
            # Random R&D investment
            params["amount"] = random.uniform(5000, 20000)
        elif action_type == "pricing":
            # Random price fluctuation
            params["price"] = random.uniform(10, 100)
            
        return {"type": action_type, "params": params}

def run_simulation(num_episodes=100):
    print(f"Running {num_episodes} episodes for sanity check...")
    
    env = StartupEnv()
    agent = RandomAgent()
    
    results = []
    
    for episode in range(num_episodes):
        # Start fresh simulation
        obs, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        # Loop until Bankruptcy or Time Limit
        while not (terminated or truncated):
            # 1. Agent decides action
            raw_action = agent.get_action(env.current_state)
            
            # 2. Adapter sanitizes action (Crucial Step!)
            clean_action = ActionAdapter.translate_action(raw_action)
            
            # 3. Environment executes action
            obs, reward, terminated, truncated, info = env.step(clean_action)
            
            total_reward += reward
            steps += 1
            
        # Log episode results
        state = env.current_state
        result = {
            "episode": episode,
            "steps": steps,
            "final_cash": state.cash,
            "final_users": state.users,
            "final_revenue": state.revenue,
            "cause": "Bankruptcy" if terminated else "Time Limit",
            "total_reward": total_reward
        }
        results.append(result)
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}: {result['cause']} after {steps} weeks. Cash: ${state.cash:,.0f}")

    # --- Analysis & Reporting ---
    df = pd.DataFrame(results)
    
    print("\n--- Simulation Summary ---")
    print(f"Success Rate (Survived): {(df['cause'] == 'Time Limit').mean():.2%}")
    print(f"Avg Duration: {df['steps'].mean():.1f} weeks")
    print(f"Avg Final Cash: ${df['final_cash'].mean():,.2f}")
    print(f"Avg Final Users: {df['final_users'].mean():.1f}")
    
    # Export for further analysis
    df.to_csv("simulation_results.csv", index=False)
    print("Results saved to simulation_results.csv")
    
    return df

if __name__ == "__main__":
    run_simulation()
