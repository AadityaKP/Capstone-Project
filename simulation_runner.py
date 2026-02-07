import sys
import os
import random
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt

print("Starting simulation runner...", flush=True)

# Add project root to python path so imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.startup_env import StartupEnv
from agents.adapter import ActionAdapter
from config import sim_config

# ==========================================
# Simulation Runner & Baseline Agent
# ==========================================

class RandomBundleAgent:
    """
    A 'Dumb' Agent that makes random ActionBundles.
    Used to stress-test the Environment and Adapter.
    """
    def get_action(self, state):
        return {
            "marketing": {
                "spend": random.uniform(1000, 20000),
                "channel": random.choice(["ppc", "brand"])
            },
            "hiring": {
                "hires": random.randint(0, 2),
                "cost_per_employee": random.uniform(8000, 12000)
            },
            "product": {
                "r_and_d_spend": random.uniform(1000, 10000)
            },
            "pricing": {
                "price_change_pct": random.uniform(-0.05, 0.05)
            }
        }

def run_simulation(num_episodes=100):
    print(f"Running {num_episodes} episodes for sanity check...")
    
    env = StartupEnv()
    agent = RandomBundleAgent()
    
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
            raw_action = agent.get_action(env.state)
            
            # 2. Adapter sanitizes action (Crucial Step!)
            clean_action = ActionAdapter.translate_action(raw_action)
            
            # 3. Environment executes action
            obs, reward, terminated, truncated, info = env.step(clean_action)
            
            total_reward += reward
            steps += 1
            
        # Log episode results
        state = env.state
        result = {
            "episode": episode,
            "steps": steps,
            "final_mrr": state.mrr,
            "final_cash": state.cash,
            "final_cac": state.cac,
            "final_ltv": state.ltv,
            "final_headcount": state.headcount,
            "interest_rate": state.interest_rate,
            "consumer_confidence": state.consumer_confidence,
            "competitors": state.competitors,
            "quality": state.product_quality,
            "cause": "Bankruptcy" if terminated else "Time Limit",
            "total_reward": total_reward
        }
        results.append(result)
        
        # Print progress every 10 episodes
        if episode % 10 == 0:
            print(f"Episode {episode}: {result['cause']} after {steps} months. MRR: ${state.mrr:,.0f} Cash: ${state.cash:,.0f}")

    # --- Analysis & Reporting ---
    df = pd.DataFrame(results)
    
    print("\n--- Simulation Summary ---")
    print(f"Success Rate (Survived): {(df['cause'] == 'Time Limit').mean():.2%}")
    print(f"Avg Duration: {df['steps'].mean():.1f} months")
    print(f"Avg Final MRR: ${df['final_mrr'].mean():,.2f}")
    print(f"Avg Final Cash: ${df['final_cash'].mean():,.2f}")
    
    # Export for further analysis
    df.to_csv("simulation_results.csv", index=False)
    print("Results saved to simulation_results.csv")
    
    return df

if __name__ == "__main__":
    run_simulation()
