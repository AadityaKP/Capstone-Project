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

from agents.baseline_agents import merge_actions

# ==========================================
# Simulation Runner & Baseline Agent
# ==========================================

class BaselineJointAgent:
    """
    Acts as a container for the C-Suite agents, merging their decisions.
    """
    def get_action(self, state):
        return merge_actions(state)

class RandomBundleAgent:
    """
    A 'Dumb' Agent that makes random ActionBundles.
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

def run_simulation(policy: str = "heuristic", num_episodes: int = 100, seed_start: int = 0):
    print(f"Running {num_episodes} episodes with Policy: {policy} (Seeds {seed_start}-{seed_start+num_episodes-1})...")
    
    env = StartupEnv()
    
    if policy == "heuristic":
        agent = BaselineJointAgent()
    elif policy == "random":
        agent = RandomBundleAgent()
    else:
        raise ValueError(f"Unknown policy: {policy}")
    
    results = []
    
    for i in range(num_episodes):
        episode_seed = seed_start + i
        
        # Start fresh simulation with FIXED SEED
        obs, _ = env.reset(seed=episode_seed)
        
        # Reseed python random for agent consistency if needed (though env.reset handles env RNG)
        # But our local random agents use `random` module directly.
        random.seed(episode_seed)
        np.random.seed(episode_seed)
        
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        # Metric tracking
        rule_40_history = []
        
        # Loop until Bankruptcy or Time Limit
        while not (terminated or truncated):
            # 1. Agent decides action
            raw_action = agent.get_action(env.state)
            
            # 2. Adapter sanitizes action
            clean_action = ActionAdapter.translate_action(raw_action)
            
            # 3. Environment executes action
            obs, reward, terminated, truncated, info = env.step(clean_action)
            
            total_reward += reward
            steps += 1
            rule_40_history.append(info.get("rule_of_40", 0))
            
        # Log episode results
        state = env.state
        
        # Calc aggregate metrics
        avg_rule_40 = np.mean(rule_40_history) if rule_40_history else 0
        months_above_40 = sum(1 for x in rule_40_history if x >= 40)
        pct_above_40 = (months_above_40 / len(rule_40_history)) * 100 if rule_40_history else 0
        
        # LTV:CAC Ratio
        ltv_cac = state.ltv / state.cac if state.cac > 0 else 0
        
        result = {
            "episode": i,
            "seed": episode_seed,
            "policy": policy,
            "steps": steps,
            "final_mrr": state.mrr,
            "final_cash": state.cash,
            "final_cac": state.cac,
            "final_ltv": state.ltv,
            "final_ltv_cac": ltv_cac,
            "final_ltv_cac": ltv_cac,
            "final_headcount": state.headcount,
            "final_valuation_multiple": state.valuation_multiple,
            "final_unemployment": state.unemployment,
            "final_innovation_factor": state.innovation_factor,
            "depression_months": state.months_in_depression,
            "cause": "Bankruptcy" if terminated else "Time Limit",
            "total_reward": total_reward,
            "avg_rule_40": avg_rule_40,
            "pct_above_40": pct_above_40
        }
        results.append(result)
        
        # Print progress every 20 episodes
        if i % 20 == 0:
            print(f"Ep {i} ({policy}): {result['cause']} after {steps} mos. MRR: ${state.mrr:,.0f} Cash: ${state.cash:,.0f}")

    # --- Analysis & Reporting ---
    df = pd.DataFrame(results)
    
    print(f"\n--- Simulation Summary ({policy}) ---")
    print(f"Survival Rate: {(df['cause'] == 'Time Limit').mean():.2%}")
    print(f"Avg Duration: {df['steps'].mean():.1f} months")
    print(f"Avg Final MRR: ${df['final_mrr'].mean():,.2f}")
    print(f"Avg Rule of 40: {df['avg_rule_40'].mean():.1f}")
    print(f"Avg Innovation Factor: {df['final_innovation_factor'].mean():.2f}")
    print(f"Avg Unemployment: {df['final_unemployment'].mean():.1f}%")
    
    return df

if __name__ == "__main__":
    # Default behavior if run directly
    run_simulation(policy="heuristic", num_episodes=5)
