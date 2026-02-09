import sys
import os
import pandas as pd
import numpy as np

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation_runner import run_simulation

def run_baseline_experiment():
    print("==================================================")
    print("PHASE 2 BASELINE EVALUATION: Random vs Heuristic")
    print("==================================================")
    
    NUM_EPISODES = 200 # Per policy
    SEED_START = 0
    
    # 1. Run Random Policy
    print("\n>>> Executing Random Policy...")
    df_random = run_simulation(policy="random", num_episodes=NUM_EPISODES, seed_start=SEED_START)
    df_random.to_csv("baseline_raw_random.csv", index=False)
    
    # 2. Run Heuristic Policy
    print("\n>>> Executing Heuristic Policy...")
    df_heuristic = run_simulation(policy="heuristic", num_episodes=NUM_EPISODES, seed_start=SEED_START)
    df_heuristic.to_csv("baseline_raw_heuristic.csv", index=False)
    
    # 3. Compute Comparative Metrics
    print("\n>>> Computing Comparative Metrics...")
    
    def compute_metrics(df, policy_name):
        survived = (df["cause"] == "Time Limit")
        survival_rate = survived.mean() * 100
        avg_lifespan = df["steps"].mean()
        
        median_mrr = df["final_mrr"].median()
        mean_mrr = df["final_mrr"].mean()
        
        # Growth Milestones
        # $1M ARR = $83,333 MRR
        # $10M ARR = $833,333 MRR
        reached_1m_arr = (df["final_mrr"] >= 83_333).mean() * 100
        reached_10m_arr = (df["final_mrr"] >= 833_333).mean() * 100
        
        # Efficiency
        avg_rule_40 = df["avg_rule_40"].mean()
        median_ltv_cac = df["final_ltv_cac"].median()
        
        # Risk
        # Bankruptcy rate is inverse of survival rate
        bankruptcy_rate = 100 - survival_rate
        
        # Median Cash at Failure (only for bankrupt episodes)
        bankrupt_mask = df["cause"] == "Bankruptcy"
        median_cash_fail = df.loc[bankrupt_mask, "final_cash"].median() if bankrupt_mask.any() else 0.0
        
        return {
            "Policy": policy_name,
            "Survival %": f"{survival_rate:.1f}%",
            "Avg Lifespan (Mo)": f"{avg_lifespan:.1f}",
            "Median MRR": f"${median_mrr:,.0f}",
            "Mean MRR": f"${mean_mrr:,.0f}",
            "% > $1M ARR": f"{reached_1m_arr:.1f}%",
            "% > $10M ARR": f"{reached_10m_arr:.1f}%",
            "Avg Rule-40": f"{avg_rule_40:.1f}",
            "Median LTV:CAC": f"{median_ltv_cac:.2f}",
            "Bankruptcy %": f"{bankruptcy_rate:.1f}%",
            "Median Cash @ Fail": f"${median_cash_fail:,.0f}"
        }

    stats_random = compute_metrics(df_random, "Random")
    stats_heuristic = compute_metrics(df_heuristic, "Heuristic")
    
    # 4. Generate & Save Table
    comparison_df = pd.DataFrame([stats_random, stats_heuristic])
    
    print("\n=== FINAL BASELINE COMPARISON ===")
    print(comparison_df.to_markdown(index=False))
    
    comparison_df.to_csv("baseline_metrics.csv", index=False)
    print("\nSaved to baseline_metrics.csv")

if __name__ == "__main__":
    run_baseline_experiment()
