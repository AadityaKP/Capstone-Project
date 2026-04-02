import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation_runner import run_simulation

OUTPUT_DIR = Path("results") / "future_experiments" / "baseline_comparison"

def run_baseline_experiment():
    print("==================================================")
    print("ORACLE MEMORY UPGRADE EVALUATION")
    print("==================================================")
    
    MODE = "eval"  # Change to "eval" for final 200-episode runs
    
    if MODE == "dev":
        NUM_EPISODES = 10
        ORACLE_FREQUENCY = 10
    else:
        NUM_EPISODES = 200
        ORACLE_FREQUENCY = 10

    print(f"\n[CONFIG] Mode: {MODE.upper()} | Episodes: {NUM_EPISODES} | Oracle Freq: {ORACLE_FREQUENCY} months")
    
    SEED_START = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    policy_runs = [
        ("boardroom", "Boardroom Baseline", OUTPUT_DIR / "boardroom_baseline_raw.csv"),
        ("oracle_v1", "Oracle v1", OUTPUT_DIR / "oracle_v1_raw.csv"),
        ("oracle_v2", "Oracle v2", OUTPUT_DIR / "oracle_v2_raw.csv"),
        ("oracle_v3", "Oracle v3", OUTPUT_DIR / "oracle_v3_raw.csv"),
    ]

    raw_results = {}
    for policy, label, filename in policy_runs:
        print(f"\n>>> Executing {label}...")
        df_result = run_simulation(
            policy=policy,
            num_episodes=NUM_EPISODES,
            seed_start=SEED_START,
            oracle_frequency=ORACLE_FREQUENCY,
        )
        df_result.to_csv(filename, index=False)
        raw_results[policy] = df_result
    
    print("\n>>> Computing Comparative Metrics...")
    
    def compute_metrics(df, policy_name):
        survived = (df["cause"] == "Time Limit")
        survival_rate = survived.mean() * 100
        avg_lifespan = df["steps"].mean()
        
        median_mrr = df["final_mrr"].median()
        mean_mrr = df["final_mrr"].mean()
        
        reached_1m_arr = (df["final_mrr"] >= 83_333).mean() * 100
        reached_10m_arr = (df["final_mrr"] >= 833_333).mean() * 100
        
        avg_rule_40 = df["avg_rule_40"].mean()
        median_ltv_cac = df["final_ltv_cac"].median()
        
        bankruptcy_rate = 100 - survival_rate
        
        bankrupt_mask = df["cause"] == "Bankruptcy"
        median_cash_fail = df.loc[bankrupt_mask, "final_cash"].median() if bankrupt_mask.any() else 0.0
        
        avg_innovation = df["final_innovation_factor"].mean()
        avg_unemployment = df["final_unemployment"].mean()
        avg_depression_months = df["depression_months"].mean()
        avg_valuation = df["final_valuation_multiple"].mean()

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
            "Median Cash @ Fail": f"${median_cash_fail:,.0f}",
            "Avg Innovation": f"{avg_innovation:.2f}",
            "Avg Unemployment": f"{avg_unemployment:.1f}%",
            "Avg Depression Mos": f"{avg_depression_months:.1f}",
            "Avg Valuation": f"{avg_valuation:.1f}x"
        }

    comparison_df = pd.DataFrame(
        [
            compute_metrics(raw_results["boardroom"], "Boardroom Baseline"),
            compute_metrics(raw_results["oracle_v1"], "Oracle v1"),
            compute_metrics(raw_results["oracle_v2"], "Oracle v2"),
            compute_metrics(raw_results["oracle_v3"], "Oracle v3"),
        ]
    )
    
    print("\n=== FINAL BASELINE COMPARISON ===")
    print(comparison_df.to_markdown(index=False))
    
    comparison_path = OUTPUT_DIR / "oracle_memory_upgrade_metrics.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\nSaved to {comparison_path}")

if __name__ == "__main__":
    run_baseline_experiment()
