"""Ollama-focused LLM test suite with actual reasoning captured in CSV."""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import csv
import time
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

from agents.llm_client import create_llm_client
from env.schemas import EnvState
from agents.proposal_agents import CFOProposalAgent, CMOProposalAgent, CPOProposalAgent

# Create CSV file for results
csv_filename = "ollama_test_results.csv"
results = []

def add_result(test_type, agent_role, agent_action, heuristic_impact, llm_impact, is_refined, duration_ms, error_message=""):
    """Add a test result to the results list."""
    results.append({
        "timestamp": datetime.now().isoformat(),
        "test_type": test_type,
        "agent_role": agent_role,
        "agent_action": agent_action,
        "heuristic_impact": heuristic_impact,
        "llm_impact": llm_impact,
        "is_refined": is_refined,
        "duration_ms": duration_ms,
        "error_message": error_message,
    })

# Create a realistic business state
state = EnvState(
    mrr=75000,
    cash=250000,
    cac=500,
    ltv=5000,
    churn_enterprise=0.02,
    churn_smb=0.05,
    churn_b2c=0.12,
    interest_rate=5.5,
    consumer_confidence=72,
    competitors=5,
    product_quality=0.78,
    price=150,
    months_elapsed=12,
    headcount=15,
    unemployment=4.2,
    innovation_factor=0.85,
    months_in_depression=0
)

print("=" * 100)
print("OLLAMA AGENT TEST - LLM REASONING CAPTURE")
print("=" * 100)
print(f"\nBusiness State:")
print(f"  MRR: ${state.mrr:,.0f}")
print(f"  Cash: ${state.cash:,.0f}")
print(f"  Headcount: {state.headcount}")
print(f"  Competitors: {state.competitors}")
print(f"  Consumer Confidence: {state.consumer_confidence}")
print(f"  Innovation Factor: {state.innovation_factor:.2f}\n")

agents_to_test = [
    ("CFO", CFOProposalAgent, "Cost Control & Runway"),
    ("CMO", CMOProposalAgent, "Growth & Acquisition"),
    ("CPO", CPOProposalAgent, "Product & Retention"),
]

for role, AgentClass, action_desc in agents_to_test:
    print("=" * 100)
    print(f"{role} AGENT - {action_desc}")
    print("=" * 100)
    
    # Heuristic test
    start = time.time()
    try:
        agent_heuristic = AgentClass(use_llm=False)
        prop_heuristic = agent_heuristic.propose(state)
        duration_heur = (time.time() - start) * 1000
        
        print(f"\n[HEURISTIC]")
        print(f"  Impact: {prop_heuristic.expected_impact}")
        print(f"  Actions: {prop_heuristic.actions}")
        print(f"  Confidence: {prop_heuristic.confidence}")
        print(f"  Duration: {duration_heur:.1f}ms")
    except Exception as e:
        print(f"[HEURISTIC] Error: {e}")
        prop_heuristic = None
        duration_heur = 0
    
    # LLM test with Ollama
    start = time.time()
    try:
        llm_client = create_llm_client("ollama", "llama3.1:8b")
        agent_llm = AgentClass(llm_client=llm_client, use_llm=True)
        prop_llm = agent_llm.propose(state)
        duration_llm = (time.time() - start) * 1000
        
        is_refined = prop_llm.expected_impact != prop_heuristic.expected_impact if prop_heuristic else False
        status_indicator = "✓ REFINED" if is_refined else "↳ FALLBACK"
        
        print(f"\n[LLM - OLLAMA]")
        print(f"  Impact: {prop_llm.expected_impact}")
        print(f"  Actions: {prop_llm.actions}")
        print(f"  Confidence: {prop_llm.confidence}")
        print(f"  Duration: {duration_llm:.1f}ms")
        print(f"  Status: {status_indicator}")
        
        # Add to results
        add_result(
            test_type="proposal_generation",
            agent_role=role,
            agent_action=action_desc,
            heuristic_impact=prop_heuristic.expected_impact if prop_heuristic else "N/A",
            llm_impact=prop_llm.expected_impact,
            is_refined=is_refined,
            duration_ms=round(duration_llm, 2),
            error_message=""
        )
        
    except Exception as e:
        print(f"[LLM - OLLAMA] Error: {e}")
        add_result(
            test_type="proposal_generation",
            agent_role=role,
            agent_action=action_desc,
            heuristic_impact=prop_heuristic.expected_impact if prop_heuristic else "N/A",
            llm_impact="ERROR",
            is_refined=False,
            duration_ms=0,
            error_message=str(e)
        )
    
    print()

# ============================================================================
# Write results to CSV
# ============================================================================
print("=" * 100)
print("WRITING RESULTS TO CSV")
print("=" * 100)

csv_path = os.path.join(os.path.dirname(__file__), csv_filename)

try:
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "timestamp",
            "test_type",
            "agent_role",
            "agent_action",
            "heuristic_impact",
            "llm_impact",
            "is_refined",
            "duration_ms",
            "error_message"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ CSV file created: {csv_path}")
    print(f"✓ Total test results: {len(results)}")
    
    # Print summary statistics
    print("\n" + "=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    
    refined = sum(1 for r in results if r["is_refined"])
    fallback = len(results) - refined
    
    print(f"Total Tests: {len(results)}")
    print(f"  ✓ Refined (Ollama improved reasoning): {refined}")
    print(f"  ↳ Fallback (used heuristic): {fallback}")
    
    if results:
        avg_duration = sum(r["duration_ms"] for r in results if r["duration_ms"] > 0) / max(1, len([r for r in results if r["duration_ms"] > 0]))
        print(f"Average Duration: {avg_duration:.1f}ms")
        
        agents_tested = set(r["agent_role"] for r in results)
        print(f"Agents Tested: {', '.join(sorted(agents_tested))}")
    
    print(f"\nCSV saved to: tests/{csv_filename}")
    
except Exception as e:
    print(f"✗ Error writing CSV: {e}")
