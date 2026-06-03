"""Comprehensive LLM test suite with CSV reporting."""
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
csv_filename = "llm_test_results.csv"
results = []

def add_result(test_type, provider, agent_role, status, response_length, error_message="", duration_ms=0):
    """Add a test result to the results list."""
    results.append({
        "timestamp": datetime.now().isoformat(),
        "test_type": test_type,
        "provider": provider,
        "agent_role": agent_role,
        "status": status,
        "response_length": response_length,
        "duration_ms": duration_ms,
        "error_message": error_message,
    })

# ============================================================================
# TEST 1: Provider Smoke Test
# ============================================================================
print("=" * 80)
print("TEST 1: PROVIDER SMOKE TEST")
print("=" * 80)

providers = ["ollama", "openai", "anthropic", "dummy"]

for provider in providers:
    start = time.time()
    try:
        client = create_llm_client(provider)
        result = client.complete(
            "You are a test assistant.",
            "Reply with the single word PONG and nothing else."
        )
        duration = (time.time() - start) * 1000
        status = "SUCCESS" if isinstance(result, str) else "FAILED"
        error_msg = "" if status == "SUCCESS" else "Did not return string"
        
        add_result(
            test_type="provider_smoke_test",
            provider=provider,
            agent_role="N/A",
            status=status,
            response_length=len(result),
            error_message=error_msg,
            duration_ms=round(duration, 2)
        )
        
        print(f"[{provider}] ✓ PASSED - Response length: {len(result)}, Duration: {duration:.1f}ms")
    except Exception as e:
        duration = (time.time() - start) * 1000
        add_result(
            test_type="provider_smoke_test",
            provider=provider,
            agent_role="N/A",
            status="FAILED",
            response_length=0,
            error_message=str(e),
            duration_ms=round(duration, 2)
        )
        print(f"[{provider}] ✗ FAILED - {str(e)[:60]}")

# ============================================================================
# TEST 2: LLM vs Heuristic Comparison
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: LLM VS HEURISTIC COMPARISON")
print("=" * 80)

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

agents_to_test = [
    ("CFO", CFOProposalAgent, "openai"),
    ("CMO", CMOProposalAgent, "anthropic"),
    ("CPO", CPOProposalAgent, "anthropic"),
]

for role, AgentClass, provider in agents_to_test:
    print(f"\n[{role} Agent]")
    
    # Heuristic test
    start = time.time()
    try:
        agent_heuristic = AgentClass(use_llm=False)
        prop_heuristic = agent_heuristic.propose(state)
        duration = (time.time() - start) * 1000
        
        add_result(
            test_type="proposal_generation",
            provider="heuristic",
            agent_role=role,
            status="SUCCESS",
            response_length=len(prop_heuristic.expected_impact),
            error_message="",
            duration_ms=round(duration, 2)
        )
        print(f"  Heuristic: ✓ PASSED - Impact: '{prop_heuristic.expected_impact[:40]}...' ({duration:.1f}ms)")
    except Exception as e:
        duration = (time.time() - start) * 1000
        add_result(
            test_type="proposal_generation",
            provider="heuristic",
            agent_role=role,
            status="FAILED",
            response_length=0,
            error_message=str(e),
            duration_ms=round(duration, 2)
        )
        print(f"  Heuristic: ✗ FAILED - {str(e)[:50]}")
    
    # LLM test
    start = time.time()
    try:
        llm_client = create_llm_client(provider)
        agent_llm = AgentClass(llm_client=llm_client, use_llm=True)
        prop_llm = agent_llm.propose(state)
        duration = (time.time() - start) * 1000
        
        # Check if LLM actually refined it (not just fallback)
        is_refined = prop_llm.expected_impact != prop_heuristic.expected_impact
        status = "SUCCESS_REFINED" if is_refined else "SUCCESS_FALLBACK"
        
        add_result(
            test_type="proposal_generation",
            provider=provider,
            agent_role=role,
            status=status,
            response_length=len(prop_llm.expected_impact),
            error_message="",
            duration_ms=round(duration, 2)
        )
        
        refinement_indicator = "→ REFINED" if is_refined else "(fallback)"
        print(f"  LLM ({provider}): ✓ PASSED - Impact length: {len(prop_llm.expected_impact)} {refinement_indicator} ({duration:.1f}ms)")
    except Exception as e:
        duration = (time.time() - start) * 1000
        add_result(
            test_type="proposal_generation",
            provider=provider,
            agent_role=role,
            status="FAILED",
            response_length=0,
            error_message=str(e),
            duration_ms=round(duration, 2)
        )
        print(f"  LLM ({provider}): ✗ FAILED - {str(e)[:50]}")

# ============================================================================
# TEST 3: Factory Function Test
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: FACTORY FUNCTION TEST")
print("=" * 80)

factory_providers = ["ollama", "openai", "anthropic", "dummy"]
for provider in factory_providers:
    start = time.time()
    try:
        client = create_llm_client(provider)
        has_complete = hasattr(client, "complete")
        duration = (time.time() - start) * 1000
        
        status = "SUCCESS" if has_complete else "FAILED"
        error_msg = "" if status == "SUCCESS" else "Missing complete() method"
        
        add_result(
            test_type="factory_function",
            provider=provider,
            agent_role="N/A",
            status=status,
            response_length=int(has_complete),
            error_message=error_msg,
            duration_ms=round(duration, 2)
        )
        
        print(f"[{provider}] ✓ PASSED - Factory created client with complete() method ({duration:.1f}ms)")
    except Exception as e:
        duration = (time.time() - start) * 1000
        add_result(
            test_type="factory_function",
            provider=provider,
            agent_role="N/A",
            status="FAILED",
            response_length=0,
            error_message=str(e),
            duration_ms=round(duration, 2)
        )
        print(f"[{provider}] ✗ FAILED - {str(e)[:60]}")

# ============================================================================
# Write results to CSV
# ============================================================================
print("\n" + "=" * 80)
print("WRITING RESULTS TO CSV")
print("=" * 80)

csv_path = os.path.join(os.path.dirname(__file__), csv_filename)

try:
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            "timestamp",
            "test_type",
            "provider",
            "agent_role",
            "status",
            "response_length",
            "duration_ms",
            "error_message"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✓ CSV file created: {csv_path}")
    print(f"✓ Total test results: {len(results)}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    passed = sum(1 for r in results if "SUCCESS" in r["status"])
    failed = sum(1 for r in results if r["status"] == "FAILED")
    refined = sum(1 for r in results if r["status"] == "SUCCESS_REFINED")
    fallback = sum(1 for r in results if r["status"] == "SUCCESS_FALLBACK")
    
    print(f"Total Tests: {len(results)}")
    print(f"  ✓ Passed: {passed}")
    print(f"    - Refined (LLM improved): {refined}")
    print(f"    - Fallback (heuristic): {fallback}")
    print(f"  ✗ Failed: {failed}")
    
    if results:
        avg_duration = sum(r["duration_ms"] for r in results if r["duration_ms"] > 0) / max(1, len([r for r in results if r["duration_ms"] > 0]))
        print(f"Average Duration: {avg_duration:.1f}ms")
        
        providers_tested = set(r["provider"] for r in results if r["provider"] != "N/A")
        print(f"Providers Tested: {', '.join(sorted(providers_tested))}")
    
    print(f"\nCSV saved to: tests/{csv_filename}")
    
except Exception as e:
    print(f"✗ Error writing CSV: {e}")
