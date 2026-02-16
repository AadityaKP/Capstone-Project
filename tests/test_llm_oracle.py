import sys
import os
import time
import json
from pprint import pprint

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.oracle_agent import OracleAgent

def test_llm_oracle():
    print("=== Testing LLM Oracle Agent ===")
    
    try:
        agent = OracleAgent(model_name="llama3.1:8b") 
    except Exception as e:
        print(f"Failed to init agent: {e}")
        return

    print("\n--- Seeding Transient Context ---")
    agent.store_episode(
        "In Q1, we increased R&D budget by 50%, which led to a breakthrough in the AI algorithm but caused a short-term cash flow dip.", 
        {"quarter": "Q1-Test"}
    )
    agent.store_causal_links([["R&D Spend", "INCREASES", "Innovation"], ["Innovation", "INCREASES", "ProductValue"]])

    query = "We are planning to increase R&D spend again. What should we expect?"
    print(f"\n--- Analyzing Query: '{query}' ---")
    
    start_time = time.time()
    result = agent.analyze_situation(query)
    duration = time.time() - start_time
    
    print(f"\n[Analysis Completed in {duration:.2f}s]")
    print("\n--- JSON Result ---")
    pprint(result)

    if not isinstance(result, dict):
        print("❌ FAILED: Result is not a dictionary.")
    elif "insight" in result:
        print("✅ PASSED: 'insight' field found.")
    else:
        print("⚠️ WARNING: JSON might be malformed or missing fields.")

    if result.get("suggested_causal_links"):
        print(f"✅ PASSED: Write-back triggered for {len(result['suggested_causal_links'])} links.")
    
    agent.close()

if __name__ == "__main__":
    test_llm_oracle()
