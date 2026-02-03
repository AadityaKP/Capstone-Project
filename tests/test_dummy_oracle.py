import sys
import os
import time

# Add parent directory to path to allow importing 'agents'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.dummy_oracle_agent import DummyOracleAgent

def test_dummy_oracle():
    print("=== Initializing Dummy Oracle Agent ===")
    try:
        agent = DummyOracleAgent()
    except Exception as e:
        print(f"Failed to initialize agent. Ensure Neo4j is running and .env is correct.\nError: {e}")
        return

    # --- Test 1: ChromaDB (Episodic Memory) ---
    print("\n--- Test 1: ChromaDB (Episodic Memory) ---")
    
    episode_text = "The startup increased marketing spend by 20% resulting in a 15% user growth."
    metadata = {"timestamp": "2023-10-27", "strategy": "aggressive_growth"}
    
    print("1. Storing episode...")
    agent.store_episode(episode_text, metadata)
    
    # Allow a moment for ingestion if needed (usually instant for local)
    time.sleep(1) 
    
    query = "impact of marketing spend"
    print(f"2. Recalling similar to: '{query}'...")
    results = agent.recall_similar_episodes(query, k=1)
    
    print("   Result:", results['documents'][0])
    print("   Metadata:", results['metadatas'][0])

    # --- Test 2: Neo4j (Semantic Memory) ---
    print("\n--- Test 2: Neo4j (Semantic Memory) ---")
    
    triples = [
        ("MarketingSpend", "INCREASES", "UserAcquisition"),
        ("UserAcquisition", "INCREASES", "ServerLoad"),
        ("ServerLoad", "INCREASES", "Cost")
    ]
    
    print("1. Storing causal links...")
    try:
        agent.store_causal_links(triples)
        
        entity = "MarketingSpend"
        print(f"2. Recalling context for: '{entity}'...")
        context = agent.recall_entity_context(entity)
        print("   Result:", context)
        
        entity2 = "UserAcquisition"
        print(f"3. Recalling context for: '{entity2}'...")
        context2 = agent.recall_entity_context(entity2)
        print("   Result:", context2)
        
    except Exception as e:
        print(f"Neo4j operation failed: {e}")

    # Cleanup
    agent.close()
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    # Ensure potential import issues are resolved if running directly
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    test_dummy_oracle()
