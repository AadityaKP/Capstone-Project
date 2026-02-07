# Oracle Agent Documentation

## Overview
The **Oracle Agent** is an LLM-augmented intelligence designed to serve as the "brain" of the startup simulator. It provides context-aware insights, predicts market effects, and builds a persistent memory of the simulation's history and causal logic.

**File Location**: `agents/oracle_agent.py`

## Architecture
The agent operates on a **Retrieval-Augmented Generation (RAG)** loop with a **Write-Back** mechanism.

1.  **Input**: Natural language user query.
2.  **Retrieval**: Fetches relevant context from two databases.
3.  **Reasoning**: Uses an LLM (Llama 3.1 via Ollama) to synthesize an answer.
4.  **Write-Back**: Stores the new experience and any inferred causal links back into memory.
5.  **Output**: Structured JSON containing insights and predictions.

## Inputs & Outputs

### Input
-   **Query (`str`)**: A natural language string describing the current simulation state or asking for advice (e.g., *"We increased marketing spend by 50% but churn is high. Why?"*)

### Output
Returns a `Dictionary` with the following structure:
```json
{
    "insight": "A one-sentence high-level summary of the situation.",
    "predicted_effects": ["List of likely future consequences", "e.g., CAC will rise"],
    "suggested_causal_links": [
        ["Subject", "PREDICATE", "Object"] 
    ],
    "store_episode_summary": "A consolidated text summary of this analysis for future recall."
}
```

## Internal Working Components

### 1. Memory Systems
The agent uses a Dual-Memory architecture:
*   **Episodic Memory (ChromaDB)**: Stores unstructured text summaries of past events (e.g., "In Q1, we hired 5 engineers"). Used for recalling similar past situations.
*   **Causal Memory (Neo4j)**: Stores structured Knowledge Graph triples (e.g., `(Marketing) --[INCREASES]--> (BrandAwareness)`). Used for understanding variable relationships.

### 2. Analysis Pipeline (`analyze_situation`)
When `analyze_situation(query)` is called:
1.  **Context Retrieval**:
    *   Calls `recall_similar_episodes(query)` to get top-k matches from Chroma.
    *   Calls `recall_entity_context(query)` to find related nodes/edges in Neo4j.
2.  **Prompt Construction**: Dynamically builds a prompt including the User Query, Past Episodes, and Causal Facts.
3.  **LLM Inference**: Sends the prompt to Ollama (`llama3.1:8b`).
4.  **Parsing & Safety**: Standardizes the LLM's raw string output into valid JSON.

### 3. Write-Back Mechanism
To ensure the agent learns over time:
*   **Episodes**: The `store_episode_summary` from the output is embedded and saved to ChromaDB.
*   **Causal Links**: Any `suggested_causal_links` (e.g., `["Price", "DECREASES", "Demand"]`) are merged into the Neo4j graph.

## Setup & Dependencies
*   **Python Libraries**: `chromadb`, `neo4j`, `ollama` (client), `pydantic`, `python-dotenv`.
*   **External Services**:
    *   **Ollama**: Must be running locally (`ollama serve`) with `llama3.1` pulled.
    *   **Neo4j**: Database must be active (default: `bolt://localhost:7687`).
    *   **ChromaDB**: usage is local file-based by default.
