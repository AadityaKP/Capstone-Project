# Component Explainer: Setup & Configuration

**Core Path**: `config/`, `seed_dbs.py`, `.env`

---

## 1. The Control Panel (`config/sim_config.py`)

This file contains the **Global Constants** that tune the simulation.

### Key Parameters
*   **Initialization**: `INITIAL_CASH` ($1M), `INITIAL_PRODUCT_QUALITY` (0.5).
*   **Time**: `MAX_STEPS` (120 months = 10 years).
*   **Physics Tuning**:
    *   `BASE_CAC`: Starting Cost of Acquisition ($50.0).
    *   `MAX_CHURN`: 30% Hard cap. `MIN_CHURN`: 2% Floor.
*   **Shock Tuning**:
    *   `INTEREST_RATE_VOLATILITY`: How much rates swing.
    *   `CONFIDENCE_VOLATILITY`: How much sentiment swings.

### Why separate Config?
*   Allows non-coders (Researchers) to tweak the simulation without touching logic.
*   Enables **A/B Testing** (Run w/ config A vs config B).

---

## 2. The Primer (`seed_dbs.py`)

This script **initializes the World Knowledge** for the Oracle Agent.

### What it does (Step-by-Step)
1.  **ChromaDB (Episodic Memory)**:
    *   Creates collection `episodes`.
    *   Injects **Synthetic History**: "In Q1, the company increased subscription price... -> Churn rose."
    *   This gives the Oracle *prior knowledge* before the simulation starts.
2.  **Neo4j (Causal Memory)**:
    *   Connects to Database.
    *   Injects **Causal Graph**: `(Price Increase) --[CAUSES]--> (Customer Churn)`.
    *   This teaches the Oracle basic economic theory.

### The Graph Schema
We inject nodes (`Concepts`) and edges (`Causes`).
*   **Nodes**: `Price Increase`, `Customer Churn`, `Revenue`.
*   **Edges**: `CAUSES`, `REDUCES`, `INCREASES`.
*   **Example Triple**: `(Marketing Spend) --[INCREASES]--> (User Acquisition)`.

### When to run?
*   **Once** before starting experiments.
*   `python seed_dbs.py`.

---

## 3. The Secrets (`.env`)

This file holds **Credentials** and **Paths**.

### Content
*   `NEO4J_URI`: `bolt://localhost:7687` (Default Neo4j port).
*   `NEO4J_PASSWORD`: Auth token for the graph database.
*   `CHROMA_PATH`: `./chroma_db` (Where to store vector data locally).
*   `OLLAMA_HOST`: `http://localhost:11434` (Where Llama 3 is running).

### Security
*   Never commit `.env` to Git. (Add to `.gitignore`).
*   Allows different setups for Local vs Cloud (e.g., swapping local Ollama for OpenAI optional).
