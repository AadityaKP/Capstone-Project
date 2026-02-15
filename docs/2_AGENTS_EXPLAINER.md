# Component Explainer: The Agent Layer

**Core Path**: `agents/`
**Key Files**: 
*   [`oracle_agent.py`](../agents/oracle_agent.py) - The Brain (LLM).
*   [`baseline_agents.py`](../agents/baseline_agents.py) - The Baseline.
*   [`adapter.py`](../agents/adapter.py) - The Filter.

---

## 1. The "Brain" (`oracle_agent.py`)

The Oracle is not just a chatbot; it is a **Retrieval-Augmented Causal Analyst**. It solves the "Hallucination Problem" by grounding answers in two databases.

### The RAG Pipeline (Step-by-Step)

When the simulation asks `analyze_situation(state)`:

1.  **Retrieve Episodic Context (ChromaDB)**:
    *   *Query*: "High Churn, Low Confidence"
    *   *Result*: "In Episode 42, churn spiked due to price hike."
    *   *Mechanism*: Cosine similarity search on vector embeddings.

2.  **Retrieve Causal Context (Neo4j)**:
    *   *Query*: `MATCH (n)-[r]->(m) WHERE n.name = 'Churn'`
    *   *Result*: `(Price) --[INCREASES]--> (Churn)`, `(Quality) --[DECREASES]--> (Churn)`
    *   *Mechanism*: Graph traversal.

3.  **Synthesize (LLM Prompting)**:
    We construct a structured prompt for **Llama 3**:
    ```text
    SYSTEM: You are a Startup Oracle.
    CONTEXT:
    - Past Episode: "Price hike caused churn."
    - Known Fact: "Quality reduces churn."
    STATE: "Churn is 5%, Price is $50, Quality is 0.5."
    TASK: Output a JSON action plan.
    ```

4.  **Prescriptive Output**:
    The LLM helps govern the board by outputting a **Prescriptive Brief**:
    *   *Insight*: "Churn is high because quality is low, despite valid pricing."
    *   *Directive*: "Increase R&D spend immediately."
    *   *Memory Update*: "Learned that low quality > high churn even if price is low."

---

## 2. The Baseline Agents (`baseline_agents.py`)

To benchmarks the Oracle, we use "Heuristic Agents" (Hard-coded Logic). They represent a standard, non-AI executive team.

### A. The CFO (Chief Financial Officer)
*   **Prime Directive**: Survival & Efficiency (Rule of 40).
*   **Logic**:
    *   *Runway Check*: `Cash / Burn`. If < 18 months -> **Hiring Freeze**.
    *   *Efficiency Check*: `LTV:CAC`. If < 3.0 -> **Raise Prices** (Try to boost LTV).
    *   *Growth check*: If `Runway > 24 months` -> Allow hiring.

### B. The CMO (Chief Marketing Officer)
*   **Prime Directive**: Growth (New MRR).
*   **Logic**:
    *   *Spend Sizing*:
        *   `LTV:CAC > 4`: Spend **$20k** (Aggressive).
        *   `LTV:CAC > 2`: Spend **$10k** (Moderate).
        *   `LTV:CAC < 2`: Spend **$2k** (Pull back).
    *   *Channel Strategy*:
        *   If `Consumer Confidence < 90`: Use **PPC** (Performance Marketing).
        *   Else: Use **Brand** (Long-term awareness).

### C. The CPO (Chief Product Officer)
*   **Prime Directive**: Retention (Churn Reduction).
*   **Logic**:
    *   *Crisis Mode*: If `Churn > 4%` -> Spend **$15k** on R&D.
    *   *Maintenance*: If `Churn < 2%` -> Spend **$3k**.
    *   *Budget Constraint*: If `Cash < $200k`, cut R&D by 50%.

---

## 3. The Translator (`adapter.py`)

The simulator is mathematically strict; LLMs are messy. The Adapter bridges this gap.

### Responsibilities
1.  **Sanitization**:
    *   *Input*: `{"spend": -500}` (Illegal negative spend).
    *   *Output*: `{"spend": 0}` (Clamped to zero).
2.  **Formatting**:
    *   *Input*: LLM might output Markdown or extra text.
    *   *Output*: Pure JSON matching the `ActionBundle` schema.
3.  **Defaulting**:
    *   If an agent crashes or returns nothing, the Adapter injects a "Do Nothing" action to keep the simulation alive.
