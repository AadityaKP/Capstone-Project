# Startup Simulator Project Explainer

This document outlines the **5-Phase Development Process** used to build the Startup/Market Simulator. It explains the "Why" and "How" of each phase, providing a roadmap for understanding the codebase.

---

## Phase 1: The Skeleton (Environment Architecture)
**Goal:** Establish the mathematical and structural foundation before adding complexity.
**Key Components:** `env/startup_env.py`, `config/sim_config.py`

In this phase, we built the **Gymnasium Environment**. We defined:
*   **State Space**: Variables like Cash, Users, Revenue.
*   **Action Space**: The structure of inputs (initially placeholders).
*   **The Loop**: `reset()` $\to$ `step()` $\to$ `reward`.

**Why this comes first:** Without a robust state transition engine (the "World"), agents have nothing to interact with. We established the constraints (e.g., rigid time steps) early.

---

## Phase 2: Single Action Dynamics
**Goal:** Make the simulation respond realistically to basic inputs.
**Key Components:** `env/business_logic.py`

We implemented the core mathematical transitions for:
*   **Marketing**: Using an S-curve (Logarithmic Diminishing Returns) model. Spending money increases specific metrics (Users, Brand) but efficiency drops as spend scales.
*   **Pricing**: A price elasticity model. Raising prices increases revenue per user (ARPU) but exponentially increases Churn (users leaving).

**Why separate:** Debugging "why is churn high?" is impossible if you implement 5 actions at once. We verified Price $\to$ Churn causality in isolation.

---

## Phase 3: Multiple Actions Dynamics
**Goal:** Introduce trade-offs and resource management.
**Key Components:** `env/business_logic.py` (expanded)

We added:
*   **Hiring**: Increases internal capacity (Headcount) but permanently increases **Burn Rate**. This forces agents to balance growth vs. survival.
*   **Product Investment**: Spending R&D cash improves `Product Quality`, which acts as a shield against Churn.

This creates the central tension: *Do I spend on Marketing (growth), Product (retention), or save Cash (survival)?*

---

## Phase 4: Agent Integration & Adapter
**Goal:** Bridge the gap between "Messy LLM Outputs" and "Strict Mathematical Inputs".
**Key Components:** `agents/adapter.py`

LLM Agents (CrewAI, Agno) output text or JSON that might be malformed (e.g., `price: "free"` or `spend: -100`).
The **ActionAdapter**:
1.  **Sanitizes**: Clamps values (no negative spend).
2.  **Validates**: Ensures action types exist.
3.  **Translates**: Converts `{"hire": 5}` into the internal dict format the Env understands.

**Why this matters:** It prevents the simulation from crashing when an Agent makes a syntax error.

---

## Phase 5: Multi-Agent & Tuning (Sanity Check)
**Goal:** Verify system stability over long horizons.
**Key Components:** `simulation_runner.py`

We built a `RandomPolicy` to spam the environment with random actions for 100+ episodes (5 years each).
*   **Observed**: Does the startup go bankrupt? (Yes, if it overspends). Does it succeed? (Yes, if lucky).
*   **Clamped Invariants**: Fixed bugs where Cash could become negative.

**Result:** A verified, crash-resistant sandbox ready for intelligent agents.
