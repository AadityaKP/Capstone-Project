# Startup Simulator: Technical Documentation

This folder contains detailed technical explainer documents for the Startup Simulator. These docs are designed to be used as a reference during code reviews or when onboarding new developers.

## 📚 Component Breakdown

### 1. [The Environment Layer](./1_ENVIRONMENT_EXPLAINER.md)
**"The Physics Engine"**
*   **What it covers**: The mathematical rules governing the simulation (Marketing S-Curves, Churn Physics), the strict `EnvState` Pydantic schema, and the Tier-1/Tier-2 Shock Engine.
*   **Key Files**: `env/startup_env.py`, `env/business_logic.py`.
*   **Read this to understand**: How the "World" works and how Shocks propagate.

### 2. [The Agent Layer](./2_AGENTS_EXPLAINER.md)
**"The Decision Makers"**
*   **What it covers**: The Heuristic Baseline Agents (CFO/CMO/CPO logic) and the **Oracle Agent's RAG Pipeline** (Episodic + Causal Memory).
*   **Key Files**: `agents/oracle_agent.py`, `agents/baseline_agents.py`.
*   **Read this to understand**: How the AI makes decisions and uses memory.

### 3. [The Runner & Data Layer](./3_RUNNER_EXPLAINER.md)
**"The Experiment Laboratory"**
*   **What it covers**: The simulation loop, headless execution, and the exact definitions of all Metrics (Rule of 40, LTV:CAC, Innovation Factor).
*   **Key Files**: `simulation_runner.py`.
*   **Read this to understand**: How experiments are run and how data is logged.

### 4. [Setup & Configuration](./4_SETUP_AND_CONFIG_EXPLAINER.md)
**"The Control Panel"**
*   **What it covers**: Global constants (`sim_config.py`), Database Seeding scripts (`seed_dbs.py`), and Environment Variables.
*   **Key Files**: `config/`, `.env`.
*   **Read this to understand**: How to tune the simulation and set up the infrastructure.

---

## 🚀 How to Use These Docs
*   **For Architecture Review**: Start with **Agents** and **Environment**.
*   **For Data Analysis**: Read **Runner & Data** to understand the CSV outputs.
*   **For Installation**: Read **Setup & Configuration**.
