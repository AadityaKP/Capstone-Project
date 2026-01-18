# Project Plan: Multi-Agent Business Simulation

This document serves as the roadmap for the Capstone Project, focusing on building a "Sandbox" business simulation governed by a "Boardroom" of Multi-Agent Reinforcement Learning (MARL) agents.

## 1. Project Overview
- **Goal**: Develop a Multi-Agent System (MAS) where agents (CFO, CPO, CMO) collaborate to optimize business KPIs (Churn, LTV, CAC) within a simulated environment.
- **Key Concepts**: Business Physics, Non-Stationarity (Market Shocks), Hierarchical RL.
- **Tech Stack**:
    - **Simulation**: PettingZoo (ParallelEnv), Gymnasium
    - **ML Framework**: PyTorch
    - **Logic**: NumPy, Scikit-learn (Curves)
    - **Visualization**: Matplotlib

## 2. Phase 1: The Sandbox & Boardroom (Core Simulation)

This phase focuses on establishing the physical laws of the business environment and the agent interfaces.

### Step 1: Repository Structure & Environment Setup
- [ ] **Repository Init**: Set up clean structure separating simulation (`env/`) and agents (`agents/`).
- [ ] **Dependencies**: Create `requirements.txt` with `pettingzoo`, `gymnasium`, `numpy`, `torch`, `matplotlib`.

### Step 2: Implement "Business Physics" (`business_logic.py`)
- [ ] **Diminishing Returns**: Math functions ensuring ad spend efficiency drops as budget scales.
- [ ] **Technical Debt**: Logic where `new_features` focus increments `bug_count`, driving up `churn_rate`.

### Step 3: Build PettingZoo Custom Environment (`startup_env.py`)
- [ ] **API**: Use `ParallelEnv` for simultaneous C-Suite decision making.
- [ ] **Agents**: Define `self.agents = ["cfo", "cpo", "cmo"]`.
- [ ] **Observation Spaces**: Continuous values (`Box`) for Cash, Churn, User Count, etc.
- [ ] **Action Spaces**:
    - **CFO**: Continuous budget allocation.
    - **CMO/CPO**: Discrete or Box choices for tactical focus.

### Step 4: Reward Functions
- [ ] **Global Reward**: Net profit or company survival duration.
- [ ] **Local Rewards**:
    - **CMO**: New Users minus CAC Penalty.
    - **CPO**: Retention Rate minus Tech Debt Penalty.

### Step 5: Market Shocks (Non-Stationarity)
- [ ] **Triggers**: Implement random events in `step()`:
    - **Competitor Launch**: Increases `base_churn_rate` (e.g., +15%).
    - **Ad Spike**: Doubles `CAC`.
- [ ] **Logging**: Record shock events for Oracle visibility.

### Step 6: Baseline Agents (`agents/baselines.py`)
- [ ] **Fragmented Baselines**: Independent PPO agents (blind to each other).
- [ ] **Centralized Baseline**: Single agent controlling the entire action vector.

### Step 7: Initial Simulation & Archival
- [ ] **Self-Play Loop**: Run 1,000 steps with random/heuristic policies.
- [ ] **Verification**: Plot Cash/Users over time to validate "Business Physics".
- [ ] **Data Logging**: Save `(state, action, reward, next_state)` to `logs/`.

## 3. Future Phases (TBD)
- **Phase 2**: The Oracle & Event Logging
- **Phase 3**: Advanced Agent Architectures
- **Phase 4**: Analysis & Visualization
