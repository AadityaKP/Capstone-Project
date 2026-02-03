# Startup/Market Simulator Implementation Plan

## Goal Description
Build a robust, Gym-style **State Transition Engine** for a startup market simulator. This simulator serves as the "World" for AI agents (CrewAI, Agno, etc.) to interact with. It allows agents to take strategic actions (marketing, hiring, pricing) and receive feedback (metrics, rewards) via a rigorous mathematical model incorporating stochastic market forces.

## User Review Required
> [!IMPORTANT]
> **Math & Logic Verification**: The transition equations (e.g., how exactly `marketing_spend` impacts `cac` and `growth`) need to be tuned. We will start with standard S-curves and linear decays but expect iteration.

> [!NOTE]
> **Time Step**: We are defaulting to **Weekly** time steps ($t$) as requested for granular simulation.

## Proposed Changes

### Core Environment (`env/`)

#### [NEW] [startup_env.py](file:///c:/College/Capstone/CapstoneProject/env/startup_env.py)
*   **Class**: `StartupEnv(gym.Env)`
*   **`__init__`**: Define action space (Dict) and observation space (Box). Initialize random seed.
*   **`reset()`**: Returns initial state $S_0$.
*   **`step(action)`**: Implements $S_t + A_t \to S_{t+1} + R_t$.
    *   Validates invariants (Cash \ge 0).
    *   Calculates derived metrics.
    *   Checks termination conditions (Bankruptcy, Time limit).

#### [NEW] [business_logic.py](file:///c:/College/Capstone/CapstoneProject/env/business_logic.py)
*   Encapsulate transition logic to keep Env clean.
*   **Functions**:
    *   `calculate_marketing_effect(spend, specific_channel_efficiency)`
    *   `calculate_churn(product_quality, pricing, competition_noise)`
    *   `calculate_burn(headcount, infrastructure_cost)`
    *   `apply_stochastic_shock(value, volatility)`

#### [NEW] [schemas.py](file:///c:/College/Capstone/CapstoneProject/env/schemas.py)
*   **Pydantic Models** for strict typing of State and Actions.
*   Ensures agents send valid JSON-serializable interactions.

### Agent Interface (`agents/`)

#### [NEW] [adapter.py](file:///c:/College/Capstone/CapstoneProject/agents/adapter.py)
*   **Translator Layer**: Converts agent string/JSON outputs into valid `env.step()` arguments.
*   Handling of "invalid" actions (e.g., spending more than available cash).

### Configuration (`config/`)

#### [NEW] [sim_config.py](file:///c:/College/Capstone/CapstoneProject/config/sim_config.py)
*   Centralized constants: `MAX_STEPS`, `INITIAL_CASH`, `MARKET_VOLATILITY`, `BASE_CAC`.

## Verification Plan

### Automated Tests
*   **Unit Tests**:
    *   `test_invariants`: Assert Cash never becomes negative without termination.
    *   `test_mechanics`: Assert Marketing spend > 0 increases Brand/Users (deterministically before noise).
    *   `test_gym_api`: Verify `check_env(StartupEnv())` passes Gymnasium compliance.
*   **Regression Tests**:
    *   Run a `RandomPolicy` for 100 episodes. Ensure 0 crashes.

### Manual Verification
*   **Sanity Check Plotting**:
    *   Run a simulation with a fixed policy (e.g., "Spend 10k/week").
    *   Plot `Revenue` vs `Time`.
    *   Visual check: "Does the startup die if burn > revenue?" "Does it grow if marketing is effective?"
