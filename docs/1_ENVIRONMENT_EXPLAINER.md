# Component Explainer: The Environment Layer

**Core Path**: `env/`
**Key Files**: 
*   [`startup_env.py`](../env/startup_env.py) - The Gym Environment.
*   [`business_logic.py`](../env/business_logic.py) - The pure physics functions.
*   [`schemas.py`](../env/schemas.py) - Data structures (Pydantic).

---

## 1. System Overview

The Environment Layer is the "Physical World" of the simulation. It is designed to be **Agent-Agnostic**, meaning it doesn't care if the actor is a random number generator, a heuristic script, or an LLM. It simply receives an **Action** and returns a **State Update** based on strict mathematical rules.

### The Simulation Loop
Using the standard OpenAI Gym (Gymnasium) interface:

1.  **Incoming Action**: A dictionary of decisions (Pricing, Hiring, Marketing).
2.  **Exogenous Shocks**: The world changes (Interest Rates, Consumer Confidence).
3.  **Endogenous Cascades**: The system checks for systemic failures (Recessions).
4.  **Business Physics**: Actions are converted into outcomes (Leads, Churn, Revenue).
5.  **State Transition**: The `EnvState` is updated for $t+1$.
6.  **Reward Calculation**: A score is generated based on the "Rule of 40".

---

## 2. State Space (`EnvState`)

The state is a strictly typed **Pydantic Model** encompassing 16 variables.

| Category | Variable | Type | Description |
| :--- | :--- | :--- | :--- |
| **Financials** | `mrr` | Float | Monthly Recurring Revenue. The primary growth metric. |
| | `cash` | Float | Liquid capital. If $<0$, the simulation terminates (Bankruptcy). |
| | `valuation_multiple` | Float | Current revenue multiple (e.g., 10x). Driven by Interest Rates. |
| **Unit Economics** | `cac` | Float | Cost to Acquire a Customer. Affected by Marketing efficiency & Competition. |
| | `ltv` | Float | Lifetime Value. `ARPU / Churn Rate`. |
| **Operations** | `headcount` | Int | Number of employees. Drives burn rate. |
| | `product_quality` | Float | 0.0-1.0. Affects Churn. improved by R&D. |
| **Macro** | `interest_rate` | Float | Base cost of capital. Triggers valuation compression. |
| | `consumer_confidence` | Float | 0-200. Affects demand conversion rates. |
| | `unemployment` | Float | % of labor force. High unemployment triggers demand spirals. |
| **Shock State** | `active_shocks` | List | (Implicit) Boolean flags for ongoing crises. |
| | `innovation_factor` | Float | 1.0 = Normal. <1.0 = Scared/Damaged R&D capability (Hysteresis). |

---

## 3. Action Space (`ActionBundle`)

Agents must output a specific dictionary structure. The `ActionAdapter` ensures these values are clamped to realistic ranges before reaching the pysics engine.

```json
{
  "marketing": {
    "spend": 50000.0,       // $ Amount
    "channel": "ppc"        // "ppc" (fast, expensive) or "brand" (slow, sticky)
  },
  "product": {
    "r_and_d_spend": 10000.0 // Investment in quality
  },
  "hiring": {
    "hires": 2,             // Headcount addition
    "cost_per_employee": 10000.0 // Salary baseline
  },
  "pricing": {
    "price_change_pct": 0.05 // +5% Price increase
  }
}
```

---

## 4. The Laws of Physics (`business_logic.py`)

This module contains the transition functions $f(S_t, A_t) \rightarrow S_{t+1}$.

### A. Marketing Physics (The S-Curve)
We model marketing returns using a **Hill Function** to enforce diminishing returns. You cannot simply spend infinite money to get infinite users.

$$ NewUsers = \beta \frac{x^\alpha}{\gamma^\alpha + x^\alpha} $$

*   $\beta$ (Beta): Max potential market capacity.
*   $\gamma$ (Gamma): Half-saturation point (how much spend to get 50% of market).
*   $\alpha$ (Alpha): Steepness of the curve.

### B. Churn Physics
Churn is not random; it is a function of dissatisfaction and macro pressure.

$$ Churn = Base \times (1 - Quality/2) \times MacroMultiplier \times TenureDecay $$

*   **Quality**: High R&D spend reduces churn.
*   **Macro**: Low Consumer Confidence increases churn (especially for SMBs).
*   **Tenure**: Older companies (implied older cohorts) have lower churn (Stickiness).

### C. The Shock Engine (Tier 1 & 2)
The environment isn't static. It includes a sophisticated crisis engine:

1.  **Exogenous (External)**:
    *   *Interest Rate Spike*: `Rate += 1.5%`. Results in `Valuation *= 0.85`.
    *   *Confidence Crash*: `Confidence -= 20`. Results in `Unemployment += 1%`.
2.  **Endogenous (Systemic Feedback)**:
    *   *Recession Cascade*: IF `Unemployment > 8%` AND `Rates > 7%` -> Triggers a self-reinforcing downward spiral in Confidence.
    *   *Hysteresis (Scarring)*: IF `Depression > 6 months` -> `Innovation Factor` permanently decays (Brain Drain).
3.  **Recovery**:
    *   Variables mean-revert slowly over years, mimicking business cycles.

---

## 5. Reward Function (The Goal)

The "Score" isn't just Revenue. It's **Sustainable Growth**. We use the **Rule of 40**:

$$ Score = (Growth\% + Margin\%) $$

*   **Penalty**: If `Rule of 40 < 0` (Burning cash with no growth), massive reward penalty.
*   **Penalty**: If `Cash <= 0` (Bankruptcy), game over (-20 reward).
*   **Penalty**: If `Innovation Factor` decays (Long term damage), penalty applied.
