# Component Explainer: Simulation Runner & Data

**Core Path**: `simulation_runner.py`
**Key Outputs**: 
*   [`simulation_results.csv`](../simulation_results.csv) - Raw Time Series.
*   [`baseline_metrics.csv`](../baseline_metrics.csv) - Comparative Analysis.

---

## 1. The Experiment Engine (`run_simulation()`)

The Runner is the "Scientific Laboratory". It conducts controlled experiments by running 100+ simulations under different policies (Random vs Heuristic vs Oracle).

### The Episode Loop
For each episode (Startup Lifecycle):

1.  **Initialization**: A fresh `StartupEnv` is created.
2.  **Reset**: The company starts with Seed Funding ($1M).
3.  **Step Loop** (Monthly):
    *   **Observation**: Agent sees `mrr`, `cash`, `valuation`.
    *   **Action**: Agent decides (e.g., `spend=$50k`).
    *   **Transition**: Physics engine updates state.
    *   **Logging**: Every variable is recorded.
4.  **Termination**:
    *   **Success**: Reaches 10 Years (120 Months).
    *   **Failure**: Cash < 0 (Bankruptcy).

### Headless Execution
The runner is designed to be **Headless** (CLI-based). This allows:
*   **Parallel Execution**: Running 1000s of simulations on a server.
*   **Reproducibility**: Fixed Random Seeds (`seed=42`) ensure every run is identical if repeated.

---

## 2. Comparative Metrics (`baseline_metrics.csv`)

We generate a report comparing policies. Here are the key definitions:

### A. Growth Metrics
*   **Mean MRR**: Average Monthly Revenue at the end of the simulation.
*   **% > $1M ARR**: Probability of reaching $83k MRR (Unicorn potential).
*   **Avg Valuation**: The final valuation multiple (e.g., 10x Revenue).

### B. Efficiency Metrics
*   **Avg Rule of 40**: The "Sustainable Growth" score.
    *   `Growth % + Profit Margin %`.
    *   Target > 40.
*   **Median LTV:CAC**: The "Unit Economics" score.
    *   `Lifetime Value / Acquisition Cost`.
    *   Target > 3.0.

### C. Resilience Metrics (Shock Engine)
These measure how well the agent survived the crisis:
*   **Survival Rate**: % of startups that didn't go bankrupt.
*   **Avg Innovation Factor**: Did the agent maintain R&D during the crash? (1.0 = Yes, <0.9 = Scarred).
*   **Avg Depression Months**: How long did the startup stay in "Low Confidence" mode?

---

## 3. Raw Data Logs (`simulation_results.csv`)

Every single month is logged for debugging.

| Column | Description |
| :--- | :--- |
| `episode` | ID of the run (0-100). |
| `step` | Month number (0-120). |
| `mrr` | Revenue. |
| `cash` | Bank balance. |
| `final_innovation_factor` | Health of R&D capability (1.0 = Normal). |
| `final_valuation_multiple` | Final Revenue Multiple (e.g., 10x). |
| `final_unemployment` | Final Macro Unemployment Rate (%). |
| `avg_rule_40` | Average Rule of 40 score over the episode. |
| `pct_above_40` | % of months where Rule of 40 was >= 40. |
| `cause` | "Time Limit" (Success) or "Bankruptcy" (Failure). |

### Usage
This CSV is consumed by:
1.  **Data Scientists**: To plot "Learning Curves".
2.  **The Oracle**: To learn from past failures ("In Episode 12, we died because...").
