# Validation package — reading guide

Self-contained copy of everything needed to understand the validation process and its
results, assembled 2026-08-30. **These are copies; the canonical originals live in
`validation/`, `report/`, and `validation/results/` and should be edited there, then
re-copied.** Relative links inside the copied documents point at the original repo
locations.

## Read in numbered order

| File | What it gives you |
|---|---|
| `01_validation_report.md` | **The results.** Verdicts (Simulator: C, Agents: B), every executed test with numbers, Tier-2 section (§5b), paper-ready claim language, limitations. |
| `02_validation_plan.md` | **The process.** Critique of the prior de facto plan, the restructured methodology, acceptance criteria fixed before results, Tier 1/2/3 prioritization, fallbacks. |
| `03_system_audit.md` | What the simulator and agents actually are: pipeline, variable inventory, action tracing, RNG regimes, leakage audit, documented-vs-implemented mismatches. |
| `04_agent_audit.md` | Every policy, its decision mechanism, and its testable predictions. |
| `05_edgar_data_audit.md` | What the 39-company EDGAR panel provides, field-by-field mapping (Direct / Proxy / Unavailable), temporal and scale compatibility rules. |
| `06_reproduction_README.md` | Package map and the exact commands to reproduce every result. |

## Supporting data

- `results/agent_scorecard.csv` — 24 agent-validation rows (dimension, test, baseline, n, effect, criterion, verdict, interpretation).
- `results/environment_scorecard.csv` — 18 environment rows incl. the E6 drawdown comparison.
- `results/claim_audit.csv` — 22 previously recorded claims re-verified against raw files.
- `results/statistical_tests.csv` — consolidated paired tests (A2 baselines + A3 oracle value).
- `results/real_company_backtest.csv` — per-company C1 backtest with the retrodiction/increment decomposition.
- `results/validation_summary.csv` — one row per test across both scorecards.
- `results/provisional_placeholders.csv` — integrity ledger: confirms every analysis is EXECUTED and no synthetic number exists anywhere in the report.
- `figures/` — the four headline figures (growth distribution vs EDGAR, policy baselines, action ladders, backtest retrodiction).

## One-paragraph summary

The simulator matches real SaaS on growth-rate distribution and growth-deceleration
at its calibration scale, but is too persistent and too volatile at once, under-spends
relative to real cost structures, produces catastrophic non-recovering drawdowns where
real ones are shallow and quickly recovered, and over-projects real-company growth by
~50pp/4 quarters — verdict: credible *comparative testbed*, not a forecasting model.
The agents' actions causally move outcomes; the boardroom beats noop/random/heuristic
at matched seeds (g 0.60–0.84, stable across 9/9 initial-condition cells); the oracle
layer beats the boardroom in 74–75/75 recorded and 20/20 replicated seeds (g ≈ 0.95),
with episodic retrieval contributing a real but small ≈3% of that gain; the LLM channel
reacts to trends and shocks, not state levels; and no agent-value claim survives at
real-company scale — verdict: evidence of policy value with important limitations.
