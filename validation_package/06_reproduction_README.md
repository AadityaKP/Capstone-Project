# Validation package

Built 2026-08-30 against branch `review2-sim-frontend`. Everything here traces to
raw data on disk; **no synthetic numbers appear anywhere in the results or the
report** (see `results/provisional_placeholders.csv` — rows mark pending/skipped
computations, none carries a synthetic value).

## Read in this order

1. [system_audit.md](system_audit.md) — what the simulator/agents actually are, action tracing, leakage audit
2. [agent_audit.md](agent_audit.md) — every policy, its mechanism, testable predictions
3. [edgar_data_audit.md](edgar_data_audit.md) — what the on-disk EDGAR panel supports
4. [validation_plan.md](validation_plan.md) — critique of the prior plan, restructured plan, pre-declared acceptance criteria, tiering
5. [../report/validation_report.md](../report/validation_report.md) — results and verdicts

## Layout

```
validation/analysis/       executable analyses (each writes into validation/results/)
validation/agents/         decision_log.csv, action_effects*, rule/LLM responsiveness
validation/real_company_backtest/  backtest.py + mapped_states.csv
validation/results/        scorecards, tests, claim audit, backtest results
validation/figures/        four Tier-1 figures
report/validation_report.md
```

EDGAR data itself lives where the repo already keeps it: `data/edgar.db`,
`data/edgar_ratios.csv`, `data/coverage_report.md`, `data/panel_extract.md`.

## Reproduction (from repo root, venv active; deterministic unless noted)

```
python validation/analysis/claim_audit.py            # recomputes recorded headline numbers
python validation/analysis/environment_battery.py    # E1-E5, E7 vs EDGAR panel
python validation/analysis/policy_baselines.py       # A2: noop/random/heuristic/boardroom, 50 matched seeds
python validation/analysis/action_effects.py         # A1: action ladders
python validation/analysis/verify_shared_world.py    # legacy shared-world property
python validation/analysis/a3_recorded_paired.py     # paired re-analysis of the recorded FULL run
python validation/analysis/a6_brief_accuracy.py      # brief expected_outcome vs realized (recorded)
python validation/analysis/a4_state_responsiveness.py  # needs Ollama + llama3.1:8b
python validation/analysis/a3_oracle_value_run.py    # needs Ollama; hours (LLM arms, 20 seeds x 4 arms)
python validation/analysis/a3_oracle_value_analyze.py  # after the run finishes
python validation/real_company_backtest/backtest.py  # C1 EDGAR backtest
python validation/analysis/a7_robustness_grid.py     # Tier-2: 3x3 initial-condition grid
python validation/analysis/a5_candidate_regret.py    # Tier-2: one-step candidate regret
python validation/analysis/e6_drawdown_recovery.py   # Tier-2: drawdowns, sim vs EDGAR
python validation/analysis/c2_allocation_consistency.py  # Tier-2: observational direction check
python validation/analysis/a8_shock_recovery.py      # A8: post-shock R40 recovery CSVs (feeds F7)
python validation/analysis/build_scorecards.py       # assembles scorecards + summary
python validation/analysis/make_figures.py
python validation/analysis/figures_review.py         # review figure set F1-F11 -> figures/review/
                                                     # (writes f4_growth_vs_scale.csv,
                                                     #  f6_paired_diffs.csv; run a8 first)
```

LLM analyses used llama3.1:8b via local Ollama at temperature 0. A4/A3 outputs
depend on that exact model; everything else is fully deterministic.
