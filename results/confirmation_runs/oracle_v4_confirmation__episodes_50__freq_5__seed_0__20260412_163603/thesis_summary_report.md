# Oracle v4 Confirmation Summary

## Run Configuration

- Output folder: `results\confirmation_runs\oracle_v4_confirmation__episodes_50__freq_5__seed_0__20260412_163603`
- Episodes per policy: 50
- Seed start: 0
- Oracle frequency: 5
- Policies: boardroom, oracle_v1, oracle_v3, oracle_v4, oracle_v4_causal
- Retrieval trace export: oracle_v3 and oracle_v4 only

## Output Files

- `primary_summary.csv`
- `primary_episode_metrics.csv`
- `primary_retrieval_trace.csv`
- `thesis_summary_report.md`

## Primary Summary

| scenario_id      | Scenario           |   Episodes |   Survival % |   Avg Final MRR |   Median Final MRR |   Avg Rule-40 Post Shock (25-60) |   Mean Recovery Time (Mo) |   Median Recovery Time (Mo) |   Recovered Shock Rate % |   Avg LLM Calls |   Avg Cache Hits |
|:-----------------|:-------------------|-----------:|-------------:|----------------:|-------------------:|---------------------------------:|--------------------------:|----------------------------:|-------------------------:|----------------:|-----------------:|
| boardroom        | Boardroom Baseline |         50 |           98 |     1.4297e+06  |   905423           |                         -47.4976 |                   7.28261 |                     3.16667 |                  69.3333 |            0    |             0    |
| oracle_v1        | Oracle v1          |         50 |          100 |     2.49403e+06 |        1.73496e+06 |                         -36.7058 |                   4.93878 |                     3       |                  78.6667 |           35.7  |             3.62 |
| oracle_v3        | Oracle v3          |         50 |          100 |     2.50777e+06 |        1.73564e+06 |                         -36.7796 |                   5.2483  |                     3.33333 |                  80      |           35.62 |             3.76 |
| oracle_v4        | Oracle v4          |         50 |          100 |     2.50774e+06 |        1.73564e+06 |                         -36.7796 |                   5.2483  |                     3.33333 |                  80      |           35.62 |             3.76 |
| oracle_v4_causal | Oracle v4 Causal   |         50 |          100 |     2.49977e+06 |        1.73793e+06 |                         -36.6679 |                   4.60884 |                     2       |                  80      |           35.68 |             3.7  |

## Significance Tests

Pairwise Mann-Whitney U tests compare each policy against the boardroom baseline.

| metric                      |      U |     p_value | significant   |   n_a |   n_b | method             | baseline_scenario_id   | comparison_scenario_id   | comparison_scenario_label   |
|:----------------------------|-------:|------------:|:--------------|------:|------:|:-------------------|:-----------------------|:-------------------------|:----------------------------|
| post_shock_avg_rule40_25_60 |  454   | 4.15735e-08 | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v1                | Oracle v1                   |
| mean_recovery_time_months   | 1196   | 0.608703    | False         |    46 |    49 | scipy_mannwhitneyu | boardroom              | oracle_v1                | Oracle v1                   |
| final_mrr                   |  918   | 0.0222952   | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v1                | Oracle v1                   |
| post_shock_avg_rule40_25_60 |  457   | 4.67226e-08 | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v3                | Oracle v3                   |
| mean_recovery_time_months   | 1158.5 | 0.816937    | False         |    46 |    49 | scipy_mannwhitneyu | boardroom              | oracle_v3                | Oracle v3                   |
| final_mrr                   |  939   | 0.0323119   | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v3                | Oracle v3                   |
| post_shock_avg_rule40_25_60 |  457   | 4.67226e-08 | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v4                | Oracle v4                   |
| mean_recovery_time_months   | 1158.5 | 0.816937    | False         |    46 |    49 | scipy_mannwhitneyu | boardroom              | oracle_v4                | Oracle v4                   |
| final_mrr                   |  939   | 0.0323119   | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v4                | Oracle v4                   |
| post_shock_avg_rule40_25_60 |  458   | 4.85725e-08 | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v4_causal         | Oracle v4 Causal            |
| mean_recovery_time_months   | 1305.5 | 0.182885    | False         |    46 |    49 | scipy_mannwhitneyu | boardroom              | oracle_v4_causal         | Oracle v4 Causal            |
| final_mrr                   |  935   | 0.0301506   | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v4_causal         | Oracle v4 Causal            |
