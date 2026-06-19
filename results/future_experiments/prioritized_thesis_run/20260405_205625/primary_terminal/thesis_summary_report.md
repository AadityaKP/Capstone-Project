# Oracle Thesis Analysis Summary

## Primary Summary

| scenario_id   | Scenario           |   Survival % |   Avg Final MRR |   Median Final MRR |   Avg Rule-40 Post Shock (25-60) |   Mean Recovery Time (Mo) |   Median Recovery Time (Mo) |   Recovered % |   Decision Difference vs Boardroom % |   Avg LLM Calls |   Avg Cache Hits |
|:--------------|:-------------------|-------------:|----------------:|-------------------:|---------------------------------:|--------------------------:|----------------------------:|--------------:|-------------------------------------:|----------------:|-----------------:|
| boardroom     | Boardroom Baseline |          100 |         15640.3 |            15640.3 |                         -47.2444 |                         3 |                           3 |       33.3333 |                             nan      |               0 |                0 |
| oracle_v1     | Oracle v1          |          100 |         17069.8 |            17069.8 |                         -31.0001 |                         3 |                           3 |       33.3333 |                              90      |              36 |                4 |
| oracle_v3     | Oracle v3          |          100 |         16667.6 |            16667.6 |                         -37.3963 |                         3 |                           3 |       33.3333 |                              90.8333 |              38 |                1 |

## Significance Tests

Pairwise Mann-Whitney U tests compare each Oracle policy against the boardroom baseline.

| metric                |   U |   p_value | significant   |   n_a |   n_b | method             | baseline_scenario_id   | comparison_scenario_id   | comparison_scenario_label   |
|:----------------------|----:|----------:|:--------------|------:|------:|:-------------------|:-----------------------|:-------------------------|:----------------------------|
| post_shock_avg_rule40 |   0 |         1 | False         |     1 |     1 | scipy_mannwhitneyu | boardroom              | oracle_v1                | Oracle v1                   |
| final_mrr             |   0 |         1 | False         |     1 |     1 | scipy_mannwhitneyu | boardroom              | oracle_v1                | Oracle v1                   |
| post_shock_avg_rule40 |   0 |         1 | False         |     1 |     1 | scipy_mannwhitneyu | boardroom              | oracle_v3                | Oracle v3                   |
| final_mrr             |   0 |         1 | False         |     1 |     1 | scipy_mannwhitneyu | boardroom              | oracle_v3                | Oracle v3                   |
