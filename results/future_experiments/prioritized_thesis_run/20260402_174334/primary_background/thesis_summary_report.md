# Oracle Thesis Analysis Summary

## Primary Summary

| scenario_id   | Scenario           |   Survival % |   Avg Final MRR |   Median Final MRR |   Avg Rule-40 Post Shock (25-60) |   Mean Recovery Time (Mo) |   Median Recovery Time (Mo) |   Recovered % |   Decision Difference vs Boardroom % |   Avg LLM Calls |   Avg Cache Hits |
|:--------------|:-------------------|-------------:|----------------:|-------------------:|---------------------------------:|--------------------------:|----------------------------:|--------------:|-------------------------------------:|----------------:|-----------------:|
| boardroom     | Boardroom Baseline |           98 |     1.4297e+06  |   905423           |                         -47.4976 |                   8.65385 |                           2 |       69.3333 |                             nan      |            0    |             0    |
| oracle_v1     | Oracle v1          |           98 |     2.41915e+06 |        1.66941e+06 |                         -37.3714 |                   5.72269 |                           2 |       79.3333 |                              98.5144 |           28.48 |             2.54 |
| oracle_v3     | Oracle v3          |           98 |     2.30814e+06 |        1.56159e+06 |                         -37.9785 |                   5.11404 |                           2 |       76      |                              98.8149 |           30.16 |             0.9  |

## Significance Tests

Pairwise Mann-Whitney U tests compare each Oracle policy against the boardroom baseline.

| metric                |   U |     p_value | significant   |   n_a |   n_b | method             | baseline_scenario_id   | comparison_scenario_id   | comparison_scenario_label   |
|:----------------------|----:|------------:|:--------------|------:|------:|:-------------------|:-----------------------|:-------------------------|:----------------------------|
| post_shock_avg_rule40 | 511 | 3.56019e-07 | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v1                | Oracle v1                   |
| final_mrr             | 969 | 0.053148    | False         |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v1                | Oracle v1                   |
| post_shock_avg_rule40 | 530 | 7.04557e-07 | True          |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v3                | Oracle v3                   |
| final_mrr             | 974 | 0.057532    | False         |    50 |    50 | scipy_mannwhitneyu | boardroom              | oracle_v3                | Oracle v3                   |
