# Oracle Thesis Analysis Summary

## Primary Summary

| scenario_id   | Scenario           |   Survival % |   Avg Final MRR |   Median Final MRR |   Avg Rule-40 Post Shock (25-60) |   Mean Recovery Time (Mo) |   Median Recovery Time (Mo) |   Recovered % |   Decision Difference vs Boardroom % |   Avg LLM Calls |   Avg Cache Hits |
|:--------------|:-------------------|-------------:|----------------:|-------------------:|---------------------------------:|--------------------------:|----------------------------:|--------------:|-------------------------------------:|----------------:|-----------------:|
| boardroom     | Boardroom Baseline |      97.3333 |     1.38996e+06 |   703216           |                         -48.3217 |                   9.23684 |                           2 |       67.5556 |                             nan      |          0      |         0        |
| oracle_v1     | Oracle v1          |      98.6667 |     2.35012e+06 |        1.4743e+06  |                         -37.7506 |                   4.81871 |                           1 |       76      |                              98.5757 |         28.2133 |         2.68     |
| oracle_v3     | Oracle v3          |      98.6667 |     2.25158e+06 |        1.36086e+06 |                         -38.5946 |                   5.57225 |                           2 |       76.8889 |                              98.5646 |         29.84   |         0.986667 |

## Significance Tests

Pairwise Mann-Whitney U tests compare each Oracle policy against the boardroom baseline.

| metric                |    U |     p_value | significant   |   n_a |   n_b | method             | baseline_scenario_id   | comparison_scenario_id   | comparison_scenario_label   |
|:----------------------|-----:|------------:|:--------------|------:|------:|:-------------------|:-----------------------|:-------------------------|:----------------------------|
| post_shock_avg_rule40 | 1084 | 8.29883e-11 | True          |    75 |    75 | scipy_mannwhitneyu | boardroom              | oracle_v1                | Oracle v1                   |
| final_mrr             | 2143 | 0.0119171   | True          |    75 |    75 | scipy_mannwhitneyu | boardroom              | oracle_v1                | Oracle v1                   |
| post_shock_avg_rule40 | 1181 | 8.76115e-10 | True          |    75 |    75 | scipy_mannwhitneyu | boardroom              | oracle_v3                | Oracle v3                   |
| final_mrr             | 2195 | 0.0203877   | True          |    75 |    75 | scipy_mannwhitneyu | boardroom              | oracle_v3                | Oracle v3                   |
