# Evidence Audit — SaaS Startup Simulation Dataset Paper

Audit date: 2026-07-20. Branch: `startup-multi`. Read-only audit: every value below was parsed from an existing file (pandas / sqlite read-only / JSON). Nothing was re-run. Values that could not be found in any file are marked **MISSING** — they were not estimated and not filled from code defaults.

**Run classification used throughout** (basis stated per run):

| Run tag | Location | Basis for classification |
|---|---|---|
| **FULL** (primary thesis run) | `results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background/` | `primary_background_metadata.json`: `num_episodes: 75`, `seed_start: 0`, `oracle_frequency: 10`, started 2026-04-04T00:25:45; files written 2026-04-12 12:12 (long background run) |
| **PILOT** (smoke run) | `results/future_experiments/prioritized_thesis_run/20260405_205625/primary_terminal/` | `primary_terminal_metadata.json`: `num_episodes: 1`, completed in 5 min (20:56→21:01 on 2026-04-05) |
| **CONFIRMATION** | `results/confirmation_runs/oracle_v4_confirmation__episodes_50__freq_5__seed_0__20260412_163603/` | Folder name encodes episodes=50, freq=5, seed=0; `thesis_summary_report.md` inside states the same; 5 policies |
| **V4-DEV** (development comparisons) | `outputs/oracle_v4_compare*/` | Small ad-hoc JSON comparisons (2–20 episodes) produced by local `run_compare.py` scripts during oracle_v4 debugging, June 2026 |

---

## 1. File Inventory

Sizes in bytes; modified dates local time (+05:30). Descriptions from headers/first rows only.

### Primary thesis run — FULL (`results/future_experiments/prioritized_thesis_run/20260404_002545/primary_background/`)

| File | Size | Modified | Contents |
|---|---:|---|---|
| `primary_background_metadata.json` | 852 | 2026-04-12 | Run config: num_episodes=75, oracle_frequency=10, seed_start=0, started 2026-04-04T00:25:45, spawn command line |
| `primary_episode_metrics.csv` | 56,911 | 2026-04-12 | 225 rows; per-episode: episode, seed, policy, steps, final_mrr/cash/cac/ltv, cause, total_reward, avg_rule_40, oracle refresh/cache/llm counters |
| `primary_episode_metric_summary.csv` | 61,277 | 2026-04-12 | Same 225 rows + `post_shock_avg_rule40` column |
| `primary_monthly_trace.csv` | 4,616,057 | 2026-04-12 | 26,984 rows; per (policy, episode, month): reward, rule_of_40, shock_label, terminated/truncated, mrr, cash, innovation, unemployment, brief fields |
| `primary_action_trace.csv` | 5,899,602 | 2026-04-12 | Per-month action records: refresh_reason, brief fields, pre/post-modifier marketing/R&D/hires |
| `primary_recovery_events.csv` | 93,233 | 2026-04-12 | 675 rows; per (policy, episode, shock): shock_month, shock_label, pre_shock_rule_40, recovery_month, recovery_time_months, recovered |
| `primary_retrieval_trace.csv` | 8,113,801 | 2026-04-12 | 26,970 rows (oracle_v3 only); retrieved memory documents with source_month, realized_outcome, outcome_bucket, memory_weight, similarity, recency |
| `primary_reward_curve.csv` | 55,356 | 2026-04-12 | Per (policy, month) mean/std/CI95 of reward and rule-40, n column |
| `primary_significance_tests.csv` | 536 | 2026-04-12 | 4 Mann-Whitney U tests vs boardroom (post_shock_avg_rule40, final_mrr × oracle_v1/v3) |
| `primary_summary.csv` | 717 | 2026-04-12 | 3-row headline table (survival %, MRR, recovery, decision-difference, LLM calls) |
| `decision_difference_detail.csv` | 5,553,604 | 2026-04-12 | Per-month oracle-vs-boardroom action deltas |
| `decision_difference_summary.csv` | 303 | 2026-04-12 | Decision difference rate per oracle policy |
| `oracle_decision_map.csv` | 1,575,531 | 2026-04-12 | Per-month brief risk level / confidence / spend change |
| `retrieval_quality.csv` | 9,143 | 2026-04-12 | Per (episode, outcome_bucket) retrieval counts and shares (oracle_v3) |
| `thesis_summary_report.md` | 2,670 | 2026-04-12 | Markdown rendering of summary + significance tables |
| `primary_background.log` | 802,677 | 2026-04-12 | Run log |
| `plot_1/2/4/5 …png` | 35–153 KB | 2026-04-12 | Reward-over-time, recovery histogram, retrieval quality, decision map plots |
| `../prioritized_run_summary.json` | 1,255 | 2026-04-12 | Copy of metadata + paths |

### Pilot smoke run — PILOT (`.../20260405_205625/primary_terminal/`)

Same 15-file layout as FULL, all dated 2026-04-05 21:01. Episode metrics = 3 rows (1 per policy, seed 0 only); monthly trace = 360 rows; retrieval trace = 330 rows; plus `primary_terminal_metadata.json` (296 B, num_episodes=1) and `../prioritized_run_summary.json` (679 B).

### Confirmation run — CONFIRMATION (`results/confirmation_runs/oracle_v4_confirmation__episodes_50__freq_5__seed_0__20260412_163603/`)

| File | Size | Modified | Contents |
|---|---:|---|---|
| `primary_episode_metrics.csv` | 76,297 | 2026-04-13 | 250 rows; 5 policies × 50 episodes; includes per-episode shock_count, recovered_shock_rate_pct, recovery-time columns |
| `primary_retrieval_trace.csv` | 12,787,683 | 2026-04-13 | 35,930 rows (oracle_v3 + oracle_v4), includes refresh_reason |
| `primary_summary.csv` | 915 | 2026-04-13 | 5-row headline table |
| `primary_summary_screenshot_no_oracle_v3.csv` / `.md` | 770 / 1,481 | 2026-04-21 | **Curated copies of the summary with the oracle_v3 row removed** ("screenshot" variants, made 8 days after the run) |
| `thesis_summary_report.md` | 5,193 | 2026-04-13 | Config (episodes 50, freq 5, seed 0, 5 policies) + summary + 12 significance tests |

`results/confirmation_runs/oracle_v4_confirmation__episodes_50__freq_5__seed_0__20260413_062028/` — **EMPTY directory** (created 2026-04-13 06:20; a started/aborted second confirmation run; no data).

### Oracle v4 development comparisons — V4-DEV (`outputs/`)

| File | Size | Modified | Contents |
|---|---:|---|---|
| `oracle_v4_compare/comparison.json` (+`compare.log`, `run_compare.py`) | 2,488 | 2026-06-13 | 2 episodes: oracle_v3 vs oracle_v4_causal_hetero (v4 died at 15.5 avg steps, both Bankruptcy) |
| `oracle_v4_compare_after_fix/comparison.json` | 2,969 | 2026-06-14 | Same pair, 2 episodes, after a fix (v4 avg_duration 67, still 0% survival) |
| `oracle_v4_compare_after_predicate_fix/comparison.json` (+`monthly_extract.json`, logs) | 1,624 / 53,197 | 2026-06-18 | v4_causal_hetero only, 2 episodes, survival 0.5; causal_confidence 0.551 over 696 obs |
| `oracle_v4_compare_final/comparison.json` (+3 logs, `run_compare.py`) | 8,045 | 2026-06-18/19 | oracle_v3 only, 20 episodes (seeds 0–19), survival 1.0 |
| `oracle_v4_compare_final_v2/comparison.json` | 16,468 | 2026-06-27 | oracle_v3 + oracle_v4_causal_hetero, 20 episodes each (seeds 0–19). **Modified in working tree vs last commit** (see §3) |
| `oracle_v4_compare_final_v2/monthly_extract.json` | 516,998 | 2026-06-27 | 2,249 records: per (episode, month) causal_stress_node, stress_persistence_months, CFO/CMO/CPO actions |
| `oracle_v4_compare_final_v2/run_compare.py` (+stdout/stderr logs) | 7,857 | 2026-06-19 | Runner: num_episodes=20, seed_start=0, oracle_frequency=5 (lines 111–113) |
| `oracle_v4_debug/seed0_trace.json`, `seed0_trace_prefix.json`, `seed0_after_fix_until_month30.json` (+2 scripts) | 36,802 / 36,802 / 79,682 | 2026-06-14 | Single-seed debug traces |
| `oracle_v4_debug_gate/seed0_latest.json` | 62,285 | 2026-06-17 | Debug-gate trace for seed 0 |

### Memory / databases

| File | Size | Modified | Contents |
|---|---:|---|---|
| `chroma_db/chroma.sqlite3` | 65,335,296 | 2026-06-27 03:57 | ChromaDB store; 1 collection `oracle_live_memories`; 27,820 embeddings; metadata keys: episode_seed, realized_outcome, run_id, source_month, stored_global_month |
| `chroma_db/723e9bca-…/` (data_level0.bin 45 MB, index_metadata.pickle 2.4 MB, …) | ~48 MB | 2026-06-27 | HNSW vector segment for the live collection |
| `chroma_db/49bd8789-…/` (167 KB) | 2026-04-05 | Orphan/stale HNSW segment dir from the April era (not referenced by the current single collection) |
| `data/startup_society.db` | 1,089,536 | 2026-06-12 | Backend app SQLite: 1 scenario, 1 simulation_run, 1 episode_result, 120 monthly_traces, 120 action_traces (demo DB, not a results artifact) |

### Tests, scripts, docs

| File | Size | Modified | Contents |
|---|---:|---|---|
| `tests/llm_test_results.csv` | 1,197 | 2026-06-03 | LLM provider smoke tests (ollama/openai), 2026-04-14 timestamps |
| `tests/ollama_test_results.csv` | 1,672 | 2026-06-03 | Sample CFO/CMO proposal generations with durations |
| `experiments/run_thesis_experiment.py`, `run_prioritized_thesis_experiment.py`, `run_oracle_v4_confirmation.py`, `thesis_analysis.py`, `oracle_v4_debug_gate.py` | — | — | Run/analysis scripts (analysis = `thesis_analysis.py`) |
| `config/sim_config.py` | — | — | Simulation constants (see §2, item 0) |
| `oracle_v4_breakdown.md`, `oracle_branch_change_summary.md` | — | — | Dev notes on oracle v4 |
| Notebooks (`.ipynb`) | — | — | **None found** outside venv |

---

## 2. Extracted Values

Format: Metric | Value | Source file | Derivation | Run classification | Notes.

### 0. Config constants (requested)

| Constant | Value | Source | Derivation |
|---|---|---|---|
| MAX_STEPS | 120 | `config/sim_config.py` | line 1 |
| INITIAL_CASH | 1,000,000.0 | `config/sim_config.py` | line 3 |
| INITIAL_PRODUCT_QUALITY | 0.1 | `config/sim_config.py` | line 6 |
| BASE_CAC | 50.0 | `config/sim_config.py` | line 8 |

Note: there is no root `config.py`; the constants live in `config/sim_config.py`. (These are code constants, not per-run evidence of what a given run used.)

### 1. Episode counts per policy

| Run | Value | Source file | Derivation |
|---|---|---|---|
| FULL | boardroom 75, oracle_v1 75, oracle_v3 75 → **225 total** | `.../primary_background/primary_episode_metrics.csv` | row count grouped by `policy` |
| PILOT | 1 per policy (boardroom, oracle_v1, oracle_v3) → 3 total | `.../primary_terminal/primary_episode_metrics.csv` | row count by `policy` |
| CONFIRMATION | 50 each for boardroom, oracle_v1, oracle_v3, oracle_v4, oracle_v4_causal → **250 total** | `.../20260412_163603/primary_episode_metrics.csv` | row count by `policy` |
| V4-DEV final_v2 | oracle_v3 20, oracle_v4_causal_hetero 20 | `outputs/oracle_v4_compare_final_v2/comparison.json` | `episodes` field / len(`per_episode`) |

### 2. Seed ranges

| Run | Value | Source file | Derivation |
|---|---|---|---|
| FULL | 0–74 per policy | `primary_episode_metrics.csv` (FULL) | min/max of `seed` per policy |
| PILOT | 0–0 | `primary_episode_metrics.csv` (PILOT) | min/max of `seed` |
| CONFIRMATION | 0–49 per policy | `primary_episode_metrics.csv` (CONFIRMATION) | min/max of `seed` per policy |
| V4-DEV final_v2 | 0–19 both policies | `comparison.json` | min/max of `per_episode[].seed` |

### 3. oracle_frequency actually used (from traces/metadata, not code defaults)

| Run | Value | Source file | Derivation |
|---|---|---|---|
| FULL | **10** | (a) `primary_background_metadata.json` `oracle_frequency: 10`; (b) `primary_action_trace.csv` | (b) months with `refresh_reason == "cadence"` are exactly {10,20,…,110}; only gap value is 10; mean cadence_refreshes = 11.0 = floor(119/10) |
| PILOT | 10 | `primary_terminal_metadata.json`; `primary_action_trace.csv` (PILOT) | same cadence months {10,…,110} |
| CONFIRMATION | **5** | folder name `__freq_5__`; `thesis_summary_report.md` ("Oracle frequency: 5"); `primary_retrieval_trace.csv` | cadence-reason retrieval months are exactly {5,10,…,115}, only gap 5; mean cadence_refreshes = 23.0 |
| V4-DEV final_v2 | 5 | `outputs/oracle_v4_compare_final_v2/run_compare.py` | `oracle_frequency=5` at line 113 (explicit argument in the runner, not a default) |

### 4. Monthly-trace rows

| Run | Value | Source file | Derivation |
|---|---|---|---|
| FULL | boardroom 8,987; oracle_v1 8,997; oracle_v3 9,000 → **26,984 total** | `primary_monthly_trace.csv` (FULL) | row count by `policy` |
| PILOT | 120 per policy → 360 total | `primary_monthly_trace.csv` (PILOT) | row count |
| CONFIRMATION | **MISSING** — no monthly trace file exported for this run | — | directory contains no `primary_monthly_trace.csv` |

### 5. Hard-shock records (shock_label != NO_SHOCK)

| Run | Value | Source file | Derivation |
|---|---|---|---|
| FULL | **675 rows**; by type: COMPETITOR_SURGE 225, RATE_HIKE 225, RECESSION 225; by month: 24→225, 48→225, 72→225; each type×month = 75 | `primary_monthly_trace.csv` (FULL) | filter `shock_label != "NO_SHOCK"`, type = prefix before ":" |
| PILOT | 9 rows, all COMPETITOR_SURGE, 3 each at months 24/48/72 | `primary_monthly_trace.csv` (PILOT) | same filter |
| CONFIRMATION | 150 shock events per policy (750 total) | `primary_episode_metrics.csv` (CONFIRMATION) | sum of `shock_count` (3 per episode × 50); monthly breakdown MISSING (no monthly trace) |

### 6. Termination breakdown / survival

| Run | Value | Source file | Derivation |
|---|---|---|---|
| FULL | boardroom: 2 Bankruptcy / 73 Time Limit (97.33%); oracle_v1: 1 / 74 (98.67%); oracle_v3: 1 / 74 (98.67%) | `primary_episode_metrics.csv` (FULL) | counts of `cause` per policy; matches `primary_summary.csv` Survival % |
| PILOT | all 3 Time Limit (100%) | `primary_episode_metrics.csv` (PILOT) | `cause` |
| CONFIRMATION | boardroom: 1 Bankruptcy / 49 Time Limit (98%); all four oracle policies: 0 / 50 (100%) | `primary_episode_metrics.csv` (CONFIRMATION) | `cause`; matches `primary_summary.csv` |
| V4-DEV final_v2 | oracle_v3: 20/20 Time Limit; oracle_v4_causal_hetero: 15 Time Limit / 5 Bankruptcy (survival 0.75) | `comparison.json` | `causes` list, `survival_rate` |

### 7. Final MRR (mean / median / std) per policy

| Run | Policy | Mean | Median | Std | Source |
|---|---|---:|---:|---:|---|
| FULL | boardroom | 1,389,958.13 | 703,215.52 | 1,424,036.91 | `primary_episode_metrics.csv` (FULL), col `final_mrr` |
| FULL | oracle_v1 | 2,350,117.30 | 1,474,303.56 | 2,369,897.82 | 〃 |
| FULL | oracle_v3 | 2,251,580.38 | 1,360,863.61 | 2,236,788.40 | 〃 |
| CONFIRMATION | boardroom | 1,429,703.14 | 905,423.26 | 1,453,477.27 | `primary_episode_metrics.csv` (CONFIRMATION) |
| CONFIRMATION | oracle_v1 | 2,494,025.33 | 1,734,964.66 | 2,465,859.32 | 〃 |
| CONFIRMATION | oracle_v3 | 2,507,768.20 | 1,735,639.51 | 2,513,313.81 | 〃 |
| CONFIRMATION | oracle_v4 | 2,507,735.89 | 1,735,639.51 | 2,513,271.12 | 〃 |
| CONFIRMATION | oracle_v4_causal | 2,499,774.01 | 1,737,930.45 | 2,493,839.29 | 〃 |
| PILOT | boardroom / v1 / v3 | 15,640.28 / 17,069.77 / 16,667.63 | (n=1) | — | `primary_episode_metrics.csv` (PILOT) |

Means/medians agree with the corresponding `primary_summary.csv` in each run.

### 8. total_reward and avg_rule_40 (mean / median / std)

| Run | Policy | total_reward | avg_rule_40 | Source |
|---|---|---|---|---|
| FULL | boardroom | −1471.66 / −1502.14 / 159.88 | −74.56 / −56.16 / 55.35 | `primary_episode_metrics.csv` (FULL) |
| FULL | oracle_v1 | −1494.43 / −1518.01 / 178.35 | −58.84 / −39.95 / 64.75 | 〃 |
| FULL | oracle_v3 | −1492.38 / −1518.30 / 175.02 | −60.64 / −40.54 / 67.14 | 〃 |
| CONFIRMATION | boardroom | −1469.56 / −1506.74 / 163.58 | −73.25 / −54.06 / 56.44 | `primary_episode_metrics.csv` (CONFIRMATION) |
| CONFIRMATION | oracle_v1 | −1494.18 / −1513.93 / 185.15 | −56.74 / −38.30 / 61.33 | 〃 |
| CONFIRMATION | oracle_v3 | −1489.54 / −1510.88 / 182.73 | −57.50 / −38.06 / 64.41 | 〃 |
| CONFIRMATION | oracle_v4 | −1489.54 / −1510.88 / 182.73 | −57.49 / −38.06 / 64.38 | 〃 |
| CONFIRMATION | oracle_v4_causal | −1491.24 / −1519.21 / 182.73 | −57.42 / −38.00 / 64.48 | 〃 |

Note for the paper: mean **total_reward is slightly worse (more negative) for oracle policies than boardroom** in both runs, even though survival/MRR/rule-40 favor the oracles.

### 9. Recovery / time-to-recover metrics

| Run | Value | Source file | Derivation |
|---|---|---|---|
| FULL | Recovered %: boardroom 67.56, v1 76.00, v3 76.89. Mean recovery months (recovered events only): boardroom 9.24 (n=152), v1 4.82 (n=171), v3 5.57 (n=173); medians 2 / 1 / 2 | `primary_recovery_events.csv` (FULL) | mean of `recovered`; stats of `recovery_time_months` where `recovered==True`; matches `primary_summary.csv` |
| CONFIRMATION | Recovered shock rate %: boardroom 69.33, v1 78.67, v3 80.0, v4 80.0, v4_causal 80.0; mean recovery time 7.28 / 4.94 / 5.25 / 5.25 / 4.61 | `primary_episode_metrics.csv` + `primary_summary.csv` (CONFIRMATION) | per-episode `recovered_shock_rate_pct`, `mean_recovery_time_months` (no separate recovery-events file in this run) |
| PILOT | 1/3 shocks recovered per policy, recovery time 3.0 | `primary_recovery_events.csv` (PILOT) | direct |

### 10. Memory / label data (ChromaDB)

| Metric | Value | Source file | Derivation |
|---|---|---|---|
| Total memory entries | **27,820** | `chroma_db/chroma.sqlite3` | `SELECT count(*) FROM embeddings` (read-only) |
| Collection | 1: `oracle_live_memories` | 〃 | `collections` table |
| Label distribution (`realized_outcome`) | GROWTH 14,590; STAGNATION 10,026; DECLINE 3,204 (sum = 27,820 ✓) | 〃 | `embedding_metadata` where key=`realized_outcome` |
| run_id spread | 40+ distinct `run_id`s; top: three runs of 5,850 each, one 2,340, one 2,189; long tail of ≤600-entry dev runs | 〃 | GROUP BY on key=`run_id` |

**Caveat:** the live Chroma store is cumulative across many runs (mtime 2026-06-27, same night as the final_v2 V4-DEV comparison). It is **not** a snapshot of the April FULL run and its label distribution cannot be attributed to any single experiment without joining on `run_id` (the CSV artifacts do not record which `run_id` a given experiment used → mapping **MISSING**). There is also an orphan April-era vector segment dir (`chroma_db/49bd8789-…`, 2026-04-05). No exported-memories file (JSON/CSV dump of the store) exists.

### 11. Retrieval-quality metrics (RAGAS or similar)

- RAGAS: **MISSING** — no file in the repo mentions RAGAS (repo-wide grep excluding venv/node_modules: zero hits).
- Nearest existing metric: `retrieval_quality.csv` (FULL and PILOT runs) — per-episode share of retrieved memories by realized-outcome bucket, oracle_v3 only. FULL-run aggregate (sum of `retrieval_count`): POSITIVE 21,971; NEGATIVE 4,910; NEUTRAL 89 (= 26,970, matches `primary_retrieval_trace.csv` row count exactly). Mean per-episode retrieval_share: POSITIVE 0.8145, NEGATIVE 0.1822, NEUTRAL 0.0494. Source: `.../primary_background/retrieval_quality.csv`, `primary_retrieval_trace.csv`.

### 12. Statistical test outputs

| Run | Tests | Source |
|---|---|---|
| FULL | Mann-Whitney U (`scipy_mannwhitneyu`), n=75 vs 75, vs boardroom: post_shock_avg_rule40 — v1: U=1084.0, p=8.30e-11 (sig); v3: U=1181.0, p=8.76e-10 (sig). final_mrr — v1: U=2143.0, p=0.0119 (sig); v3: U=2195.0, p=0.0204 (sig) | `.../primary_background/primary_significance_tests.csv` |
| CONFIRMATION | 12 Mann-Whitney U tests vs boardroom (4 policies × 3 metrics). post_shock_avg_rule40_25_60: all sig (p ≈ 4.2–4.9e-08, n=50/50). final_mrr: all sig (p = 0.0223–0.0323). mean_recovery_time_months: none sig (p = 0.18–0.82, n=46/49) | `.../20260412_163603/thesis_summary_report.md` (table; no separate CSV in this folder) |
| PILOT | 4 tests, all p=1.0, n=1/1 (degenerate) | `.../primary_terminal/primary_significance_tests.csv` |

### 13. Oracle stats per run (means per episode)

| Run | Policy | llm_calls | cache_hits | refresh_requests | cadence_refreshes | event_refreshes | Source |
|---|---|---:|---:|---:|---:|---:|---|
| FULL | oracle_v1 | 28.21 | 2.68 | 30.89 | 11.0 | 18.89 | `primary_episode_metrics.csv` (FULL) |
| FULL | oracle_v3 | 29.84 | 0.99 | 30.83 | 11.0 | 18.83 | 〃 |
| CONFIRMATION | oracle_v1 | 35.70 | 3.62 | 39.32 | 23.0 | 15.32 | `primary_episode_metrics.csv` (CONFIRMATION) |
| CONFIRMATION | oracle_v3 | 35.62 | 3.76 | 39.38 | 23.0 | 15.38 | 〃 |
| CONFIRMATION | oracle_v4 | 35.62 | 3.76 | 39.38 | 23.0 | 15.38 | 〃 |
| CONFIRMATION | oracle_v4_causal | 35.68 | 3.70 | 39.38 | 23.0 | 15.38 | 〃 |
| V4-DEV final_v2 | oracle_v3 | 36.2 | — | — | — | — | `comparison.json` (`avg_llm_calls`) |
| V4-DEV final_v2 | oracle_v4_causal_hetero | 45.5 (+ proposal cache_hits 18.95; proposal_sources llm 910 / reuse 960 / cache_hit 379; causal_confidence 0.5526 over 6,747 obs; stress_nodes Churn_Spike 1564 / Steady_State 549 / Cash_Shortage 134 / CAC_Pressure 2) | `comparison.json` |

### 14. Dataset-release artifacts

| Artifact | Status |
|---|---|
| Project README | **MISSING** (only `.pytest_cache/README.md` and venv/node_modules files) |
| LICENSE | **MISSING** |
| DOI / citation file | **MISSING** |
| Schema documentation | **MISSING** as docs; only code schemas (`env/schemas.py`, `oracle/schemas.py`) and dev notes (`oracle_v4_breakdown.md`, `oracle_branch_change_summary.md`) |
| Export scripts | **MISSING**; `seed_dbs.py` seeds databases, no script exports the Chroma memories or packages the CSVs |

---

## 3. Verification Results

### Checks that PASS

1. **3 policies × 75 = 225** — FULL run `primary_episode_metrics.csv` has exactly 75 rows per policy for boardroom/oracle_v1/oracle_v3, 225 rows total. The previously discussed plan matches this run's actual files, including oracle_frequency=10 and seed_start=0 (metadata + cadence trace). CONFIRMATION deviates by design: 5 policies × 50 = 250, freq 5.
2. **Monthly rows = sum of steps** — FULL: monthly-trace rows per policy (8,987 / 8,997 / 9,000; total 26,984) exactly equal the per-policy sums of the `steps` column in episode metrics; all 225 episodes match row-for-row (0 mismatches). Same for PILOT (360 = 360). The sub-9,000 counts are explained by the bankruptcies (episodes shorter than 120).
3. **Shock months exactly 24/48/72** — FULL monthly trace: the only months with `shock_label != NO_SHOCK` are 24, 48, 72; 225 records at each (675 total = 225 episodes × 3). `primary_recovery_events.csv` agrees (675 events, shock_month ∈ {24,48,72}).
4. **Shock type follows seed % 3** — In the FULL run, every episode has exactly one shock type across all three shock months (types_per_seed = 1), and the mapping is exact with zero exceptions: seed%3==0 → COMPETITOR_SURGE, seed%3==1 → RATE_HIKE, seed%3==2 → RECESSION (75 episode-months per cell of the type×month grid). Note the cycle order in the data is (competitor_surge, rate_hike, recession) — consistent with the claimed cycle.
5. **Seeds contiguous, no gaps/duplicates** — FULL: seeds 0–74 per policy, contiguous, 0 duplicates. CONFIRMATION: 0–49 per policy, contiguous, 0 duplicates. PILOT: seed 0. V4-DEV final_v2: seeds 0–19.
6. **Chroma label counts sum** — 3,204 + 14,590 + 10,026 = 27,820 = total embeddings. ✓
7. **Internal consistency of summaries** — `primary_summary.csv` values (survival %, MRR mean/median, recovery %, recovery times, LLM calls) recompute exactly from `primary_episode_metrics.csv` / `primary_recovery_events.csv` in both FULL and CONFIRMATION runs. `retrieval_quality.csv` totals equal the `primary_retrieval_trace.csv` bucket counts (26,970).

### Flags, gaps, and contradictions

1. **`primary_summary_screenshot_no_oracle_v3.csv` / `.md` (CONFIRMATION folder, dated 2026-04-21)** are edited copies of `primary_summary.csv` with the **oracle_v3 row deleted** (values otherwise identical). The filename says "screenshot". If any figure/table in the paper came from these, it silently omits a policy whose numbers exist in the canonical file. Both files' values were cross-checked; no numeric contradiction, but the row removal must be disclosed or the canonical file used.
2. **`outputs/oracle_v4_compare_final_v2/comparison.json` is modified in the git working tree** (git diff: +295 lines appending the `oracle_v4_causal_hetero` block to the committed version, which contained only `oracle_v3`; commit date 2026-06-19, file mtime 2026-06-27). The uncommitted state is the only place the final v4_causal_hetero 20-episode result exists.
3. **oracle_v3 numbers are re-used, not re-run, across V4-DEV files**: seed-0 `final_mrr` = 16,852.409670841043 is bit-identical in `oracle_v4_compare_after_fix`, `oracle_v4_compare_final`, and `oracle_v4_compare_final_v2` comparison.json (and the v3 block in `oracle_v4_compare/comparison.json` differs — 16,878.39 — predating a fix). The "final_v2" oracle_v3 baseline is therefore a carried-over result, not a fresh run alongside v4_causal_hetero.
4. **V4-DEV survival contradiction with CONFIRMATION (context-dependent, not a file error)**: `oracle_v4_compare_final_v2/comparison.json` shows oracle_v4_causal_hetero survival 0.75 (5 bankruptcies / 20 episodes), while the CONFIRMATION run's `oracle_v4_causal` policy shows 100% survival over 50 episodes. These are different policy variants (`oracle_v4_causal` vs `oracle_v4_causal_hetero` with heterogeneous LLM proposals) from different dates (April vs June); the paper must not mix them.
5. **oracle_v3 ≈ oracle_v4 near-duplication in CONFIRMATION**: oracle_v4's median final MRR, post-shock rule-40, recovery stats, and significance tests are identical to oracle_v3's (U=457, p=4.67e-08 for both; medians equal to 9+ digits), and means differ only in the 5th significant figure. Any claim that v4 differs from v3 is not supported by this file.
6. **CONFIRMATION run has no monthly trace, no recovery-events file, no significance CSV** — those metrics exist only in aggregated per-episode columns and the markdown report. Monthly shock-month verification for this run is therefore not possible from files (marked MISSING in §2.5).
7. **Chroma store is cross-run cumulative** — 40+ run_ids; no artifact maps experiment names to run_ids, so the 27,820 memories / label distribution cannot be tied to the FULL or CONFIRMATION run. The three 5,850-entry run_ids share an identical DECLINE/GROWTH/STAGNATION split (620/3,063/2,167 twice; 619/3,079/2,152 once), consistent with repeated 20-episode dev runs, but this is inference — the mapping itself is MISSING.
8. **Empty run directory**: `oracle_v4_confirmation__episodes_50__freq_5__seed_0__20260413_062028/` exists with zero files (an aborted rerun the morning after the kept confirmation run).
9. **Total reward direction**: oracle policies have *worse* mean total_reward than boardroom in both FULL and CONFIRMATION runs (§2.8) while headline metrics (survival, MRR, rule-40, recovery) favor the oracles. Not a file contradiction, but the paper should not claim reward improvement.
10. **PILOT folder timestamp anomaly**: the 1-episode terminal run (2026-04-05) was executed *after* the FULL background run started (2026-04-04) but finished 7 days before it; both live under `prioritized_thesis_run/`. File dates, not folder order, identify the primary result.
11. **No RAGAS, no dataset-release files** (README/LICENSE/DOI/schema doc/export script) exist anywhere in the repo — all MISSING (§2.11, §2.14).
