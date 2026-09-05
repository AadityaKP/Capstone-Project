# Calibration protocol — ROUND 2 — FROZEN BEFORE ANY RESULT

Branch `round2` (from `physics-v2`). Committed before any round-2 code exists and
before any round-2 state table is built. Results are evaluated against this text
verbatim; deviations are labelled where they occur. Round 1 (`PROTOCOL.md`,
commit `484fd35`) is closed: its HOLDOUT was evaluated once and is never used
again for fitting or selection.

## Why a round 2

Round 1 recorded two FAILs. Both diagnoses point at the *backtest mapping and the
financing rule*, not at the fitted physics:

* Corridor artifact FAIL (boardroom growth std ratio 0.16): every company enters
  with identical assumed churn, price and LTV:CAC=3, so the state-responsive
  corridor sees near-identical states.
* Financing FAIL (2/6): the rescue-regime rule (fires only when runway < 18 mo)
  under-finances companies that in reality raised at ~48 months of runway.

Round 2 changes **nothing that was fitted in round 1**. `SATURATION_ACQUISITION_RATE`
stays at 0.0727. Round 2 changes exactly two things, both behind flags with
defaults = round-1 behaviour:

1. `mapping_version="v2"` — company-specific CAC derived from each company's own
   observed S&M efficiency (no look-ahead).
2. `financing_model="opportunistic"` — raise hazard by runway bin, active from
   month 1, parameters from CAL-company quarters only.

## Non-negotiable rules

1. **No round-1 artifact is modified.** `real_company_backtest.csv`,
   `real_company_backtest_v2.csv`, `panel_split.csv`, every scorecard, every
   figure and every CSV under `validation/` stays byte-identical. Round-2
   outputs carry the suffix `_r2`.
2. **Round-1 fitted constants are frozen.** No physics constant changes. If a
   round-2 result would be improved by re-fitting `s`, that is a finding, not an
   action.
3. **Evaluation set EVAL2 is fresh and touched once.** EVAL2 = the 19 round-1
   HOLDOUT companies, each initialized from a *later* quarter than round 1 used
   (rule in §Operationalization). These states were never used in any fit or
   parameter estimate. EVAL2 is run once, in Phase R2-4.
4. **Development set DEV2 = the 20 round-1 CAL companies at the same later
   quarter.** Any number of DEV2 runs is allowed for debugging the mapping and
   the financing rule; the count is disclosed in the report.
5. **All round-2 parameters are estimated from CAL companies only** (financing
   hazard table, CAC clamp percentiles). D4 in round 1 used the full panel; the
   round-2 hazard table is recomputed on CAL quarters only and the difference
   is reported.
6. **No look-ahead.** Every per-company mapping quantity uses only quarters at
   or before the initialization quarter.
7. **Every change sits behind a flag**, default = round-1 behaviour
   (`mapping_version="v1"`, `financing_model="rescue"`). The round-1 HOLDOUT
   numbers must reproduce exactly from the `round2` branch with round-1 flags.

## Acceptance criteria (frozen now)

Primary mapping: $250 ARPA (as in round 1). $50 re-run reported as sensitivity.
All criteria are evaluated on EVAL2 only.

| ID | Criterion | PASS | PARTIAL |
|---|---|---|---|
| R2-C1 | median \|4q cumulative growth error\|, hold arm, round-1 `s` frozen (this is an out-of-time test of the round-1 fit) | ≤ 10 pp | ≤ 20 pp |
| R2-SIGN | growth-sign agreement, hold arm | ≥ 70% | — |
| R2-CORR | std across companies of boardroom 4q growth ≥ 1/3 of std of actual 4q growth **AND** Spearman(hold growth, actual growth) > 0.3 | both | — |
| R2-FIN-a | share of EVAL2 companies with > 50% seed survival under hold, `financing_model="opportunistic"` (ground truth: every panel company survived these quarters) | ≥ 90% | ≥ 75% |
| R2-FIN-b | within-run ablation: of companies with 0 surviving seeds under hold with financing **off**, share that survive (>50% seeds) with financing **on** | ≥ 80% | — |
| R2-REG | `round2` branch with round-1 flags reproduces `real_company_backtest_v2.csv` HOLDOUT rows exactly, and legacy flags reproduce the research-scale A2/E1 numbers exactly | exact | — |
| R2-VAR | std across companies of *hold-arm* LTV:CAC at initialization under `mapping_version="v2"` is > 0 with interquartile range ≥ 0.5 (the mapping produces state variety; reported on DEV2 before EVAL2 is run — this is a precondition, not a result) | — | — |

Secondary (reported, not criteria): signed median error; per-company table;
hypergrowth tail; boardroom − hold increment (in-model counterfactual only);
$50 mapping sensitivity; number of DEV2 iterations.

## Operationalization (frozen with the criteria)

**Initialization quarter.** For each company let q0 be the round-1
initialization quarter (earliest complete quarter). Round 2 initializes at
q0 + 8 if at least 12 of the 19 HOLDOUT companies have ≥ 13 complete quarters
from q0 (so that q0+8 has four subsequent quarters); otherwise q0 + 4. The
choice is made by `make_eval2.py` from coverage metadata before any result
exists and is printed and committed. Companies lacking the required quarters are
excluded and listed.

**State mapping (shared with v1).** MRR = quarterly revenue / 3; monthly S&M and
R&D = quarterly / 3; cash as reported; hold arm holds S&M and R&D at their
initialization-quarter monthly values for 12 months. Price/ARPA and churn remain
assumed (2.7%/mo at $250) and labelled as such.

**Company-specific CAC (`mapping_version="v2"`).** With q the initialization
quarter, ARPA the assumed price, c_m the assumed monthly churn:

```
trailing_SM      = Σ S&M over quarters q-3 .. q
net_new_rev      = Rev_q − Rev_{q-4}
churned_rev_est  = 12 · c_m · mean(Rev over q-3 .. q)      # ≈ Σ 3·c_m·Rev per quarter
gross_new_rev    = max(net_new_rev + churned_rev_est, ε)
new_customers    = gross_new_rev / (3 · ARPA)                # quarterly rev → customers
CAC_company      = trailing_SM / new_customers
```

CAC is clamped to the [p5, p95] of the same quantity computed over **CAL
companies' quarters** (all quarters with 4 trailing quarters available); the
clamp bounds and the number of clamped companies are reported. `state.cac` is
set to `CAC_company`; `state.ltv` follows from ARPA and c_m as in v1, so
LTV:CAC now varies across companies. Nothing else in the mapping changes.

**Opportunistic financing (`financing_model="opportunistic"`).** Using the D4
raise definition unchanged, over CAL companies' *burning* quarters:

* Runway bins (months): [0,12), [12,24), [24,48), [48,∞).
* q_b = share of burning quarters in bin b with a raise; monthly hazard
  h_b = 1 − (1 − q_b)^(1/3).
* K_b = median over raises in bin b of (raise amount / monthly net burn).
* Each simulated month, if the company is burning (net cash flow < 0): with
  probability h_b a raise of K_b × monthly net burn is added to cash. The draw
  uses the environment RNG stream so matched seeds remain matched across arms.
  Not agent-visible. Active from month 1.
* Bins with < 10 CAL burning quarters inherit the neighbouring bin's values;
  this is reported.

**Per-company sim growth, C1 error, corridor std, Spearman** — identical to
round 1 wording (median over surviving seeds of Q4/Qinit − 1; a company with 0
surviving seeds under an arm is excluded from that arm's medians but counted in
R2-FIN).

## Phase map

* **R2-1 Data.** `make_eval2.py` → `eval2_states.csv` (EVAL2 + DEV2 rows,
  column `split ∈ {DEV2, EVAL2}`), `cac_mapping_r2.csv` (per-company CAC, clamp
  flags), `financing_hazard_r2.json` (bins, q_b, h_b, K_b, n per bin). Committed
  before any simulation.
* **R2-2 Code.** Flags `mapping_version`, `financing_model`; unit tests
  (`tests/test_round2.py`): hazard math, CAC formula on a hand-worked example,
  no-look-ahead assertion (mapping function receives only rows ≤ q), flags
  default to round-1 behaviour.
* **R2-3 Gates (DEV2 only).** R2-REG exact-match; R2-VAR precondition on DEV2;
  DEV2 hold-arm run(s) to confirm financing fires and the mapping is sane.
  Iteration count recorded.
* **R2-4 EVAL2 — once.** 4 arms (noop-hold, hold, heuristic, boardroom) × 30
  matched seeds × 12 months × {financing on, off} × {$250, $50}. Outputs
  `real_company_backtest_r2.csv`. Verdicts computed by script, not by hand.
* **R2-5 Report.** `calibration_report_round2.md` with every criterion and
  verdict including failures; `_r2` scorecard rows appended; validation report
  §5d.

If R2-C1 or any other criterion FAILS on EVAL2, that is the result. There is no
round 3 inside this protocol.
