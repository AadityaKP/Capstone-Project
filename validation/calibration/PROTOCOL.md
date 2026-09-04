# physics_v2 calibration protocol — FROZEN BEFORE ANY RESULT

Committed on branch `physics-v2` before any diagnostic was run and before any
constant was changed. Every criterion below was fixed at this commit; results
(Phase 4) are evaluated against this text verbatim. Any deviation must be
labelled as a deviation where it occurs.

## Non-negotiable rules

1. **Nothing is fixed before it is measured.** Phase 1 diagnostics (D1–D5) run
   and are written to disk before any constant changes.
2. **Split-panel calibration.** The 39 panel companies are randomly split into
   CAL (~20) and HOLDOUT (~19), stratified by revenue scale (terciles of
   initialization-quarter revenue) and by actual 4-quarter growth sign. Fixed
   seed (`SPLIT_SEED = 20260904`), split written to
   `validation/calibration/panel_split.csv` BEFORE any fitting. All fitting
   uses CAL only. HOLDOUT is touched exactly once, in Phase 4.
3. **Acceptance criteria (frozen now, before results):**
   * **C1-v2 (HOLDOUT):** median |4q cumulative growth error| ≤ 10 pp → PASS,
     ≤ 20 pp → PARTIAL, else FAIL. Growth-sign agreement ≥ 70% → PASS.
   * **Corridor-artifact check (HOLDOUT):** across companies, the std of
     simulated boardroom 4q growth ≥ 1/3 of the std of actual 4q growth, AND
     Spearman corr between sim hold-arm growth and actual growth > 0.3.
     (Kills "+43.7% for everyone".)
   * **Financing check:** of the companies that previously went bankrupt
     in-sim on all seeds under their own real spend (v1 backtest: ASAN, CRWD,
     DOMO, ESTC, RPD, TENB), ≥ 80% now survive under hold.
   * **Regression check:** with legacy config, the deterministic
     research-scale suite reproduces recorded numbers exactly (Phase 3).
   * **E4-v2 (research scale, new corridor):** sim median discretionary-spend
     ratio inside EDGAR [p10, p90] = [37%, 93%] of revenue (S&M+R&D).
4. **Never overwrite v1 results.** All new outputs get `_v2` suffixes or a
   `physics_version` column. Scorecards get appended rows, not edits.
5. **Every physics change sits behind config flags** (`marketing_curve="v2"`,
   `financing_enabled`, `corridor="scale_aware"`), default = legacy. Recorded
   v1 results must remain reproducible from the same commit.
6. **One round trip.** If C1-v2 FAILS on HOLDOUT, that is the result — the fit
   is not iterated against HOLDOUT.

## Operationalization (frozen with the criteria, before any run)

* **Primary mapping** for all criteria: the $250-ARPA price assumption (the v1
  primary). The $50 mapping is re-run and reported as sensitivity only.
* **Per-company sim growth** = median over surviving seeds of 4q cumulative
  revenue growth (quarter 4 revenue / initialization-quarter revenue − 1),
  exactly as v1 computed it. A company with zero surviving seeds under an arm
  has undefined growth for that arm and is excluded from medians for that arm
  (but counts in the financing check and is reported).
* **C1-v2 error** = per-company (sim hold growth − actual 4q growth); the
  criterion is the median of |error| across evaluable HOLDOUT companies.
* **Corridor-artifact std** = std across evaluable HOLDOUT companies of
  per-company boardroom sim growth, compared to std of actual 4q growth over
  the same companies. Spearman is between per-company hold sim growth and
  actual growth over the same companies.
* **Financing check** denominator: the 6 v1 all-seed hold-arm deaths present
  in HOLDOUT ∪ CAL (listed above); "survive" = >50% of seeds complete 12
  months under hold with `financing_enabled` v2 physics. Reported for both
  splits, criterion applies to all 6 (they are a diagnosis-derived set, not a
  fitted set).
* **D2 churn mechanism operationalization:** (b1) churn pinned flat to the
  assumed band median (2.7%/mo at $250), bypassing the simulator's
  quality/macro/tenure multipliers — isolates the multiplier stack;
  (b2) "company-implied churn" = the flat churn that makes sim hold growth
  match actual growth (bisection, 10 seeds) — reported as the churn the data
  would require; if implausible (> 8%/mo) churn assumptions cannot be the
  dominant term.

## Phase map

* Phase 1 — diagnostics, no fixes: D1 scale audit → `scale_audit.csv`;
  D2 error attribution on 6 CAL companies (2 per size tercile);
  D3 saturation-constant sensitivity sweep (identifiability check — if the
  |error|-vs-constant curve is flat, STOP and report, fitting would be fake);
  D4 financing evidence from the panel (raise frequency, size vs quarterly
  burn, runway at raise); D5 E2/E5 volatility attribution (diagnose only).
* Phase 2 — fixes behind flags: F1 marketing curve v2 (one global free
  parameter, fit on CAL, bootstrap CI); F2 financing rule (params from D4
  medians, not invention; environment rule, not agent-visible); F3 scale-aware
  corridor (every dollar floor/cap → % of MRR or cash, anchored to EDGAR
  bands; legacy preserved behind flag); F4 deterministic unit tests.
* Phase 3 — regression gate: legacy flags must reproduce recorded numbers
  exactly (A2 subset + E1 aggregation); v2 flags at research scale must not
  flip the A2 ordering (if boardroom stops beating noop → STOP and report).
* Phase 4 — the only HOLDOUT touch: 4 arms × 30 matched seeds × 12 months,
  same decomposition as C1; oracle_v3 (Ollama) on ≥ 8 HOLDOUT companies ×
  10 seeds; E1/E3/E4 research-scale v2 rows appended to the scorecard.
* Phase 5 — reporting: `calibration_report.md` with every frozen criterion and
  verdict including failures; `_v2` scorecard rows; report §5c; README.
