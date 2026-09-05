# Round-2 log

Branch `round2` from `physics-v2`. Pre-registration committed before any code:
`validation/calibration/PROTOCOL_round2.md`, `validation/oracle/BRIEF_V2_SPEC.md`,
`validation/round2/ROUND2_PLAN.md` (verbatim). Tag `r2-preregistered`.

## Decisions (day 1, recorded before any round-2 run)

| # | Decision | Rationale |
|---|---|---|
| D1 | Legacy physics stays the paper's primary for research-scale results. v2-physics re-runs of A2 and A3 are appended as robustness rows. | All 75-seed runs are legacy; re-running everything is not possible in 4 days. Robustness rows convert the calibration into a strength. |
| D2 | Calibration round 2 runs under `PROTOCOL_round2.md`, time-boxed to day 2. EVAL2 = HOLDOUT-19 at q0+8; exactly two changes (company-specific CAC; opportunistic financing); no re-fit of `s`. | Both FAIL diagnoses point at mapping/financing, not physics. Same-company later-quarter states are the only fresh evaluation set available; disclosed. |
| D3 | Brief v2 under `BRIEF_V2_SPEC.md`. Variant chosen by the frozen rule. If both fail B1, brief v1 stays and A4 FAIL is a limitation. | Level-blindness is the one agent defect that hurts at real scale and is fixable by prompt design. |
| D4 | Memory narrative: the value is the brief mechanism (trend/shock reactor); episodic retrieval is a small, significant modulator (≈3%). The random-shock ablation decides whether even that survives. | Matches the recorded evidence; avoids over-claiming. |
| D5 | Outcome metrics in the paper: final MRR, survival, post-shock Rule-of-40 recovery, brief accuracy. "Global Reward" is dropped (reward pre-declared excluded; oracle has *worse* reward while winning every headline metric). | Claim audit. |
| D6 | E6 (permanent drawdowns) stays a limitation unless day 4 is free; then a `shock_recovery="mean_revert"` flag + 20-seed A3 rerun is a stretch item. | Structural; changes the headline world. |

## Session log

- **S1** (2026-09-05): pre-registration committed, tag `r2-preregistered`. No code.
