# physics_v2 calibration report

Date: 2026-09-05 · Branch `physics-v2` · Protocol frozen before any result in
[PROTOCOL.md](PROTOCOL.md) (commit `484fd35`). Every number here is computed
from data on disk; failures are reported as results, not iterated away.

## 0. Verdict summary (frozen criteria, HOLDOUT, primary $250 mapping)

| Criterion (frozen bar) | Result | Verdict |
|---|---|---|
| C1-v2 median \|4q growth error\| ≤ 10pp PASS / ≤ 20pp PARTIAL | **8.1pp** (v1: 49.6pp; same-subset v1: 51.3pp) | **PASS** |
| Growth-sign agreement ≥ 70% | **100%** (19/19) | **PASS** |
| Corridor artifact: boardroom growth std ≥ actual/3 AND Spearman(hold, actual) > 0.3 | std 0.046 vs actual 0.294 (ratio **0.16**); Spearman **0.73** (p=4e-4) | **FAIL** |
| Financing: ≥ 80% of the 6 v1 all-seed bankrupts survive under hold | **2/6** (CRWD 26/30, DOMO 21/30 survive; ASAN/ESTC/RPD/TENB 8/30) | **FAIL** |
| Regression: legacy reproduces recorded numbers exactly | A2 per-seed episodes + E1 aggregation exact | **PASS** |
| E4-v2 research scale: spend ratio inside EDGAR [37%, 93%] | **67.6%** (v1: 8.0–8.5%) | **PASS** |

Secondary (not criteria): HOLDOUT signed median error −3.2pp (the sim now
slightly *under*-projects); CAL median |error| 13.0pp — HOLDOUT ≤ CAL, no
overfit signature; $50-ARPA sensitivity mapping 9.0pp. The boardroom−hold
increment, negative for 100% of companies in v1, is **+5.2pp median, positive
for 84%** of HOLDOUT companies under v2 (in-model counterfactual only).

## 1. Split

39 panel companies → CAL 20 / HOLDOUT 19, stratified by init-revenue tercile
and actual 4q-growth sign, seed 20260904, written to `panel_split.csv` before
any fitting (commit `484fd35`). All fitting used CAL only. HOLDOUT was
evaluated exactly once (§5). The six v1 hold-arm all-seed bankruptcies split
3/3 (CAL: ASAN, DOMO, TENB; HOLDOUT: CRWD, ESTC, RPD).

**Protocol incident, disclosed.** A Phase 4 run was launched ~15 minutes
before the D1 audit finished consolidating. D1 then surfaced an unconverted
scale bug (competitive-entry saturation, §2.1). The premature run's outputs
were **deleted unread** — no number from it was ever observed — the fix was
made on D1 evidence alone, the CAL fit was re-run, and Phase 4 ran once. The
committed Phase 4 results are the only HOLDOUT evaluation that was ever read.

## 2. Phase 1 diagnostics (all on disk before any fix; commits `60bfea7`, `9d6…`)

### 2.1 D1 scale audit (`scale_audit.csv`)
274 constants classified across the physics/agent stack; 62 flagged
yes/maybe, adversarially verified. Beyond the known suspects, one new
finding: `competitive_entry_shock`'s market-attractiveness anchor
`(mrr − 50k)/50k` saturates its sigmoid at EDGAR scale, pinning entry
probability at the 0.2/month ceiling for every real company — double the
calibration rate, ~−22%/yr expected price erosion, a one-way competitor
ratchet — on the live shock path even with scheduled shocks off.

### 2.2 D2 error attribution (6 CAL companies, 2 per tercile)
Mechanism switches, RNG-aligned (patched functions consume identical draws):

| Mechanism zeroed/pinned | Median growth removed |
|---|---|
| (a) marketing acquisition | **82.2pp** |
| (c) expansion MRR (flat 2%/mo) | 40.5pp |
| (b1) churn multiplier stack (pin flat 2.7%) | 20.2pp |

(b2) The flat churn required to reproduce actual growth is 4.7–20%+/mo for
4 of 6 companies — implausible vs the 2.7% benchmark. **Conclusion: the
acquisition curve dominates; churn is reported as sensitivity, not fit.**

### 2.3 D3 identifiability (same 6, s over 4 orders of magnitude)
|error| vs `SATURATION_ACQUISITION_RATE` has a clear interior minimum
(~0.075 → 30.9pp on these deliberately hard companies) against 44.9pp at
s→2e-4 and 2,578pp at s=2.0. Not flat → fitting is meaningful
(`d3_saturation_sweep.png`).

### 2.4 D4 financing evidence (`d4_financing_summary.json`)
113 raises in 350 burning company-quarters (32%). Raises happen at every
runway level (median runway at raise: 48 months — opportunistic financing
dominates). The conditional table breaks at 18 months: P(raise/quarter |
burning) ≥ 0.50 below, 0.26 above. Rescue-regime parameters: **R = 18mo,
K = 24.4× monthly net burn (median rescue raise), p = 0.261/month.**

### 2.5 D5 volatility attribution (diagnose only)
E2/E5 excesses are structural, not noise: freezing the Hill draws AND all
random macro shocks still leaves quarterly growth std 0.095 (2× EDGAR) and
pushes lag-1 autocorr to 0.99. Neither retuning demand noise nor shock
probabilities can fix E2; not attempted here (`d5_volatility_attribution.png`).

## 3. Phase 2 fixes (all behind flags; defaults legacy; F4 tests)

| Constant / mechanism | Old (legacy) | New (v2 flag) | Provenance |
|---|---|---|---|
| `SATURATION_ACQUISITION_RATE` | 0.20 (ASSUMED) | **0.0727 fitted**, CI [0.0475, 0.1113] (`marketing_curve="v2"`) | CAL fit, §4 |
| Financing | none | R=18mo, K=24.4×, p=0.261/mo (`financing_enabled`) | D4 panel medians |
| Competitive entry attractiveness | $50k sigmoid anchor | pinned to calibration point (`competitive_entry="scale_neutral"`) | D1 |
| Boardroom mkt cap | max(30% cash, $20k×sa) | max(30% cash, 64.1% MRR) | EDGAR S&M p90 |
| Boardroom R&D floor | ($20k + deficit·$50k)×sa | MRR × max(13.1%, deficit×10%) | EDGAR R&D p10 |
| Boardroom mkt floor | max($5k×sa, 2% MRR) | 2% MRR | floor, not strategy |
| v4-causal R&D cap | max(25% cash, $30k×sa) | max(25% cash, 36.5% MRR) | EDGAR R&D p90 |
| Proposal-eval burn proxy | headcount×$10k | `business_logic.monthly_burn` | real burn exists |
| CMO tiers | $20k/$10k/$2k ×scale | 40%/20%/4% of current MRR | ≡ legacy at $50k; p50/p10–25 EDGAR |
| CPO tiers | $15k/$8k/$3k ×scale; guard $200k×scale | 30%/16%/6% of MRR; guard 4×MRR | ≡ legacy at $50k; ~p75/p25 EDGAR |
| CFO recruiting cost | $10k×scale | $10k unscaled | per-head, not per-revenue |

`scale_absolutes`/`scale` (multiply-by-initial-mrr/50k) is superseded by the
corridor, which tracks *current* MRR — the v1 approach froze every company's
ratios at its starting size, which is precisely the "+43.7% for everyone"
artifact the C1 decomposition exposed.

**D1 items left unconverted, with justification:**
- *Churn tenure decay* (`compute_churn_rate`, months_elapsed×0.4): treats every
  backtest company as founded at month 0, cutting effective churn below the
  band median over the episode. D2 measured the whole churn-multiplier stack
  at ~20pp; the effect is identical across CAL and HOLDOUT and its mean is
  absorbed by the fitted s (fitting both would be unidentifiable on 4q
  growth). Recorded as a known bias.
- *Reward-function constants* (mrr/$1M etc.): reward is a pre-declared
  excluded outcome (claim audit); no arm optimizes it online.
- *Oracle memory tiers* (`get_mrr_tier`, mrr_bracket): every real company maps
  to the top tier — degrades retrieval specificity equally for all companies;
  disclosed as a limitation of the exploratory oracle-at-scale run (§6).
- *Hiring caps* (10 hires; ActionModifier 0/1/2): count-scaled; hiring has no
  revenue channel (A1) and $8k/mo payroll is negligible at EDGAR burn scale.
- *Verifier-rejected flags* (oracle burn fn, LTV:CAC 3.0, competitors>8,
  proposal cache buckets, margin default): adversarial verify disagreed;
  recorded in `scale_audit.csv` with notes.

## 4. F1 fit (CAL only) — `marketing_fit.md`, `f1_loss_curve.csv`

Hold arm, 10 matched seeds, full v2 package. Loss (CAL median |error|) falls
from 32.1pp (s=0.01) to **12.9pp at s=0.0727** and rises to 110.7pp by
s=0.30; the nearest grid point to the falsified 0.20 gives 49.7pp — matching
v1's recorded 49.6pp failure. Bootstrap CI over companies [0.0475, 0.1113].
Churn-band sensitivity at the fit (reported, not fit): flat 2.0%/2.7%/3.4%
churn moves CAL median |error| by single-digit pp (table in
`marketing_fit.md`).

## 5. Phase 3 gate + Phase 4 HOLDOUT (details)

Gate (`p3_gate_summary.json`): legacy exact-match on all five per-seed episode
metrics and the E1 aggregation; v2 at research scale keeps boardroom > noop
in 10/10 seeds (median final MRR $640k vs $3.3k; heuristic $529k; random dies
10/10), spend ratio 8.5% → 72.3%.

HOLDOUT backtest (`real_company_backtest_v2.csv`, 4 arms × 30 seeds × 12mo ×
39 companies × 2 price mappings, v1 file untouched): verdicts in §0. Notes:
- Hypergrowth is the residual error tail: CRWD (+103% actual) and DDOG (+88%)
  under-project by ~44–56pp — a single global saturation parameter cannot
  produce top-decile growth; slow growers (BAND +7%) over-project by ~27pp.
- **Corridor-artifact FAIL diagnosis:** the *hold* arm now varies with real
  spend and correlates 0.73 with actual growth, but the *boardroom* arm still
  compresses cross-company variance (std 0.046 vs v1's 0.054 — barely moved).
  The binding constraint is the mapping, not the floors: every company enters
  with the same assumed churn, price and LTV:CAC=3, so the state-responsive
  tiers see near-identical states. Company-specific churn/CAC are
  unobservable in XBRL (declared not-validated in the data audit); within
  this mapping the criterion appears unreachable, which the protocol treats
  as a FAIL, not an excuse.
- **Financing FAIL diagnosis:** raises fire (99 raise-events across HOLDOUT
  hold arms) and save CRWD and DOMO, but the four worst burners need
  financing within 1–3 months of episode start; at p=0.261/month the chance
  of no raise in 2 draws is 55%. Real companies raised *before* runway got
  short (median runway at raise 48mo — D4). The rescue-regime
  parameterization mandated by the protocol (rule params from D4 medians)
  under-finances relative to the opportunistic financing these companies
  actually did. Recording the FAIL honestly; a pre-emptive financing model
  would need different — and fresh — parameters, and another HOLDOUT is not
  available for it in this protocol round.

## 6. Oracle at real scale (first-ever numbers; exploratory)

oracle_v3 (llama3.1:8b via Ollama, temp 0), 8 HOLDOUT companies spanning the
revenue range × 10 matched seeds, v2 flags, research-default prompt, memory
accruing across seeds within a company (mirrors the recorded research
design). Results: `validation/results/oracle_v3_real_scale_v2.csv`; summary
appended below when the run completes. Caveats: oracle memory tiers saturate
at real scale (§3); the brief channel is level-blind (A4) — briefs at these
states read LOW risk / ACCELERATING and the ActionModifier scales marketing
up, so the oracle arm spends more than the corridor-bounded boardroom.

## 7. Runtime

D1 audit workflow ~15 min (12 agents); D2+D3+D5 ~4 min total; D4 <10s;
F1 fit ~7 min (×2, refit after D1); Phase 3 gate ~4 min; Phase 4 backtest
~19 min; E-battery v2 ~4 min; oracle runs ~1–2 h (background).

## 8. Reproduction

```
python validation/calibration/make_split.py            # split (already frozen)
python validation/calibration/d2_error_attribution.py  # D2
python validation/calibration/d3_saturation_sweep.py   # D3
python validation/calibration/d4_financing_evidence.py # D4
python validation/calibration/d5_volatility_attribution.py  # D5
python validation/calibration/f1_marketing_fit.py      # F1 fit (CAL only)
python -m pytest tests/test_physics_v2.py              # F4
python validation/calibration/p3_regression_gate.py    # Phase 3 gate
python validation/calibration/p4_holdout_backtest.py   # Phase 4 (HOLDOUT)
python validation/calibration/p4_e_battery_v2.py       # E-battery v2
python validation/calibration/p4_oracle_holdout.py     # oracle (needs Ollama)
```
