# System audit — what is actually implemented

Audit date: 2026-08-30. Branch: `review2-sim-frontend` (clean working tree at `fbd9401`).
Method: every statement below was traced to source code or an on-disk artifact; file:line
references are given. Where documentation and implementation disagree, the disagreement is
flagged. This document is the factual basis for `validation/validation_plan.md`.

---

## 1. The actual pipeline

```
INPUTS  seed s, initial_config (defaults: MRR $50k, cash $1M, CAC $50, price $50,
        product_quality 0.1, headcount 1)                    [env/startup_env.py:140-164]
   │
STATE   EnvState: 16 monthly variables (see §2)              [env/schemas.py:5-47]
   │
OBSERVE Policies receive the FULL EnvState object, not the gym observation vector.
        run_simulation calls agent.get_action(env.state)     [simulation_runner.py:358]
        (the Box observation space at startup_env.py:108-111 is dead weight for every
        policy in the repo — nothing consumes _get_obs())
   │
RETRIEVE (oracle_v3/v4 modes only, on refresh months) trend context from a 5-month
        deque + ≤3 Chroma memories filtered to the CURRENT run_id + (v4_causal, under
        active shock) Neo4j graph context                     [oracle/oracle.py:117-171]
   │
DECIDE  Boardroom: 3 rule-based C-suite proposals → scored → composed into one
        ActionBundle → ActionModifier scales it by the LLM brief → sanity bounds →
        dynamic minimum floors → cash-conflict resolver        [boardroom/boardroom.py:109-326]
   │
PARSE   ActionAdapter.translate_action clamps/typos-proofs the bundle
        (spend ≥ 0, hires ≥ 0, price change clamped to [-0.5, +1.0])  [agents/adapter.py]
   │
STEP    StartupEnv.step: shocks → cascade/hysteresis/recovery → marketing Hill response
        → expansion → R&D → churn → MRR update → cash update → pricing → hiring clip →
        burn subtraction → CAC/LTV update → Rule-of-40 → reward   [env/startup_env.py:168-289]
   │
OUTCOME terminated = cash ≤ 0; truncated = 120 months. Per-episode metrics
        (final MRR/cash, survival, avg Rule-of-40, recovery events) [simulation_runner.py:459-498]
```

Time resolution: **monthly**. Horizon: **120 months** (research), 12 months (product what-if).
Success/failure: bankruptcy (cash ≤ 0) vs. surviving to the time limit.

### Stochastic components (per step)

| Draw | Site | Count |
|---|---|---|
| Interest-rate shock (p=0.1) | business_logic.py:46 | 1 |
| Consumer-confidence shock (p=0.1) | :52 | 1 |
| Competitive entry (p dynamic in MRR) | :57 | 1 |
| Recession cascade (conditional on macro) | :89 | 0–1 (legacy) / 1 (deterministic_rng) |
| Marketing Hill params (alpha, gamma, beta) | :210-222 | 3 (legacy) / 1 (scale-aware) |
| Pricing elasticity U(−0.9, −0.2) | :358 | 1 |

Plus **deterministic scheduled hard shocks** at months 24/48/72 with type = `seed % 3`
cycling (competitor_surge, rate_hike, recession) [startup_env.py:195-198]. These are a
research fixture, disabled in the founder profile.

### RNG regimes — critical for any comparison

- **Legacy (default, used by ALL recorded results):** physics draws from the *global*
  `random` module. A policy that also draws globally (only `random` does) shifts the
  environment's stream. For non-drawing policies (heuristic, boardroom, all oracle modes)
  the macro world is exogenous and shared at equal seed, because per-step draw count is
  constant except the recession cascade, whose trigger depends only on macro state, which
  is itself action-independent. **Consequence: the recorded boardroom-vs-oracle
  comparisons are seed-matched in practice, but the `random` policy is not comparable to
  anything in legacy mode.** (Verified empirically in `analysis/verify_shared_world.py`.)
- **`deterministic_rng: True` (added later, off by default):** env owns a private
  generator, fixed draws per step. This is the only defensible mode for new ablations and
  is used for all new experiments in this validation.

## 2. Variable inventory

"Agent-visible" = present in the EnvState handed to policies (all are). "EDGAR proxy"
refers to the quarterly panel in `data/edgar.db` (see `validation/edgar_data_audit.md`).

| Variable | Meaning | Unit | Update mechanism | Externally observable? | EDGAR proxy |
|---|---|---|---|---|---|
| mrr | Monthly recurring revenue | $/mo | churn decay + Hill marketing response + expansion + pricing | yes | quarterly revenue ÷ 3 (**direct**) |
| cash | Liquid cash | $ | + MRR·margin − burn − spend − hiring | yes | cash & short-term investments (**direct**) |
| cac | Customer acquisition cost | $ | spend ÷ est. new users, macro-scaled | no | ~ magic-number inverse (**weak proxy**) |
| ltv | Lifetime value | $ | price ÷ churn | no | none (**unavailable**) |
| churn_enterprise/smb/b2c | Segment churn | %/mo | shocks only (×1.2, ×1.5, ×2) | no | none in XBRL; ChartMogul benchmarks (**range check only**) |
| interest_rate | Macro rate | % | +1.5 shock, +4 rate-hike | yes (FRED, not EDGAR) | out of EDGAR scope |
| consumer_confidence | Demand index | 0–200 | −20 shock, −25/−10 cascades, +2 recovery | yes (not EDGAR) | out of scope |
| competitors | Direct competitors | count | +1 entry, +3 surge | partially | none |
| product_quality | Quality score | 0–1 | R&D saturating gain | no | none (**unavailable**) |
| price | ARPU | $/user/mo | price action, −10%/−25% shock cuts | no | none (**unavailable**) |
| headcount | FTEs | count | hiring action | annual 10-K prose only | **not usable quarterly** |
| monthly_burn | Fixed opex | $/mo | None in research runs → headcount × $8,000 | yes | opex from income statement (**proxy**) |
| valuation_multiple | Revenue multiple | × | ×0.85/×0.6 shocks, mean-revert to 10 | market data, not EDGAR | out of scope |
| unemployment | Macro | % | +1 shock, +4 recession, +0.5 cascade | yes (BLS) | out of scope |
| innovation_factor | R&D efficiency / scarring | 0–1 | −5%/mo in depression, +0.003 recovery, R&D gain | no | none |
| months_in_depression | Hysteresis counter | months | confidence < 50 counter | no | none |

Derived/logged: rule_of_40 (growth% + margin%, where "margin" = −burn/MRR), reward
(MRR/1M − threshold penalties), runway (cash/burn), LTV:CAC.

**Action space** (all four submitted every month):
marketing {spend $, channel ppc|brand} · hiring {hires, cost_per_employee} ·
product {r_and_d_spend $} · pricing {price_change_pct}.
EDGAR observability: R&D spend **direct** (R&D % of revenue), marketing spend **direct
for the 39-company panel** (S&M % of revenue), hiring annual-only, pricing/channel
**unobservable**.

## 3. Behaviour notes that shape what can be validated

1. **The composed action is heavily post-processed.** Whatever the proposals say,
   `_apply_dynamic_minimums` forces R&D ≥ max(MRR·deficit·0.10, $20k–$70k) and
   marketing ≥ max($5k, 2% MRR) [boardroom.py:741-758]; sanity bounds cap marketing at
   max(30% cash, $20k) and hires at 10; the conflict resolver cuts marketing → hiring →
   R&D when the plan exceeds cash. Policy differences therefore express themselves
   *within* a corridor, not freely. Any "action-effect" test must measure effects of the
   post-processed action.
2. **The LLM's only lever in the headline policies is the brief.** oracle_v1/v3/v4
   proposals are rule-based (`agents/proposal_agents.py` wraps `baseline_agents`; the LLM
   contributes rationale *text* only). The brief moves actions two ways: WeightAdapter
   (weights barely matter — see below) and ActionModifier (multiplicative: marketing
   ×0.3–1.56, R&D ×0.81–2.4, hiring cap 0–2) [oracle/action_modifier.py]. **The
   brief→action mapping is deterministic and enumerable**, which makes agent
   state-responsiveness testable cheaply: state → prompt → brief (LLM) → fixed modifier.
3. **Score weights are near-decorative.** Proposal scores/weights feed `p.confidence`,
   but the final action is composed by *fixed role* (CMO marketing, CFO hiring/pricing,
   CPO R&D scaled by innovation deficit) [boardroom.py:196-220] — confidence is recorded,
   never used to select among proposals. The WeightAdapter path affects nothing
   downstream except the R&D scaling via `global_innov_score` (taken from proposals[0]'s
   score vector, which is state-only). Documented behaviour ("negotiation", "consensus")
   overstates the mechanism: `NegotiationState.consensus_reached` is set to True
   unconditionally.
4. **Reward is misaligned with headline metrics** and the thesis does not use it as the
   headline (correctly): oracle policies show *worse* mean total_reward than boardroom in
   both recorded runs while beating it on survival/MRR/Rule-of-40 (evidence_audit.md §2.8).
   The validation treats reward as diagnostic only.
5. **oracle_v3 ≈ oracle_v4 in the recorded CONFIRMATION run to 9+ digits** (medians
   identical). oracle_v4 without an active causal-graph shock context degenerates to v3
   behaviour. No claim of a v3/v4 difference is supportable from recorded data.
6. **oracle_v4_causal_hetero (true LLM proposals) is the only policy where an LLM
   composes actions** — and in its recorded 20-episode dev run it *underperformed*
   (survival 0.75 vs 1.0 for oracle_v3). It requires Ollama + Neo4j and ~45 LLM
   calls/episode.
7. **Product counterfactual machinery exists and is seed-matched:**
   `backend/whatif_service.py` rolls an arbitrary EnvState forward under
   recommended/hold/rule-based arms on shared seeds and shock tape, with
   survivor-aware banding. This is the natural chassis for the EDGAR backtest.

## 4. Action tracing — do actions reach the state?

Traced end-to-end (proposal → compose → modifier → bounds → adapter → env):

- marketing.spend → `compute_new_mrr` Hill response → mrr, and → cash (subtracted), and
  → cac re-estimate. **Effective.** Legacy curve is *absolute-dollar* (β up to $50k/$100k
  new MRR per month regardless of company size) — at the $50k-MRR research anchor this is
  a strong lever; at founder/EDGAR scale it is nonsense unless `scale_aware_marketing` is on.
- product.r_and_d_spend → `apply_innovation_investment` gain × (1 − innovation_factor).
  **Legacy path is a near-dead lever in normal conditions**: innovation_factor starts at
  1.0, so the multiplier is 0 until depression scarring lowers it. R&D still moves
  `compute_expansion_mrr` (upsell multiplier ≤ 1.5×). The code's own comment calls the
  founder-scale version of this out; the research path retains the defect by design
  (recorded-run compatibility). Any claimed "R&D lever" in research runs operates almost
  entirely through the expansion term and the post-scarring repair, not through quality.
- hiring.hires → cash (one-time cost) + headcount → burn (headcount × $8,000). **Effective
  and mostly harmful** in-model: headcount produces no revenue. CFO hires on runway > 24mo.
- pricing.price_change_pct → price and mrr × (1+Δ)(1+elasticity·Δ). **Effective**; with
  elasticity ~U(−0.9,−0.2), E[net effect] of small raises is positive — a modeling
  opinion, and the elasticity is recorded as *unidentified* in calibration.
- Clipping/overrides that can silence an agent: hires clipped by cash/18 affordability
  [startup_env.py:233-236] and by sanity cap 10; marketing capped at 30% of cash; R&D
  floored upward (a "cut R&D" decision below the floor is overridden); entire plan
  shrunk by the conflict resolver when cash-short.

**Conclusion: actions do change state; magnitudes differ hugely by lever** (marketing ≫
pricing > hiring(negative) > R&D(legacy, weak)). This is quantified in the Tier-1
action-effect experiment rather than asserted.

## 5. Leakage audit

| Channel | Finding |
|---|---|
| Future states/shocks in prompts | **None.** `build_prompt` uses current state, ≤5-month trend history, matured memories only [oracle/prompt_builder.py]. |
| Memory maturation | A snapshot is stored only after its 6-month outcome is realized (`_mature_pending_memories`), and `end_episode` force-matures the tail. At decision time in month t, retrievable memories were realized at ≤ t. **No look-ahead within an episode.** |
| Cross-run contamination | Chroma retrieval filters `where={"run_id": self.run_id}` [oracle/memory.py:191]. The cumulative 27,820-entry store cannot leak across runs. **But memory accumulates across episodes within one run** — later episodes benefit from earlier ones. This is the intended transfer mechanism; it must be described as such (episode order matters, episodes are not i.i.d. for the memory-bearing arms). |
| Shock-schedule learning | Shocks always land at months 24/48/72 and memories carry `source_month`; a memory-bearing policy can in principle learn the *timetable*, which no real company could. Bounded threat (memories don't label shocks), flagged as a limitation of the shock-handling claim. |
| Confirmation/test trajectories | The CONFIRMATION run re-uses seeds 0–49, overlapping FULL seeds 0–74 — it is a re-run, not an independent replication; fine as reproduction, wrong to pool. |
| EDGAR backtest | Calibration constants (ChartMogul 2023, SaaS Capital 2026) post-date the 2011–2022 initialization states used in the backtest. Standard benchmark-backtest caveat; disclosed in the backtest doc. No *company-specific* future data enters the initial state mapping. |
| Curated artifact | `primary_summary_screenshot_no_oracle_v3.csv/.md` are edited copies with the oracle_v3 row deleted (made 8 days post-run). Numbers match the canonical file; any figure sourced from them silently omits a policy. **Use the canonical `primary_summary.csv` only.** |

## 6. Documented vs implemented — mismatches worth flagging

| Documented / implied | Actual |
|---|---|
| "Boardroom negotiation / consensus" | Fixed-role composition; consensus flag hard-coded True (boardroom.py:245) |
| "Heterogeneous LLM agents" as headline | Headline policies use rule-based proposals; LLM writes briefs (+ rationale text). True LLM proposals exist only in `oracle_v4_causal_hetero`, which lost to v3 in its only recorded run |
| "R&D lever" (research profile) | Near-zero quality effect while innovation_factor = 1.0; acts via expansion upsell only |
| oracle_v4 distinct from v3 | Recorded results identical to ~9 digits absent causal-graph context |
| Reward as objective | Reward disagrees in sign with headline outcomes across policies; not usable as the success metric |
| README ("integrated app") | Accurate for the product surface; silent on the research harness, which lives in `simulation_runner.py` + `experiments/` |
| gym observation space | Unused by every policy; EnvState is passed directly |

## 7. Existing experimental record (verified on disk)

- **FULL** thesis run: 75 seeds × {boardroom, oracle_v1, oracle_v3}, freq 10, legacy RNG.
  Full monthly/action/retrieval traces. Headline: survival 97.3/98.7/98.7%, mean final
  MRR $1.39M / $2.35M / $2.25M, Mann-Whitney vs boardroom significant on post-shock
  Rule-of-40 (p≈1e-10) and final MRR (p≈0.01–0.02).
- **CONFIRMATION**: 50 seeds × 5 policies (adds oracle_v4, oracle_v4_causal), freq 5.
  No monthly trace exported.
- **V4-DEV**: 20-episode oracle_v3 vs oracle_v4_causal_hetero (survival 1.00 vs 0.75).
  The hetero block exists only as an uncommitted working-tree file per the July audit;
  on this branch it is present in `outputs/oracle_v4_compare_final_v2/`.
- 55/55 core tests pass on this branch (determinism, stats machinery, calibration,
  API integration).
- Runtime measured on this machine: heuristic/boardroom ≈ 0.01–0.02 s/episode (120
  months). LLM-bearing policies need Ollama (installed, llama3.1:8b pulled, server
  currently down); recorded runs averaged ~28–36 LLM calls/episode.
