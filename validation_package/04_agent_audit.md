# Agent audit — every policy, what it actually does

Companion to `validation/system_audit.md` §3–4. All paths relative to repo root.

## Policy inventory (as buildable by `simulation_runner._build_agent_for_policy`)

| Policy | Inputs | Action mechanism | LLM? | External services | Downstream effect | Failure modes |
|---|---|---|---|---|---|---|
| `heuristic` | EnvState | CFO+CMO+CPO rule agents merged directly (no boardroom post-processing) | no | none | direct env action | fixed dollar tiers tuned to $50k MRR; no floors, no conflict resolver — can overspend |
| `random` | none | uniform draws per bundle | no | none | direct env action | draws from **global** RNG → perturbs the world in legacy mode; only comparable under `deterministic_rng` |
| `boardroom` | EnvState | 3 rule proposals → compose → floors/bounds/resolver | no | none | corridor-constrained action | R&D floor forces spend regardless of proposals |
| `boardroom_oracle` / `oracle_v1` | EnvState (+brief) | boardroom + LLM brief → WeightAdapter + ActionModifier | brief only | Ollama | marketing ×0.3–1.56, R&D ×0.81–2.4, hire cap 0–2 | falls back to neutral brief on LLM failure (parse_ok=False) — silently becomes ~boardroom |
| `oracle_v1_no_modifier` | as v1 | weights-only (NoOpActionModifier) | brief only | Ollama | near-none (weights are near-decorative, see system audit §3.3) | ablation arm; expected ≈ boardroom |
| `oracle_v3` | + trend context + ≤3 Chroma memories | as v1, brief conditioned on memories | brief only | Ollama + Chroma | same modifier corridor | memory quality depends on episode order within run |
| `oracle_v3_no_memory` | as v3 minus memories | — | brief only | Ollama | — | retrieval ablation arm |
| `oracle_v4` | as v3 | — | brief only | Ollama + Chroma | **empirically identical to v3** absent graph context | do not claim v4 ≠ v3 |
| `oracle_v4_causal` | + Neo4j graph context under active shock | as v3 + causal R&D cap | brief only | + Neo4j | graph context only reaches the prompt when a shock is active | Neo4j down → silently degrades to v4 |
| `oracle_v4_causal_no_memory` | graph, no Chroma | — | brief only | Ollama + Neo4j | semantic-only ablation arm | graph priors partly hand-seeded |
| `oracle_v4_causal_hetero` | + role causal contexts | **BatchedCausalProposalGenerator: LLM writes the proposals** | proposals + brief | Ollama + Neo4j + Chroma | only policy where an LLM composes the plan | recorded run: 0.75 survival vs 1.00 for v3; parser fallback to rule proposals on failure |
| (product) `hold` / `recommended` / `rule_based` | EnvState | whatif_service arms: fixed bundle held, or scaled heuristic | no | none | direct | product-profile only |

## Where each rule agent is state-responsive (testable predictions)

- **CFO** (`agents/baseline_agents.py:23`): hires 1 iff runway > 24mo AND LTV/CAC ≥ 3;
  price +5% iff LTV/CAC < 3. → Prediction: hiring switches OFF as runway crosses 24.
- **CMO** (:48): marketing $20k/$10k/$2k by LTV/CAC bands (>4 / >2 / else); channel ppc
  iff confidence < 90. → Prediction: spend is a step function of LTV/CAC.
- **CPO** (:68): R&D $15k/$8k/$3k by avg churn bands (>4% / >2% / else); halved when
  cash < $200k. → Prediction: R&D steps up with churn, down with low cash.
- **Boardroom overlay**: R&D floor scales with innovation deficit; conflict resolver
  cuts marketing → hiring → R&D as cash tightens. → Prediction: as cash → 0, composed
  marketing → floor then → 0 before R&D goes to its protected minimum.
- **Oracle overlay** (`oracle/action_modifier.py`): risk CRITICAL ⇒ marketing ×0.3,
  R&D ×1.5, hires 0; growth COLLAPSING ⇒ marketing ×0.3; innovation CRITICAL ⇒ R&D ×1.6.
  → Prediction: brief severity monotonically decreases marketing and hiring, increases R&D.
  The LLM's contribution is exactly the state→brief mapping; the modifier arithmetic is fixed.

## What the recorded evidence already shows (and does not)

Shows (FULL n=75/arm, CONFIRMATION n=50/arm, legacy RNG but world shared across
non-drawing policies — see system audit §1):
- oracle_v1/v3 > boardroom on final MRR (p≈0.01–0.02) and post-shock Rule-of-40 (p≈1e-8…1e-10);
  recovery rate 76–80% vs 67–69%.
- No significant recovery-time difference in CONFIRMATION (p 0.18–0.82).

Does not show:
- any comparison against **no-action, random, or plain heuristic** (never run in a headline experiment);
- v4/v3 separation; retrieval value (v3 vs v3_no_memory designed but not present in
  recorded headline outputs); any per-decision regret; any external validity.

These gaps define Tier 1 of the validation plan.
