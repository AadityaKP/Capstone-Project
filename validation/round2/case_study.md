# Case study: episodic retrieval changing a decision (seed 15, month 60)

**Selection rule (frozen before looking):** among all (seed, month) points in
the recorded A3 live replication where (1) the brief label differed between
oracle_v3 and oracle_v3_no_memory, (2) marketing or R&D spend differed by
>20%, and (3) the memory arm's MRR was higher 6 months later, rank by the
6-months-later MRR advantage. 115 points qualified; this is the top
one. No cherry-picking beyond this rule.

**Recorded numbers.** At month 60 the memory arm read risk
LOW where the no-memory arm read MEDIUM; marketing spend
32,760 vs 22,100 (48%
divergence); 3 memories retrieved. Six months later
the memory arm's MRR is 10.4% higher.

**Figure:** `validation/figures/review/f8_case_study_seed15.png`
(paired MRR / cash / Rule-of-40, months 48-72, decision month marked).

## Decision-level evidence (from the fidelity-checked replay)

### oracle_v3 at month 60 (replay fidelity: rel diff 1.43e-03)

- Brief: risk=LOW, growth=ACCELERATING, innovation=LOW, expected=GROWTH, confidence=0.85 (source: llm)
- Marketing: 20,000 -> 32,760; R&D: 20,999 -> 17,860
- Retrieved (weight 0.884): “Phase: GROWTH | Churn: HIGH | Innovation: HEALTHY Episode month 33: MRR 1,089,802, avg churn 0.054, innovation 1.000. Trends were MRR INCREASING, innovation FLAT, churn FLAT. After 6 months the realized outcome was GROWTH.”
- Retrieved (weight 0.881): “Phase: GROWTH | Churn: HIGH | Innovation: HEALTHY Episode month 30: MRR 752,192, avg churn 0.074, innovation 1.000. Trends were MRR INCREASING, innovation FLAT, churn INCREASING. After 6 months the realized outcome was GROWTH.”
- Retrieved (weight 0.880): “Phase: GROWTH | Churn: HIGH | Innovation: HEALTHY Episode month 48: MRR 1,558,560, avg churn 0.074, innovation 1.000. Trends were MRR INCREASING, innovation FLAT, churn FLAT. After 6 months the realized outcome was GROWTH.”

### oracle_v3_no_memory at month 60 (replay fidelity: rel diff 0.00e+00)

- Brief: risk=MEDIUM, growth=ACCELERATING, innovation=MEDIUM, expected=GROWTH, confidence=0.85 (source: llm)
- Marketing: 20,000 -> 22,100; R&D: 20,999 -> 23,099


**Caveat.** One episode, selected for a positive outcome by a stated rule;
this illustrates the mechanism (retrieval -> brief -> ActionModifier ->
spend), it does not quantify it - the quantification is the paired A6
ablation (+$37.9k mean, ~3% of the oracle layer's gain).
