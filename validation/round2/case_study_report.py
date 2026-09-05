"""S8 (part 3): case-study figure + write-up.

Figure: paired MRR / cash / Rule-of-40 paths for seed 15, months 48-72,
oracle_v3 vs oracle_v3_no_memory, decision month 60 marked - from the
RECORDED monthly traces (validation/results/a3/). The write-up quotes the
retrieved memory and brief from case_study_traces.json when the replay has
run; until then it renders the figure and the recorded-numbers half.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
A3 = ROOT / "validation/results/a3"
OUT = ROOT / "validation/round2"
FIG = ROOT / "validation/figures/review"
SEED = 15
MONTH = 60
LO, HI = 48, 72


def main() -> None:
    v3 = pd.read_csv(A3 / "monthly_oracle_v3.csv")
    nm = pd.read_csv(A3 / "monthly_oracle_v3_no_memory.csv")
    v3 = v3[(v3.seed == SEED) & v3.month.between(LO, HI)]
    nm = nm[(nm.seed == SEED) & nm.month.between(LO, HI)]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, title, fmt in [
            (axes[0], "mrr", "MRR", "${x:,.0f}"),
            (axes[1], "cash", "Cash", "${x:,.0f}"),
            (axes[2], "rule_of_40", "Rule of 40", "{x:.0f}")]:
        ax.plot(v3.month, v3[col], label="oracle_v3 (memory)", lw=1.8)
        ax.plot(nm.month, nm[col], label="oracle_v3_no_memory", lw=1.8, ls="--")
        ax.axvline(MONTH, color="red", lw=1, ls=":", label=f"decision month {MONTH}")
        ax.set_title(title)
        ax.set_xlabel("month")
        if ax is axes[0]:
            ax.legend(fontsize=7)
    fig.suptitle(f"Case study: seed {SEED} - retrieval-driven divergence "
                 f"(selected by the frozen ranking rule)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"f8_case_study_seed{SEED}.png", dpi=150)
    print(f"figure -> {FIG}/f8_case_study_seed{SEED}.png")

    traces_path = OUT / "case_study_traces.json"
    quotes = ""
    if traces_path.exists():
        tr = json.loads(traces_path.read_text())
        for policy in ("oracle_v3", "oracle_v3_no_memory"):
            months = {m["month"]: m for m in tr[policy]["months"]}
            m = months.get(MONTH) or months.get(min(months))
            brief = m.get("brief") or {}
            quotes += f"\n### {policy} at month {m['month']} " \
                      f"(replay fidelity: rel diff {tr[policy]['fidelity_rel_diff']:.2e})\n\n"
            quotes += (f"- Brief: risk={brief.get('risk_level')}, "
                       f"growth={brief.get('growth_outlook')}, "
                       f"innovation={brief.get('innovation_urgency')}, "
                       f"expected={brief.get('expected_outcome')}, "
                       f"confidence={brief.get('confidence')} "
                       f"(source: {m.get('brief_source')})\n")
            pre = (m.get("pre_modifier_action") or {})
            post = (m.get("post_modifier_action") or {})
            quotes += (f"- Marketing: {pre.get('marketing', {}).get('spend', 0):,.0f} "
                       f"-> {post.get('marketing', {}).get('spend', 0):,.0f}; "
                       f"R&D: {pre.get('product', {}).get('r_and_d_spend', 0):,.0f} "
                       f"-> {post.get('product', {}).get('r_and_d_spend', 0):,.0f}\n")
            for mem in (m.get("retrieved_memories") or [])[:3]:
                doc = (mem.get("document") or "").replace("\n", " ")
                quotes += (f"- Retrieved (weight {mem.get('memory_weight', 0):.3f}): "
                           f"“{doc}”\n")

    cand = pd.read_csv(OUT / "case_study_candidates.csv")
    top = cand.iloc[0]
    md = f"""# Case study: episodic retrieval changing a decision (seed {SEED}, month {MONTH})

**Selection rule (frozen before looking):** among all (seed, month) points in
the recorded A3 live replication where (1) the brief label differed between
oracle_v3 and oracle_v3_no_memory, (2) marketing or R&D spend differed by
>20%, and (3) the memory arm's MRR was higher 6 months later, rank by the
6-months-later MRR advantage. {len(cand)} points qualified; this is the top
one. No cherry-picking beyond this rule.

**Recorded numbers.** At month {MONTH} the memory arm read risk
{top.risk_v3} where the no-memory arm read {top.risk_nm}; marketing spend
{top.mkt_v3:,.0f} vs {top.mkt_nm:,.0f} ({top.spend_divergence:.0%}
divergence); {int(top.memory_count_v3)} memories retrieved. Six months later
the memory arm's MRR is {top.mrr_adv_6mo_pct:.1f}% higher.

**Figure:** `validation/figures/review/f8_case_study_seed{SEED}.png`
(paired MRR / cash / Rule-of-40, months {LO}-{HI}, decision month marked).

## Decision-level evidence (from the fidelity-checked replay)
{quotes if quotes else "_(pending: run case_study_replay.py to fill quotes)_"}

**Caveat.** One episode, selected for a positive outcome by a stated rule;
this illustrates the mechanism (retrieval -> brief -> ActionModifier ->
spend), it does not quantify it - the quantification is the paired A6
ablation (+$37.9k mean, ~3% of the oracle layer's gain).
"""
    (OUT / "case_study.md").write_text(md, encoding="utf-8")
    print("write-up -> validation/round2/case_study.md")


if __name__ == "__main__":
    main()
