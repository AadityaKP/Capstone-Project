"""B1 gate (BRIEF_V2_SPEC.md): A4 level sweeps for brief v1 / v2a / v2b.

Same sweep design and monotonicity statistics as the pre-declared A4 battery
(validation/analysis/a4_state_responsiveness.py), plus the fifth new sweep
LTV:CAC ↓ (reported; the gate counts only the original four). llama3.1:8b,
temperature 0, oracle_v1 prompt, no memory.

PASS rule (frozen): >= 3 of the original 4 sweeps move (|rho| >= 0.5 in the
expected direction) for v2a; if v2a fails and v2b passes, v2b is the
candidate (floor share reported); if neither, brief v2 is FAIL.

Writes validation/results/a4_level_sweeps_bv2.csv and prints the verdict.
Optionally takes a model name argument (used later for the second-LLM
sensitivity run, which writes elsewhere).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from oracle.action_modifier import ActionModifier  # noqa: E402
from oracle.oracle import Oracle  # noqa: E402

sys.path.insert(0, str(ROOT / "validation/analysis"))
from a4_state_responsiveness import ORD, SWEEPS, base_state, brief_row  # noqa: E402

OUT = ROOT / "validation/results"

SWEEPS_R2 = dict(SWEEPS)
SWEEPS_R2["ltv_cac_down"] = dict(
    levels=[500.0, 400.0, 300.0, 200.0, 100.0, 50.0],  # cac=100 -> ratio 5..0.5
    make=lambda v: base_state(ltv=float(v)),
    expect="risk_up_as_ratio_drops")

VARIANTS = {
    "v1": dict(brief_version="v1", brief_guardrails=False),
    "v2a": dict(brief_version="v2", brief_guardrails=False),
    "v2b": dict(brief_version="v2", brief_guardrails=True),
}


def run_variant(variant: str, cfg: dict, model=None) -> pd.DataFrame:
    kwargs = {}
    if model is not None:
        from agents.llm_client import create_llm_client
        kwargs["llm"] = create_llm_client("ollama", model)
    oracle = Oracle(mode="oracle_v1", memory_store=None,
                    enable_memory_retrieval=False, **cfg, **kwargs)
    mod = ActionModifier()
    rows = []
    for tag, sweep in SWEEPS_R2.items():
        for v in sweep["levels"]:
            row = brief_row(oracle, mod, sweep["make"](v), tag, v)
            row["variant"] = variant
            row["floors"] = "; ".join(oracle.last_floor_applied)
            rows.append(row)
            print(f"  [{variant}] {tag} level={v}: {row['risk']}/{row['growth']}"
                  f" mkt x{row['mkt_mult']:.2f} rd x{row['rd_mult']:.2f}"
                  + (f" FLOORS[{row['floors']}]" if row["floors"] else ""),
                  flush=True)
    return pd.DataFrame(rows)


def rho_checks(df: pd.DataFrame, variant: str) -> pd.DataFrame:
    sub_all = df[df.variant == variant]

    def rho_of(tag, col, ordinal=False, flip=False):
        sub = sub_all[sub_all.sweep == tag]
        y = sub[col].map(ORD) if ordinal else sub[col]
        x = sub.level.astype(float)
        if flip:
            x = -x
        r = spearmanr(x, y).statistic
        return float(r) if np.isfinite(r) else 0.0

    checks = [
        ("runway_down", "risk rises as cash falls", True,
         rho_of("runway_down", "risk", ordinal=True, flip=True)),
        ("churn_up", "R&D multiplier rises with churn", True,
         rho_of("churn_up", "rd_mult")),
        ("confidence_down", "marketing multiplier falls with confidence", True,
         rho_of("confidence_down", "mkt_mult")),
        ("competitors_up", "risk rises with competitors", True,
         rho_of("competitors_up", "risk", ordinal=True)),
        ("ltv_cac_down", "risk rises as LTV:CAC falls (new, reported)", False,
         rho_of("ltv_cac_down", "risk", ordinal=True, flip=True)),
    ]
    out = pd.DataFrame(checks, columns=["sweep", "expectation", "gated", "rho"])
    out["variant"] = variant
    out["moved"] = out.rho >= 0.5
    return out


def main(model=None) -> None:
    frames, checks = [], []
    for variant, cfg in VARIANTS.items():
        print(f"--- {variant} ---", flush=True)
        frames.append(run_variant(variant, cfg, model=model))
        checks.append(rho_checks(frames[-1], variant))
    df = pd.concat(frames, ignore_index=True)
    cdf = pd.concat(checks, ignore_index=True)
    if model is None:
        df.to_csv(OUT / "a4_level_sweeps_bv2.csv", index=False)
        cdf.to_csv(OUT / "a4_level_sweeps_bv2_checks.csv", index=False)

    print("\n", cdf.to_string(index=False))
    verdicts = {}
    for variant in VARIANTS:
        sub = cdf[(cdf.variant == variant) & cdf.gated]
        verdicts[variant] = int(sub.moved.sum())
        print(f"{variant}: {verdicts[variant]}/4 gated sweeps moved")
    if verdicts["v2a"] >= 3:
        chosen = "v2a"
    elif verdicts["v2b"] >= 3:
        chosen = "v2b"
    else:
        chosen = "FAIL"
    floor_share = float((df[df.variant == "v2b"].floors != "").mean())
    print(f"\nB1 verdict: chosen variant = {chosen} "
          f"(v1 {verdicts['v1']}/4 as comparator; v2b floor share {floor_share:.0%})")


if __name__ == "__main__":
    main(model=sys.argv[1] if len(sys.argv) > 1 else None)
