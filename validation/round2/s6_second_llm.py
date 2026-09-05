"""S6: second-LLM A4 sensitivity (reported, no gate; BRIEF_V2_SPEC.md).

Purpose: is level-blindness a property of llama3.1:8b or of the prompt?
Runs the five level sweeps for brief v1 and v2a with a second Ollama model.
(B1 FAILED, so there is no 'chosen' variant; v2a is used as the level-block
representative - the informative comparison for the question above. Recorded
as a deviation note in LOG.md.)

Combines with the llama3.1:8b rows already in a4_level_sweeps_bv2.csv ->
validation/results/a4_level_sweeps_models.csv (model column).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation/round2"))

from b1_level_sweeps import VARIANTS, rho_checks, run_variant  # noqa: E402

RESULTS = ROOT / "validation/results"
CANDIDATES = ["qwen2.5:7b-instruct", "mistral:7b-instruct", "gemma2:9b"]


def pick_model() -> str:
    out = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    for m in CANDIDATES:
        if m.split(":")[0] in out:
            return m
    raise SystemExit("no candidate model present; pull qwen2.5:7b-instruct first")


def main() -> None:
    model = pick_model()
    print(f"second model: {model}")
    frames, checks = [], []
    for variant in ("v1", "v2a"):
        df = run_variant(variant, VARIANTS[variant], model=model)
        df["model"] = model
        frames.append(df)
        c = rho_checks(df, variant)
        c["model"] = model
        checks.append(c)

    base = pd.read_csv(RESULTS / "a4_level_sweeps_bv2.csv")
    base = base[base.variant.isin(["v1", "v2a"])].copy()
    base["model"] = "llama3.1:8b"
    combined = pd.concat([base, *frames], ignore_index=True)
    combined.to_csv(RESULTS / "a4_level_sweeps_models.csv", index=False)

    base_checks = pd.read_csv(RESULTS / "a4_level_sweeps_bv2_checks.csv")
    base_checks = base_checks[base_checks.variant.isin(["v1", "v2a"])].copy()
    base_checks["model"] = "llama3.1:8b"
    all_checks = pd.concat([base_checks, *checks], ignore_index=True)
    all_checks.to_csv(RESULTS / "a4_level_sweeps_models_checks.csv", index=False)

    print("\nmoved-count per (model, variant) [gated sweeps only]:")
    g = all_checks[all_checks.gated].groupby(["model", "variant"]).moved.sum()
    print(g.to_string())


if __name__ == "__main__":
    main()
