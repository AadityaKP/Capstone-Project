"""Addendum A decomposition arms (PROTOCOL_addendum_A.md; S12). LLM job - queued.

One arm per invocation: python validation/round2/a3_decomp.py {da,db,dc,dd,l1}

  da  oracle_v3_no_modifier, v2 physics          -> a3_decomp_da.csv
  db  oracle_v3 + modifier_bound="tier", v2      -> a3_decomp_db.csv
  dc  boardroom + oracle_v3, v2 physics +
      shock_recovery="mean_revert" (both new)    -> a3_decomp_dc.csv
  dd  oracle_v3, qwen2.5:7b-instruct, v2         -> a3_decomp_dd.csv
  l1  oracle_v3, qwen2.5:7b-instruct, legacy     -> a3_decomp_l1.csv

Same harness as the arms these pair against: run_simulation, seeds 0-19,
oracle_frequency=10, brief v1. The v2 arms reuse the exact recorded config of
a3_oracle_value_v2phys.csv (see a3_v2phys/meta_*.json): marketing_curve="v2",
competitive_entry="scale_neutral", corridor="scale_aware", financing OFF.
The addendum's common-configuration line says financing_enabled=True, but the
recorded boardroom arms it pairs against ran with financing off; a paired
design on the same seeds requires matching the recorded config, so financing
stays off (deviation recorded in LOG.md before any run). Gate analysis lives
in gates_decomp.py (run once, in the morning).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.llm_client import create_llm_client  # noqa: E402
from oracle.oracle import Oracle  # noqa: E402
from simulation_runner import run_simulation  # noqa: E402

RESULTS = ROOT / "validation/results"
N_EPISODES = 20
FREQ = 10
QWEN = "qwen2.5:7b-instruct"
V2_ENV = {"deterministic_rng": True, "marketing_curve": "v2",
          "competitive_entry": "scale_neutral"}
V2_AGENT = {"corridor": "scale_aware"}
LEGACY_ENV = {"deterministic_rng": True}


def _qwen_oracle() -> Oracle:
    # brief v1 / memory retrieval on by default: identical to the recorded
    # oracle_v3 arms except for the LLM behind the brief.
    return Oracle(mode="oracle_v3", llm=create_llm_client("ollama", QWEN))


def _ollama_ok(model: str) -> bool:
    # Fail fast instead of degrading: LLMClient returns "" when Ollama is
    # unreachable, and the parser then falls back to a neutral brief, which
    # would produce complete-looking CSVs for a dead-LLM arm.
    resp = create_llm_client("ollama", model).complete(
        "You are a health check.", "Reply with the single word OK.")
    return bool(resp.strip())


# arm -> ordered (policy, output label) runs, env config, agent overrides.
# Overrides are built lazily so importing this module constructs no Oracle.
ARMS = {
    "da": dict(runs=[("oracle_v3_no_modifier", "oracle_v3_no_modifier")],
               env=V2_ENV, agent=lambda: dict(V2_AGENT),
               physics="v2", model="llama3.1:8b"),
    "db": dict(runs=[("oracle_v3", "oracle_v3_tier_bound")],
               env=V2_ENV,
               agent=lambda: {**V2_AGENT, "modifier_bound": "tier"},
               physics="v2", model="llama3.1:8b"),
    "dc": dict(runs=[("boardroom", "boardroom_mr"),
                     ("oracle_v3", "oracle_v3_mr")],
               env={**V2_ENV, "shock_recovery": "mean_revert"},
               agent=lambda: dict(V2_AGENT),
               physics="v2", model="llama3.1:8b"),
    "dd": dict(runs=[("oracle_v3", "oracle_v3_qwen")],
               env=V2_ENV,
               agent=lambda: {**V2_AGENT, "oracle_instance": _qwen_oracle()},
               physics="v2", model=QWEN),
    "l1": dict(runs=[("oracle_v3", "oracle_v3_qwen_legacy")],
               env=LEGACY_ENV,
               agent=lambda: {"oracle_instance": _qwen_oracle()},
               physics="legacy", model=QWEN),
}


def main(arm: str, num_episodes: int = N_EPISODES,
         out_dir: Path = RESULTS) -> None:
    spec = ARMS[arm]
    if not _ollama_ok(spec["model"]):
        sys.exit(f"Ollama preflight FAILED for {spec['model']}; "
                 f"aborting arm {arm} before any episode runs")
    print(f"Ollama preflight OK ({spec['model']})", flush=True)
    out_csv = out_dir / f"a3_decomp_{arm}.csv"
    # Arm-level resume (same contract as a3_random_shock.py): a label already
    # complete in the output CSV is kept and skipped; partial labels are
    # discarded and re-run whole (oracle memory accrues across episodes
    # within an arm).
    frames, done = [], set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv)
        done = {p for p, g in prev.groupby("policy")
                if g.seed.nunique() >= num_episodes}
        if done:
            frames.append(prev[prev.policy.isin(done)])
            print(f"resuming; complete labels kept: {sorted(done)}", flush=True)
    for policy, label in spec["runs"]:
        if label in done:
            continue
        t0 = time.time()
        overrides = spec["agent"]()
        df, _ = run_simulation(
            policy=policy, num_episodes=num_episodes, seed_start=0,
            oracle_frequency=FREQ, environment_config=dict(spec["env"]),
            oracle_overrides=overrides,
            return_action_trace=True, return_monthly_trace=False,
        )
        df["policy"] = label
        df["physics_version"] = spec["physics"]
        frames.append(df)
        meta = dict(arm=arm, policy=policy, label=label,
                    episodes=num_episodes, freq=FREQ, env=spec["env"],
                    agent={k: v for k, v in overrides.items()
                           if k != "oracle_instance"},
                    model=spec["model"],
                    wall_seconds=round(time.time() - t0, 1),
                    llm_calls_total=int(df.llm_calls.sum())
                    if "llm_calls" in df else 0)
        (out_dir / f"a3_decomp_{arm}_meta_{label}.json").write_text(
            json.dumps(meta))
        pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)
        print(f"DONE {label}: {meta}", flush=True)
    if not _ollama_ok(spec["model"]):
        sys.exit(f"Ollama postflight FAILED for {spec['model']}: arm {arm} "
                 "is SUSPECT - an outage mid-run silently degrades briefs "
                 "to the neutral fallback; check the log before trusting it")
    print(f"Ollama postflight OK ({spec['model']})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ARMS:
        sys.exit(f"usage: a3_decomp.py {{{','.join(ARMS)}}}")
    main(sys.argv[1])
