"""RS-2x: pre-declared extension of the random-shock ablation to seeds 21-40
(PROTOCOL_addendum_A.md; S12). LLM job - queued.

oracle_v3 / oracle_v3_no_memory only (RS-2x pairs them against each other),
legacy physics, deterministic_rng, freq 10, shock_schedule="random" - the
exact harness of a3_random_shock.py on new seeds. Writes
validation/results/a3_oracle_value_rs_ext.csv (episode rows with the
schedule) and a3_rs_ext_monthly.csv; the recorded n=20 and pooled n=40
analyses live in gates_decomp.py (run once, in the morning).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import random  # noqa: E402

from agents.adapter import ActionAdapter  # noqa: E402
from agents.llm_client import create_llm_client  # noqa: E402
from env.startup_env import StartupEnv  # noqa: E402
from simulation_runner import _build_agent_for_policy  # noqa: E402

RESULTS = ROOT / "validation/results"
SEEDS = list(range(21, 41))
FREQ = 10
MODEL = "llama3.1:8b"
ENV = {"deterministic_rng": True, "shock_schedule": "random"}
POLICIES = ["oracle_v3", "oracle_v3_no_memory"]


def _ollama_ok(model: str = MODEL) -> bool:
    # Fail fast instead of degrading: LLMClient returns "" when Ollama is
    # unreachable, and the parser then falls back to a neutral brief - both
    # arms would silently collapse to identical neutral-brief trajectories.
    resp = create_llm_client("ollama", model).complete(
        "You are a health check.", "Reply with the single word OK.")
    return bool(resp.strip())


def main(seeds: list[int] | None = None, ep_path: Path | None = None,
         mo_path: Path | None = None) -> None:
    # Arm-level resume: a policy arm already complete in the episodes CSV is
    # skipped (an arm must be whole - oracle memory accrues across episodes
    # within an arm, so partial arms are discarded and re-run). Lets the job
    # survive an interrupted session without redoing finished arms.
    seeds = SEEDS if seeds is None else list(seeds)
    ep_path = ep_path or RESULTS / "a3_oracle_value_rs_ext.csv"
    mo_path = mo_path or RESULTS / "a3_rs_ext_monthly.csv"
    if not _ollama_ok():
        sys.exit(f"Ollama preflight FAILED for {MODEL}; "
                 "aborting before any episode runs")
    print(f"Ollama preflight OK ({MODEL})", flush=True)
    ep_rows, monthly_rows = [], []
    done_policies: set[str] = set()
    if ep_path.exists():
        prev = pd.read_csv(ep_path)
        done_policies = {p for p, g in prev.groupby("policy")
                         if g.seed.nunique() >= len(seeds)}
        prev = prev[prev.policy.isin(done_policies)]
        ep_rows = prev.to_dict("records")
        if mo_path.exists():
            prev_mo = pd.read_csv(mo_path)
            monthly_rows = prev_mo[prev_mo.policy.isin(done_policies)].to_dict("records")
        print(f"resuming; complete arms kept: {sorted(done_policies)}", flush=True)
    for policy in POLICIES:
        if policy in done_policies:
            continue
        t0 = time.time()
        agent = _build_agent_for_policy(policy, FREQ)
        for seed in seeds:
            env = StartupEnv(initial_config=dict(ENV))
            env.reset(seed=seed)
            if hasattr(agent, "start_episode"):
                agent.start_episode(seed)
            if hasattr(agent, "set_shock_label"):
                agent.set_shock_label(None)
            random.seed(seed)
            np.random.seed(seed)
            schedule = list(env.shock_months)
            terminated = truncated = False
            llm0 = (agent.get_episode_stats().get("llm_calls", 0)
                    if hasattr(agent, "get_episode_stats") else 0)
            while not (terminated or truncated):
                month = env.state.months_elapsed
                action = ActionAdapter.translate_action(agent.get_action(env.state))
                _, _, terminated, truncated, info = env.step(action)
                if hasattr(agent, "set_shock_label"):
                    agent.set_shock_label(info.get("shock_label"))
                monthly_rows.append(dict(policy=policy, seed=seed, month=month,
                                         rule_of_40=info["rule_of_40"],
                                         mrr=env.state.mrr,
                                         shock_label=info["shock_label"]))
            llm1 = (agent.get_episode_stats().get("llm_calls", 0)
                    if hasattr(agent, "get_episode_stats") else 0)
            ep_rows.append(dict(policy=policy, seed=seed,
                                steps=env.state.months_elapsed,
                                survived=int(not terminated),
                                final_mrr=env.state.mrr,
                                final_cash=env.state.cash,
                                shock_m1=schedule[0], shock_m2=schedule[1],
                                shock_m3=schedule[2],
                                llm_calls=max(0, llm1 - llm0)))
            print(f"{policy} seed {seed}: mrr={env.state.mrr:,.0f} "
                  f"schedule={schedule} llm={ep_rows[-1]['llm_calls']}",
                  flush=True)
            pd.DataFrame(ep_rows).to_csv(ep_path, index=False)
        # monthly rows land after each completed ARM so an interrupted session
        # keeps whole arms only (matching the episode-CSV resume contract)
        pd.DataFrame(monthly_rows).to_csv(mo_path, index=False)
        print(f"DONE {policy} in {time.time() - t0:.0f}s", flush=True)
    if not _ollama_ok():
        sys.exit(f"Ollama postflight FAILED for {MODEL}: this run is "
                 "SUSPECT - an outage mid-run silently degrades briefs to "
                 "the neutral fallback; check the log before trusting it")
    print(f"Ollama postflight OK ({MODEL})", flush=True)


if __name__ == "__main__":
    main()
