"""Phase 3 regression gate. Must pass before Phase 4 (protocol rule 3).

Part A (legacy): re-run A2 (noop/random/heuristic/boardroom x 10 seeds x 120
months, deterministic_rng, legacy flags) and diff per-seed episode outcomes
against the RECORDED validation/results/policy_comparison_episodes.csv - must
match exactly. Also re-aggregate an E1-style median quarterly growth from the
fresh replay vs the recorded decision_log.csv - must match exactly.

Part B (v2 research scale): same suite with marketing_curve="v2" (fitted),
corridor="scale_aware" for heuristic/boardroom. financing_enabled stays False
at research scale (the F2 config disable; documented). Reports:
  - A2 ordering (STOP if boardroom stops beating noop on median final MRR)
  - E4 discretionary-spend ratio (expected to improve toward EDGAR bands)
  - E1 median quarterly growth under v2

Writes p3_regression_gate.csv (per-seed outcomes, physics_version column) and
p3_gate_summary.json. Exits non-zero if the exact-match gate fails.
"""
from __future__ import annotations

import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_lib import CAL_DIR, NOOP, ROOT  # noqa: E402

from agents.adapter import ActionAdapter  # noqa: E402
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent  # noqa: E402
from agents.proposal_agents import (CFOProposalAgent, CMOProposalAgent,  # noqa: E402
                                    CPOProposalAgent)
from boardroom.boardroom import Boardroom  # noqa: E402
from env.startup_env import StartupEnv  # noqa: E402
from simulation_runner import BoardroomAgent, RandomBundleAgent  # noqa: E402

N_SEEDS = 10
MAX_MONTHS = 120
POLICIES = ["noop", "random", "heuristic", "boardroom"]


def make_policy(policy: str, version: str, seed: int):
    corridor = "scale_aware" if version == "v2" else "legacy"
    if policy == "random":
        return RandomBundleAgent()
    if policy == "boardroom":
        if version == "v2":
            board = Boardroom(
                [CFOProposalAgent(corridor=corridor),
                 CMOProposalAgent(corridor=corridor),
                 CPOProposalAgent(corridor=corridor)],
                use_oracle=False, corridor=corridor)
            board.start_episode(seed)
            return board
        agent = BoardroomAgent(oracle_mode="none")
        agent.start_episode(seed)
        return agent
    if policy == "heuristic":
        return (CFOAgent(corridor=corridor), CMOAgent(corridor=corridor),
                CPOAgent(corridor=corridor))
    return None


def get_action(policy: str, state, agent):
    if policy == "noop":
        return deepcopy(NOOP)
    if policy == "heuristic":
        action = {}
        for a in agent:
            action.update(a.act(state))
        return action
    if isinstance(agent, Boardroom):
        return agent.decide(state)
    return agent.get_action(state)


def run_episode(policy: str, seed: int, version: str) -> dict:
    config = {"deterministic_rng": True}
    if version == "v2":
        # fitted rate via the module constant; financing stays off at research
        # scale (the F2 config disable)
        config.update({"marketing_curve": "v2",
                       "competitive_entry": "scale_neutral"})
    env = StartupEnv(initial_config=config)
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    agent = make_policy(policy, version, seed)

    r40_hist, months, spend_ratio = [], [], []
    mrr_path = []
    terminated = truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        mrr_before = env.state.mrr
        action = ActionAdapter.translate_action(get_action(policy, env.state, agent))
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        r40_hist.append(info["rule_of_40"])
        mrr_path.append(env.state.mrr)
        spend_ratio.append((action["marketing"]["spend"]
                            + action["product"]["r_and_d_spend"]) / max(1.0, mrr_before))
    s = env.state
    q = [sum(mrr_path[i:i + 3]) for i in range(0, len(mrr_path) - len(mrr_path) % 3, 3)]
    growth = [q[i] / q[i - 1] - 1 for i in range(1, len(q)) if q[i - 1] > 0]
    return dict(physics_version=version, policy=policy, seed=seed,
                steps=s.months_elapsed, survived=int(not terminated),
                final_mrr=s.mrr, final_cash=s.cash,
                avg_rule40=float(np.mean(r40_hist)),
                total_reward=total_reward,
                median_qoq_growth=float(np.median(growth)) if growth else np.nan,
                median_spend_ratio=float(np.median(spend_ratio)))


def main() -> None:
    rows = []
    for version in ("legacy", "v2"):
        for policy in POLICIES:
            for seed in range(N_SEEDS):
                rows.append(run_episode(policy, seed, version))
        print(f"{version}: done")
    df = pd.DataFrame(rows)
    df.to_csv(CAL_DIR / "p3_regression_gate.csv", index=False)

    # ---- Part A: exact-match gate vs recorded A2 episodes
    rec = pd.read_csv(ROOT / "validation/results/policy_comparison_episodes.csv")
    rec = rec[rec.seed < N_SEEDS]
    fresh = df[df.physics_version == "legacy"]
    merged = rec.merge(fresh, on=["policy", "seed"], suffixes=("_rec", "_new"))
    assert len(merged) == len(POLICIES) * N_SEEDS, "missing rows in comparison"
    exact = True
    for col in ("final_mrr", "final_cash", "steps", "survived", "total_reward"):
        a, b = merged[f"{col}_rec"], merged[f"{col}_new"]
        ok = np.allclose(a, b, rtol=0, atol=1e-6)
        print(f"legacy exact-match {col}: {'OK' if ok else 'MISMATCH'}")
        exact &= ok

    # E1-style aggregation from recorded decision_log vs fresh replay
    log = pd.read_csv(ROOT / "validation/agents/decision_log.csv",
                      usecols=["policy", "seed", "month", "post_mrr"])
    log = log[log.seed < N_SEEDS]
    log["quarter"] = log.month // 3
    lq = (log.groupby(["policy", "seed", "quarter"])
             .agg(qrev=("post_mrr", "sum"), n=("post_mrr", "size")).reset_index())
    lq = lq[lq.n == 3].sort_values(["policy", "seed", "quarter"])
    lq["g"] = lq.groupby(["policy", "seed"]).qrev.pct_change()
    rec_e1 = float(lq[lq.policy == "boardroom"].g.median())
    new_e1 = float(fresh[fresh.policy == "boardroom"].median_qoq_growth.median())
    # decision_log medians are per-quarter across all quarters; the replay
    # stores per-episode medians - compare the recomputed-from-log per-episode
    # median instead for an exact-form match
    lq_ep = lq[lq.policy == "boardroom"].groupby("seed").g.median()
    new_ep = fresh[fresh.policy == "boardroom"].set_index("seed").median_qoq_growth
    e1_ok = np.allclose(lq_ep.sort_index(), new_ep.sort_index(), rtol=0, atol=1e-9)
    print(f"legacy E1 per-episode median growth match: {'OK' if e1_ok else 'MISMATCH'}"
          f" (recorded overall {rec_e1:.4f}, fresh {new_e1:.4f})")
    exact &= e1_ok

    # ---- Part B: v2 research-scale report
    v2 = df[df.physics_version == "v2"]
    summary = {}
    for version, sub in df.groupby("physics_version"):
        summary[version] = {
            p: dict(survival=float(g.survived.mean()),
                    median_final_mrr=float(g.final_mrr.median()),
                    median_spend_ratio=float(g.median_spend_ratio.median()),
                    median_qoq_growth=float(g.median_qoq_growth.median()))
            for p, g in sub.groupby("policy")}

    piv = v2.pivot(index="seed", columns="policy", values="final_mrr")
    board_beats_noop = int((piv.boardroom > piv.noop).sum())
    ordering_ok = piv.boardroom.median() > piv.noop.median()
    e4_v2 = summary["v2"]["boardroom"]["median_spend_ratio"]
    e4_legacy = summary["legacy"]["boardroom"]["median_spend_ratio"]

    gate = dict(exact_match_gate="PASS" if exact else "FAIL",
                v2_boardroom_beats_noop_median=bool(ordering_ok),
                v2_boardroom_gt_noop_seeds=f"{board_beats_noop}/{N_SEEDS}",
                e4_spend_ratio_legacy=e4_legacy, e4_spend_ratio_v2=e4_v2,
                edgar_band=[0.369, 0.937], summary=summary)
    (CAL_DIR / "p3_gate_summary.json").write_text(json.dumps(gate, indent=2))
    print(json.dumps({k: v for k, v in gate.items() if k != "summary"}, indent=2))
    for version in ("legacy", "v2"):
        print(f"\n{version}:")
        for p in POLICIES:
            s = summary[version][p]
            print(f"  {p:10s} surv={s['survival']:.0%} medMRR=${s['median_final_mrr']:,.0f} "
                  f"spend%={s['median_spend_ratio']:.1%} g={s['median_qoq_growth']:.1%}")
    if not exact:
        sys.exit(1)
    if not ordering_ok:
        print("STOP: boardroom no longer beats noop at research scale under v2")
        sys.exit(2)


if __name__ == "__main__":
    main()
