"""A5: candidate-action regret for the boardroom policy (exploratory).

At 24 decision states sampled from boardroom trajectories (8 seeds x months
{6, 30, 50} - months 30 and 50 sit in the post-shock windows), we compare the
policy's own month-1 action against a 48-bundle candidate grid. Evaluation is a
one-step deviation: month 1 = candidate (or the policy's own action), months
2-12 = boardroom policy, identical seeds (10 evaluation seeds per candidate,
deterministic_rng). Outcome: terminal 12-month MRR (survival tracked).

  candidate_regret(state) = best candidate outcome - policy outcome
                            (means over matched evaluation seeds)

This is CANDIDATE regret over a 48-point grid, not global optimality - the
grid does not span the continuous action space. Exploratory per the plan:
reported as magnitudes, no pass/fail.

Writes validation/agents/candidate_regret.csv (per state) and
candidate_regret_detail.csv (per state x candidate).
"""
from __future__ import annotations

import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from agents.adapter import ActionAdapter
from env.startup_env import StartupEnv
from simulation_runner import BoardroomAgent

OUT = ROOT / "validation/agents"

SAMPLE_SEEDS = range(8)
SAMPLE_MONTHS = [6, 30, 50]
EVAL_SEEDS = range(10)
HORIZON = 12

MARKETING = [0.0, 5_000.0, 15_000.0, 40_000.0]
RND = [0.0, 10_000.0, 40_000.0]
HIRES = [0, 1]
PRICE = [0.0, 0.05]
CANDIDATES = [
    {"marketing": {"spend": m, "channel": "ppc"},
     "hiring": {"hires": h, "cost_per_employee": 10_000.0},
     "product": {"r_and_d_spend": r},
     "pricing": {"price_change_pct": p}}
    for m, r, h, p in product(MARKETING, RND, HIRES, PRICE)
]


def sample_states():
    """Full EnvState snapshots + the policy's own action at each sampled month."""
    states = []
    for seed in SAMPLE_SEEDS:
        env = StartupEnv(initial_config={"deterministic_rng": True})
        env.reset(seed=seed)
        agent = BoardroomAgent(oracle_mode="none")
        agent.start_episode(seed)
        for month in range(max(SAMPLE_MONTHS) + 1):
            action = agent.get_action(env.state)
            if month in SAMPLE_MONTHS:
                states.append(dict(sample_seed=seed, month=month,
                                   state=env.state.model_copy(deep=True),
                                   policy_action=ActionAdapter.translate_action(
                                       deepcopy(action))))
            _, _, terminated, _, _ = env.step(
                ActionAdapter.translate_action(deepcopy(action)))
            if terminated:
                break
    return states


def rollout(state, first_action, eval_seed):
    env = StartupEnv(initial_config={"deterministic_rng": True})
    env.reset(seed=eval_seed)
    env.state = state.model_copy(deep=True)
    agent = BoardroomAgent(oracle_mode="none")
    agent.start_episode(eval_seed)
    for month in range(HORIZON):
        if month == 0:
            action = deepcopy(first_action)
        else:
            action = agent.get_action(env.state)
        _, _, terminated, _, _ = env.step(ActionAdapter.translate_action(action))
        if terminated:
            return env.state.mrr, 0
    return env.state.mrr, 1


def evaluate(state, first_action):
    vals, surv = [], []
    for s in EVAL_SEEDS:
        m, alive = rollout(state, first_action, s)
        vals.append(m)
        surv.append(alive)
    return float(np.mean(vals)), float(np.mean(surv))


def main():
    samples = sample_states()
    per_state, detail = [], []
    for smp in samples:
        st = smp["state"]
        policy_mean, policy_surv = evaluate(st, smp["policy_action"])
        cand_means = []
        for ci, cand in enumerate(CANDIDATES):
            mean, surv = evaluate(st, cand)
            cand_means.append(mean)
            detail.append(dict(sample_seed=smp["sample_seed"], month=smp["month"],
                               candidate=ci,
                               mkt=cand["marketing"]["spend"],
                               rnd=cand["product"]["r_and_d_spend"],
                               hires=cand["hiring"]["hires"],
                               price=cand["pricing"]["price_change_pct"],
                               mean_final_mrr=mean, survival=surv))
        cand_means = np.array(cand_means)
        best = cand_means.max()
        regret = best - policy_mean
        rank = int((cand_means > policy_mean).sum()) + 1
        pa = smp["policy_action"]
        per_state.append(dict(
            sample_seed=smp["sample_seed"], month=smp["month"],
            state_mrr=st.mrr, state_cash=st.cash,
            policy_mkt=pa["marketing"]["spend"], policy_rnd=pa["product"]["r_and_d_spend"],
            policy_hires=pa["hiring"]["hires"],
            policy_mean_final_mrr=policy_mean, policy_survival=policy_surv,
            best_candidate_mean=best,
            best_mkt=CANDIDATES[int(cand_means.argmax())]["marketing"]["spend"],
            best_rnd=CANDIDATES[int(cand_means.argmax())]["product"]["r_and_d_spend"],
            candidate_regret=regret,
            regret_pct_of_best=regret / best if best > 0 else np.nan,
            policy_rank_of_49=rank))
        print(f"seed {smp['sample_seed']} month {smp['month']}: policy {policy_mean:,.0f} "
              f"best {best:,.0f} regret {regret:,.0f} ({regret/best:.1%}) rank {rank}/49")

    ps = pd.DataFrame(per_state)
    ps.to_csv(OUT / "candidate_regret.csv", index=False)
    pd.DataFrame(detail).to_csv(OUT / "candidate_regret_detail.csv", index=False)
    print(f"\nmedian regret {ps.candidate_regret.median():,.0f} "
          f"({ps.regret_pct_of_best.median():.1%} of best); "
          f"median policy rank {ps.policy_rank_of_49.median():.0f}/49; "
          f"regret==0 in {(ps.candidate_regret <= 0).sum()}/{len(ps)} states")


if __name__ == "__main__":
    main()
