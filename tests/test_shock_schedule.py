"""S7: shock_schedule flag tests. Fixed reproduces {24,48,72}; random is
deterministic per seed and identical across policies (schedule drawn from the
episode world RNG at reset, before any policy acts)."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation/calibration"))

from backtest_lib import NOOP
from env.startup_env import StartupEnv


def test_fixed_schedule_reproduces_recorded_timetable():
    env = StartupEnv(initial_config={"deterministic_rng": True})
    env.reset(seed=5)
    assert env.shock_months == [24, 48, 72]
    env2 = StartupEnv(initial_config={"deterministic_rng": True,
                                      "shock_schedule": "fixed"})
    env2.reset(seed=5)
    # identical trajectories: the fixed path consumes no extra draws
    for _ in range(30):
        env.step(deepcopy(NOOP))
        env2.step(deepcopy(NOOP))
    assert env.state.mrr == env2.state.mrr and env.state.cash == env2.state.cash


def test_random_schedule_deterministic_per_seed_and_spaced():
    months = {}
    for seed in (0, 1, 7):
        drawn = []
        for _ in range(2):  # same seed twice -> same schedule
            env = StartupEnv(initial_config={"deterministic_rng": True,
                                             "shock_schedule": "random"})
            env.reset(seed=seed)
            drawn.append(tuple(env.shock_months))
        assert drawn[0] == drawn[1]
        m = drawn[0]
        assert all(12 <= x <= 108 for x in m)
        assert m[1] - m[0] >= 12 and m[2] - m[1] >= 12
        months[seed] = m
    assert len(set(months.values())) > 1  # different seeds differ


def test_random_schedule_identical_across_policies():
    # two envs, same seed, different action streams -> same schedule and the
    # shock actually fires at the drawn months
    envs = []
    for spend in (0.0, 5_000.0):
        env = StartupEnv(initial_config={"deterministic_rng": True,
                                         "shock_schedule": "random"})
        env.reset(seed=3)
        envs.append((env, spend))
    assert envs[0][0].shock_months == envs[1][0].shock_months
    target = set(envs[0][0].shock_months)
    for env, spend in envs:
        hit = []
        action = deepcopy(NOOP)
        action["marketing"]["spend"] = spend
        for _ in range(109):
            _, _, term, trunc, info = env.step(deepcopy(action))
            if info["shock_label"] != "NO_SHOCK":
                hit.append(env.state.months_elapsed - 1)
            if term or trunc:
                break
        assert set(hit) == target
