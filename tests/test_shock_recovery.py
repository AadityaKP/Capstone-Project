"""S11: shock_recovery flag tests. Default leaves trajectories byte-identical;
mean_revert returns ~87.5% of the hard-shock price/churn damage within 9
months and touches nothing else."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation/calibration"))

from backtest_lib import NOOP
from env.startup_env import StartupEnv


def _run(config, months=40, seed=0):
    env = StartupEnv(initial_config=config)
    env.reset(seed=seed)
    path = []
    for _ in range(months):
        _, _, term, trunc, _ = env.step(deepcopy(NOOP))
        path.append((env.state.mrr, env.state.cash, env.state.price,
                     env.state.churn_smb))
        if term or trunc:
            break
    return env, path


def test_default_is_byte_identical():
    _, a = _run({"deterministic_rng": True})
    _, b = _run({"deterministic_rng": True, "shock_recovery": "none"})
    assert a == b


def test_mean_revert_half_life_on_competitor_surge():
    # seed 0 -> shock_cycle[0] = competitor_surge at month 24:
    # price *= 0.75, churn_smb *= 1.5. With mean_revert, the recorded deltas
    # decay with a 3-month half-life: 12.5% remaining after 9 months.
    env, _ = _run({"deterministic_rng": True, "shock_recovery": "mean_revert"},
                  months=25, seed=0)
    dev = dict(env._shock_deviations)
    assert "price" in dev and dev["price"] < 0        # price was cut
    assert "churn_smb" in dev and dev["churn_smb"] > 0
    d0 = dev["price"]
    for _ in range(9):
        env.step(deepcopy(NOOP))
    remaining = env._shock_deviations.get("price", 0.0)
    assert remaining / d0 == pytest.approx(0.5 ** 3, rel=1e-9)


def test_mean_revert_restores_price_toward_pre_shock():
    # with no competitive entries in the way the flag should leave price
    # strictly higher after recovery than without the flag
    base_cfg = {"deterministic_rng": True}
    mr_cfg = {"deterministic_rng": True, "shock_recovery": "mean_revert"}
    _, a = _run(base_cfg, months=40, seed=0)
    _, b = _run(mr_cfg, months=40, seed=0)
    price_none = a[-1][2]
    price_mr = b[-1][2]
    assert price_mr > price_none
    # identical world before the first shock
    assert a[:24] == b[:24]
