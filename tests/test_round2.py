"""R2-2 tests (PROTOCOL_round2.md): hazard math, CAC worked example,
no-look-ahead, defaults reproduce round-1 behaviour."""
from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation/calibration"))

from env.startup_env import StartupEnv
from backtest_lib import NOOP
from make_eval2 import company_cac
from eval2_lib import build_state_eval2, eval2_env_config

HAZARD = {"bins": [0.0, 12.0, 24.0, 48.0],
          "h": [0.295, 0.084, 0.141, 0.100],
          "K": [26.4, 20.0, 9.8, 26.5]}


# ---------------------------------------------------------------- hazard math

def test_monthly_hazard_conversion_hand_computed():
    # q = 0.65/quarter -> h = 1 - (1-0.65)^(1/3) = 1 - 0.35^(1/3) = 0.295268
    assert 1 - (1 - 0.65) ** (1 / 3) == pytest.approx(0.295268, abs=1e-5)
    # the committed table used exactly this conversion
    import json
    hz = json.loads((ROOT / "validation/calibration/financing_hazard_r2.json").read_text())
    for r in hz["rows"]:
        assert r["h_b"] == pytest.approx(1 - (1 - r["q_b"]) ** (1 / 3), abs=1e-9)


def _opportunistic_env(cash, burn):
    return StartupEnv(initial_config={
        "deterministic_rng": True, "scheduled_shocks": False,
        "financing_enabled": True, "financing_model": "opportunistic",
        "financing_hazard": HAZARD,
        "initial_cash": cash, "monthly_burn": burn, "initial_mrr": 5_000.0})


def test_opportunistic_bin_selection_and_raise_size():
    # net burn ~ 30k - 5k*margin(1.0) = 25k; runway ~ 40k/25k = 1.6 months
    # -> bin [0,12): h=0.295, K=26.4. Over 12 draws a raise is near-certain.
    env = _opportunistic_env(40_000.0, 30_000.0)
    env.reset(seed=1)
    raised = 0.0
    for _ in range(12):
        _, _, term, trunc, info = env.step(deepcopy(NOOP))
        raised += info["financing_raise"]
        if term or trunc:
            break
    assert env.financing_events, "no raise fired in the lowest runway bin"
    ev = env.financing_events[0]
    assert ev["runway_before"] < 12.0
    assert ev["amount"] == pytest.approx(26.4 * ev["net_burn"], rel=1e-9)


def test_opportunistic_requires_hazard():
    with pytest.raises(ValueError):
        StartupEnv(initial_config={"financing_enabled": True,
                                   "financing_model": "opportunistic"})


def test_financing_default_is_rescue_and_reproduces_round1():
    base = {"deterministic_rng": True, "scheduled_shocks": False,
            "financing_enabled": True, "initial_cash": 40_000.0,
            "monthly_burn": 30_000.0, "initial_mrr": 5_000.0}
    paths = []
    for extra in ({}, {"financing_model": "rescue"}):
        env = StartupEnv(initial_config={**base, **extra})
        env.reset(seed=3)
        path = []
        for _ in range(12):
            _, _, term, trunc, info = env.step(deepcopy(NOOP))
            path.append((env.state.mrr, env.state.cash, info["financing_raise"]))
            if term or trunc:
                break
        paths.append(path)
    assert paths[0] == paths[1]


# ---------------------------------------------------------------- CAC formula

def test_company_cac_worked_example():
    # Hand-worked: quarters qi 0..8, revenue 100..180 (step 10), sm_pct 0.5.
    # At qi=8, price=250, c_m=0.027:
    #   trailing_SM   = 0.5 * (150+160+170+180)          = 330
    #   net_new_rev   = 180 - 140                         = 40
    #   churned_est   = 12 * 0.027 * mean(150..180=165)   = 53.46
    #   gross_new     = 93.46
    #   new_customers = 93.46 / 750                       = 0.1246133...
    #   CAC           = 330 / 0.1246133                   = 2648.19...
    df = pd.DataFrame(dict(
        qi=range(9), revenue=[100 + 10 * i for i in range(9)],
        sm_pct_revenue=[0.5] * 9))
    got = company_cac(df, qi=8, price=250.0)
    expected = 330.0 / ((40 + 12 * 0.027 * 165) / (3 * 250.0))
    assert got == pytest.approx(expected, rel=1e-12)
    assert got == pytest.approx(2648.19, abs=0.01)


def test_company_cac_no_look_ahead():
    # future rows must not change the result: the function may only use qi_ <= qi
    past = pd.DataFrame(dict(qi=range(9), revenue=[100 + 10 * i for i in range(9)],
                             sm_pct_revenue=[0.5] * 9))
    future = pd.DataFrame(dict(qi=[9, 10], revenue=[1e9, 1e9],
                               sm_pct_revenue=[0.99, 0.99]))
    with_future = pd.concat([past, future], ignore_index=True)
    assert company_cac(past, 8, 250.0) == company_cac(with_future, 8, 250.0)


def test_company_cac_missing_trailing_returns_none():
    df = pd.DataFrame(dict(qi=[5, 6, 7, 8], revenue=[1, 2, 3, 4],
                           sm_pct_revenue=[0.5] * 4))  # no qi-4
    assert company_cac(df, 8, 250.0) is None


# ---------------------------------------------------------------- mapping flag

def test_mapping_v1_matches_round1_state():
    ev = pd.read_csv(ROOT / "validation/calibration/eval2_states.csv")
    row = next(r for r in ev.itertuples() if r.price_assumed == 250.0)
    s1 = build_state_eval2(row, "v1")
    assert s1.cac == pytest.approx(250.0 / 0.027 / 3.0)   # round-1 identity
    assert s1.ltv == pytest.approx(250.0 / 0.027)
    s2 = build_state_eval2(row, "v2")
    assert s2.cac == pytest.approx(float(row.cac_v2))
    assert s2.ltv == s1.ltv  # LTV unchanged; only CAC varies


def test_eval2_env_config_defaults_are_round1():
    cfg = eval2_env_config(financing=True)
    assert "financing_model" not in cfg  # engine default = rescue = round 1
    assert cfg["marketing_curve"] == "v2"
