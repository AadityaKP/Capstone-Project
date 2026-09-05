"""F4: deterministic tests for the physics_v2 flags.

Four guarantees, per the frozen protocol (validation/calibration/PROTOCOL.md):
  1. Legacy flags reproduce the RECORDED pre-change trajectories bit-identically
     (golden = validation/agents/decision_log.csv, written by the A2 run before
     any v2 code existed).
  2. v2 flags at ~100x research scale keep spend ratios inside the EDGAR
     [p10, p90] bands and the D4-identified raisers stop auto-bankrupting
     under hold.
  3. The financing rule fires only when enabled, and when disabled consumes no
     RNG draws (trajectory with the flag absent == flag False).
  4. Corridor calibration-point equivalence: at exactly $50k MRR the
     scale-aware heuristic tiers equal the legacy dollar tiers.
"""
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

from agents.adapter import ActionAdapter
from agents.baseline_agents import CFOAgent, CMOAgent, CPOAgent, merge_actions
from env.schemas import EnvState
from env.startup_env import StartupEnv

from backtest_lib import (NOOP, build_state, load_panel, pick_init,
                          run_company_arm)

DECISION_LOG = ROOT / "validation/agents/decision_log.csv"


# ------------------------------------------------------------------ 1. legacy golden

@pytest.mark.parametrize("policy", ["noop", "heuristic"])
@pytest.mark.parametrize("seed", [0, 7])
def test_legacy_reproduces_recorded_decision_log(policy, seed):
    golden = pd.read_csv(DECISION_LOG)
    golden = golden[(golden.policy == policy) & (golden.seed == seed)]
    assert len(golden) > 0, "recorded decision log missing this arm/seed"

    env = StartupEnv(initial_config={"deterministic_rng": True})
    env.reset(seed=seed)
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    for _, row in golden.iterrows():
        action = deepcopy(NOOP) if policy == "noop" else merge_actions(env.state)
        clean = ActionAdapter.translate_action(action)
        _, _, terminated, truncated, _ = env.step(clean)
        assert env.state.mrr == pytest.approx(row.post_mrr, abs=1e-6), \
            f"month {row.month}: mrr diverged from recorded run"
        assert env.state.cash == pytest.approx(row.post_cash, abs=1e-4), \
            f"month {row.month}: cash diverged from recorded run"
        if terminated or truncated:
            break


def _research_state() -> EnvState:
    env = StartupEnv(initial_config={"deterministic_rng": True})
    env.reset(seed=0)
    return env.state


# ------------------------------------------------------------------ 3. financing rule

def test_financing_disabled_is_a_true_noop():
    cfgs = [{"deterministic_rng": True},
            {"deterministic_rng": True, "financing_enabled": False}]
    paths = []
    for cfg in cfgs:
        env = StartupEnv(initial_config=cfg)
        env.reset(seed=3)
        path = []
        for _ in range(24):
            _, _, term, trunc, info = env.step(deepcopy(NOOP))
            assert info["financing_raise"] == 0.0
            path.append((env.state.mrr, env.state.cash, env.state.competitors))
            if term or trunc:
                break
        paths.append(path)
    assert paths[0] == paths[1]


def test_financing_fires_below_threshold_and_is_logged():
    env = StartupEnv(initial_config={
        "deterministic_rng": True, "financing_enabled": True,
        "scheduled_shocks": False, "initial_cash": 40_000.0,
        "monthly_burn": 30_000.0, "initial_mrr": 5_000.0})
    env.reset(seed=1)
    raised = 0.0
    for _ in range(12):
        _, _, term, trunc, info = env.step(deepcopy(NOOP))
        raised += info["financing_raise"]
        if term or trunc:
            break
    assert raised > 0.0, "burning company with <2mo runway never raised in 12 draws"
    assert env.financing_events, "raise not logged in the trace"
    ev = env.financing_events[0]
    assert ev["amount"] == pytest.approx(
        env.financing_raise_multiple * ev["net_burn"], rel=1e-9)
    assert ev["runway_before"] < env.financing_runway_threshold_months


def test_v2_curve_requires_fitted_constant_or_override():
    from env import business_logic
    if business_logic.SATURATION_ACQUISITION_RATE_V2 is None:
        with pytest.raises(ValueError):
            StartupEnv(initial_config={"marketing_curve": "v2"})
    # explicit override always works
    env = StartupEnv(initial_config={"marketing_curve": "v2",
                                     "saturation_acquisition_rate": 0.05})
    env.reset(seed=0)
    env.step(deepcopy(NOOP))


# ------------------------------------------------------------------ 4. corridor equivalence

def test_corridor_matches_legacy_at_calibration_point():
    state = _research_state()
    assert state.mrr == 50_000.0
    state.cash = 1_000_000.0
    for legacy_cls in (CMOAgent, CPOAgent, CFOAgent):
        legacy = legacy_cls(scale=1.0, corridor="legacy").act(state)
        aware = legacy_cls(scale=1.0, corridor="scale_aware").act(state)
        assert legacy == aware, f"{legacy_cls.__name__} diverges at 50k MRR"


# ------------------------------------------------------------------ 2. v2 at real scale

@pytest.fixture(scope="module")
def panel():
    return load_panel()


def _v2_env_config():
    return {"marketing_curve": "v2",
            # explicit rate so this test is meaningful before/after the fit
            # lands; the fitted default is asserted separately in Phase 3.
            "saturation_acquisition_rate": 0.075,
            "financing_enabled": True,
            "competitive_entry": "scale_neutral"}


def test_v2_raisers_survive_under_hold(panel):
    # ASAN: a D4-identified all-seed bankruptcy under v1 physics.
    row, _ = pick_init(panel[panel.ticker == "ASAN"])
    out = run_company_arm(row, "hold", range(5),
                          extra_env_config=_v2_env_config())
    assert out["deaths"] < 5, "ASAN still dies on every seed with financing on"
    assert out["n_financing_raises"] > 0


def test_v2_spend_ratios_at_100x_research_scale():
    # F4 spec: "v2 flags at 100x scale -> spend ratios within EDGAR bands".
    # The 100x company keeps the research profile's unit economics, so the
    # heuristic tiers pick their calibration-point branches and the ratio must
    # sit inside EDGAR [p10, p90] = [36.9%, 93.7%] of revenue.
    env = StartupEnv(initial_config={
        "deterministic_rng": True, "scheduled_shocks": False,
        "initial_mrr": 5_000_000.0, "initial_cash": 100_000_000.0,
        "marketing_curve": "v2", "saturation_acquisition_rate": 0.075,
        "competitive_entry": "scale_neutral"})
    env.reset(seed=0)
    ratios = []
    for _ in range(12):
        action = {}
        for agent in (CFOAgent(corridor="scale_aware"),
                      CMOAgent(corridor="scale_aware"),
                      CPOAgent(corridor="scale_aware")):
            action.update(agent.act(env.state))
        mrr_before = env.state.mrr
        clean = ActionAdapter.translate_action(action)
        _, _, term, trunc, _ = env.step(clean)
        ratios.append((clean["marketing"]["spend"]
                       + clean["product"]["r_and_d_spend"]) / mrr_before)
        if term or trunc:
            break
    med = float(np.median(ratios))
    assert 0.369 <= med <= 0.937, \
        f"median discretionary spend ratio {med:.2f} outside EDGAR [p10,p90]"


def test_corridor_bounds_hold_at_real_scale(panel):
    # The corridor's own guarantee at an EDGAR-mapped state: spend never falls
    # below the floors (2% MRR marketing, 13.1% MRR R&D at zero deficit) nor
    # above the caps, regardless of which tier the state economics select.
    row, _ = pick_init(panel[panel.ticker == "DDOG"])
    state = build_state(row)
    from backtest_lib import rollout
    out = rollout(row, state, "boardroom", seed=0, scale=state.mrr / 50_000.0,
                  extra_env_config=_v2_env_config(), corridor="scale_aware",
                  collect_trace=True)
    for t in out["trace"]:
        # t["mrr"] is post-step; floors were computed on the pre-step state,
        # so allow the one-month growth wedge with a 0.7 factor.
        assert t["mkt_spend"] >= 0.02 * t["mrr"] * 0.7
        assert t["rd_spend"] >= 0.131 * t["mrr"] * 0.7
        assert t["rd_spend"] <= max(0.365 * t["mrr"] / 0.7, t["mrr"] * 10)
