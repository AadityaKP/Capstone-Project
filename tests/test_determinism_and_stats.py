"""Tests for seed-matched comparison and the ablation statistics.

These exist because the failures they guard against are silent. A confounded
ablation still produces a tidy table of p-values; nothing in the output says the
policies were run against different worlds. So the property is asserted directly.
"""

from __future__ import annotations

import math
import random

import numpy as np
import pytest

from env import business_logic
from env.schemas import EnvState
from env.startup_env import StartupEnv
from experiments.thesis_analysis import (
    MEMORY_ABLATION_SCENARIOS,
    cohens_d,
    effect_magnitude,
    holm_bonferroni,
    mean_difference_ci,
)

ACTION = {
    "marketing": {"spend": 5000.0, "channel": "ppc"},
    "hiring": {"hires": 0, "cost_per_employee": 10000.0},
    "product": {"r_and_d_spend": 3000.0},
    "pricing": {"price_change_pct": 0.0},
}


def _macro_path(seed: int, policy_draws: int, deterministic: bool, months: int = 60):
    """Exogenous macro state. It must not depend on how much RNG a POLICY used."""
    env = StartupEnv(initial_config={"deterministic_rng": deterministic})
    env.reset(seed=seed)
    path = []
    for _ in range(months):
        for _ in range(policy_draws):
            random.random()  # stand-in for a policy sampling its own actions
        _, _, terminated, truncated, info = env.step(ACTION)
        state = info["state"]
        path.append((state["competitors"], round(state["interest_rate"], 4),
                     round(state["unemployment"], 4)))
        if terminated or truncated:
            break
    return path


# --------------------------------------------------------------------------
# The property the whole ablation rests on
# --------------------------------------------------------------------------
def test_policy_rng_use_does_not_perturb_the_world():
    """A policy that samples its own actions must not change the environment it
    is being evaluated in. This is what makes 'same seed list' mean anything."""
    assert _macro_path(7, 0, True) == _macro_path(7, 6, True)


def test_legacy_mode_still_has_the_confound():
    """Documents the behaviour deterministic_rng exists to fix. If this ever
    passes, the default changed and every recorded result moved with it."""
    assert _macro_path(7, 0, False) != _macro_path(7, 6, False)


def test_draw_count_is_state_independent_under_deterministic_rng():
    """apply_recession_cascade drew only when unemployment > 8 and rate > 7,
    making per-step consumption state-dependent. Two policies reaching different
    macro states then desynchronise permanently."""
    def draws_per_step(trigger: bool) -> set[int]:
        env = StartupEnv(initial_config={"deterministic_rng": True})
        env.reset(seed=3)
        if trigger:
            env.state.unemployment, env.state.interest_rate = 9.0, 8.0
        counter = {"n": 0}
        real_random, real_uniform = env._rng.random, env._rng.uniform
        env._rng.random = lambda *a, **k: (counter.__setitem__("n", counter["n"] + 1),
                                           real_random(*a, **k))[1]
        env._rng.uniform = lambda *a, **k: (counter.__setitem__("n", counter["n"] + 1),
                                            real_uniform(*a, **k))[1]
        counts = set()
        for _ in range(5):
            counter["n"] = 0
            env.step(ACTION)
            counts.add(counter["n"])
        return counts

    assert draws_per_step(True) == draws_per_step(False)


def test_deterministic_rng_is_reproducible_from_the_seed_alone():
    assert _macro_path(11, 0, True) == _macro_path(11, 0, True)


def test_deterministic_rng_still_varies_by_seed():
    assert _macro_path(11, 0, True) != _macro_path(12, 0, True)


def test_private_stream_is_isolated_from_the_global_module():
    """Reseeding the global module mid-episode must not touch the physics."""
    env = StartupEnv(initial_config={"deterministic_rng": True})
    env.reset(seed=5)
    first = [round(env.step(ACTION)[4]["state"]["mrr"], 6) for _ in range(5)]

    env2 = StartupEnv(initial_config={"deterministic_rng": True})
    env2.reset(seed=5)
    second = []
    for _ in range(5):
        random.seed(999)  # hostile external reseed
        second.append(round(env2.step(ACTION)[4]["state"]["mrr"], 6))
    assert first == second


def test_cascade_always_draw_preserves_the_condition():
    """Drawing unconditionally must not change WHEN the cascade fires, only how
    the stream is consumed."""
    def confidence_after(always_draw: bool, unemployment: float, rate: float) -> float:
        state = EnvState(
            mrr=50000, cash=100000, cac=50, ltv=7000,
            churn_enterprise=0.01, churn_smb=0.03, churn_b2c=0.05,
            interest_rate=rate, consumer_confidence=100.0, competitors=5,
            product_quality=0.5, price=50.0, unemployment=unemployment,
        )
        business_logic.apply_recession_cascade(
            state, rng=random.Random(0), always_draw=always_draw
        )
        return state.consumer_confidence

    # Below the threshold the cascade must never fire, either way.
    assert confidence_after(True, 4.0, 3.0) == 100.0
    assert confidence_after(False, 4.0, 3.0) == 100.0
    # Above it, both variants take the same branch for the same seeded draw.
    assert confidence_after(True, 9.0, 8.0) == confidence_after(False, 9.0, 8.0)


def test_business_logic_defaults_to_the_global_module():
    """rng=None must behave exactly as before, or every recorded result moves."""
    state_a = EnvState(mrr=50000, cash=100000, cac=50, ltv=7000, churn_enterprise=0.01,
                       churn_smb=0.03, churn_b2c=0.05, interest_rate=3.0,
                       consumer_confidence=100.0, competitors=5, product_quality=0.5,
                       price=50.0)
    state_b = state_a.model_copy(deep=True)
    random.seed(4)
    business_logic.interest_rate_shock(state_a)
    random.seed(4)
    business_logic.interest_rate_shock(state_b, rng=random)
    assert state_a.interest_rate == state_b.interest_rate


# --------------------------------------------------------------------------
# The four-arm memory ablation
# --------------------------------------------------------------------------
def test_memory_ablation_crosses_both_memory_systems():
    """The claim is about the architecture, so all four cells must be present:
    neither / episodic / semantic / both."""
    ids = [s["scenario_id"] for s in MEMORY_ABLATION_SCENARIOS]
    assert ids == ["memory_none", "memory_episodic_only",
                   "memory_semantic_only", "memory_full"]


def test_memory_ablation_policies_are_buildable():
    from simulation_runner import _build_agent_for_policy
    for scenario in MEMORY_ABLATION_SCENARIOS:
        if scenario["policy"] in {"oracle_v4_causal", "oracle_v4_causal_no_memory"}:
            continue  # needs Neo4j; construction is covered by the suite itself
        assert _build_agent_for_policy(scenario["policy"], 10) is not None


# --------------------------------------------------------------------------
# Effect size, interval, correction
# --------------------------------------------------------------------------
def test_cohens_d_sign_and_magnitude():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = a + 5.0
    assert cohens_d(a, b) > 0          # b greater than a
    assert cohens_d(b, a) < 0
    assert abs(cohens_d(a, a)) < 1e-9


def test_cohens_d_is_hedges_corrected():
    """Uncorrected d is biased upward at small n; the correction pulls it down."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    pooled = math.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    raw = (np.mean(b) - np.mean(a)) / pooled
    assert abs(cohens_d(a, b)) < abs(raw)


def test_cohens_d_handles_degenerate_input():
    assert math.isnan(cohens_d(np.array([1.0]), np.array([2.0])))
    assert cohens_d(np.array([2.0, 2.0]), np.array([2.0, 2.0])) == 0.0


def test_effect_magnitude_bands():
    assert effect_magnitude(0.1) == "negligible"
    assert effect_magnitude(0.3) == "small"
    assert effect_magnitude(0.6) == "medium"
    assert effect_magnitude(1.5) == "large"
    assert effect_magnitude(-1.5) == "large"   # magnitude, not direction
    assert effect_magnitude(float("nan")) == "unknown"


def test_confidence_interval_brackets_the_difference():
    a = np.random.default_rng(0).normal(0, 1, 60)
    b = np.random.default_rng(1).normal(2, 1, 60)
    difference, low, high = mean_difference_ci(a, b)
    assert low < difference < high
    assert low > 0          # a genuine 2-sigma separation should exclude zero


def test_confidence_interval_includes_zero_for_identical_samples():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0, 1, 80), rng.normal(0, 1, 80)
    _, low, high = mean_difference_ci(a, b)
    assert low < 0 < high


def test_holm_is_monotonic_and_never_below_raw():
    raw = [0.001, 0.01, 0.04, 0.5]
    adjusted = holm_bonferroni(raw)
    assert all(adj >= p for adj, p in zip(adjusted, raw))
    assert adjusted == sorted(adjusted)


def test_holm_rejects_what_bonferroni_would_and_more():
    """Holm must be at least as powerful as plain Bonferroni at the same alpha."""
    raw = [0.001, 0.02, 0.03]
    holm = holm_bonferroni(raw)
    bonferroni = [min(1.0, p * len(raw)) for p in raw]
    assert all(h <= b + 1e-12 for h, b in zip(holm, bonferroni))


def test_holm_caps_at_one_and_skips_nan():
    adjusted = holm_bonferroni([0.9, 0.95, float("nan")])
    assert adjusted[0] <= 1.0 and adjusted[1] <= 1.0
    assert math.isnan(adjusted[2])


def test_holm_changes_a_borderline_verdict():
    """The correction has to actually bite, or it is decoration. A raw p of 0.013
    inside a family of eight is not significant."""
    raw = [0.013] + [0.4] * 7
    assert raw[0] < 0.05
    assert holm_bonferroni(raw)[0] > 0.05
