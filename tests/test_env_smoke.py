import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.startup_env import StartupEnv


def test_reset_matches_current_observation_layout():
    env = StartupEnv()
    obs, info = env.reset(seed=7)

    assert obs.shape == (16,)
    assert obs[0] == 50_000
    assert obs[1] == 1_000_000
    assert env.state.headcount == 1
    assert info == {}


def test_step_advances_time_and_preserves_bundle_shape():
    env = StartupEnv()
    env.reset(seed=3)

    action = {
        "marketing": {"spend": 5_000.0, "channel": "ppc"},
        "hiring": {"hires": 1, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 2_000.0},
        "pricing": {"price_change_pct": 0.0},
    }
    obs, reward, terminated, truncated, info = env.step(action)

    assert obs.shape == (16,)
    assert isinstance(reward, float)
    assert env.state.months_elapsed == 1
    assert env.state.headcount == 2
    assert "rule_of_40" in info
    assert info["state"]["headcount"] == 2
    assert not terminated
    assert not truncated


def test_hard_shock_label_is_returned_at_deterministic_timesteps():
    env = StartupEnv()
    env.reset(seed=1)
    env.state.months_elapsed = 24

    action = {
        "marketing": {"spend": 0.0, "channel": "ppc"},
        "hiring": {"hires": 0, "cost_per_employee": 10_000.0},
        "product": {"r_and_d_spend": 0.0},
        "pricing": {"price_change_pct": 0.0},
    }
    _, _, _, _, info = env.step(action)

    assert info["shock_label"].startswith("RATE_HIKE")
