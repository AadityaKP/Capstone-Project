"""Addendum A (S12): the oracle_v3_no_modifier policy and the decomp arm
registry match PROTOCOL_addendum_A.md. No LLM calls."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation/round2"))

from oracle.action_modifier import NoOpActionModifier
from simulation_runner import _build_agent_for_policy


def test_oracle_v3_no_modifier_wiring():
    agent = _build_agent_for_policy("oracle_v3_no_modifier", 10)
    board = agent.boardroom
    assert board.oracle_mode == "oracle_v3"
    assert board.use_oracle is True
    assert board.enable_action_modifier is False
    assert isinstance(board.action_modifier, NoOpActionModifier)
    assert board.oracle.mode == "oracle_v3"
    assert board.oracle.enable_memory_retrieval is True
    assert board.oracle.brief_version == "v1"


def test_noop_modifier_leaves_actions_unchanged():
    action = {"marketing": {"spend": 5000.0},
              "product": {"r_and_d_spend": 2000.0},
              "hiring": {"hires": 1}}
    assert NoOpActionModifier().modify(action, brief=None) == action


def test_decomp_arm_registry_matches_protocol():
    import a3_decomp as d
    assert list(d.ARMS) == ["da", "db", "dc", "dd", "l1"]
    assert d.N_EPISODES == 20 and d.FREQ == 10
    assert d.QWEN == "qwen2.5:7b-instruct"
    # v2 arms reuse the recorded a3_v2phys config exactly (meta_*.json)
    assert d.V2_ENV == {"deterministic_rng": True, "marketing_curve": "v2",
                        "competitive_entry": "scale_neutral"}
    assert d.V2_AGENT == {"corridor": "scale_aware"}
    assert d.ARMS["da"]["runs"] == [("oracle_v3_no_modifier",
                                     "oracle_v3_no_modifier")]
    assert d.ARMS["db"]["agent"]()["modifier_bound"] == "tier"
    assert d.ARMS["dc"]["env"]["shock_recovery"] == "mean_revert"
    assert d.ARMS["dc"]["env"]["marketing_curve"] == "v2"
    assert d.ARMS["dc"]["runs"][0] == ("boardroom", "boardroom_mr")
    assert d.ARMS["dd"]["model"] == d.QWEN
    assert d.ARMS["l1"]["env"] == {"deterministic_rng": True}
    assert d.ARMS["l1"]["model"] == d.QWEN


def test_rs_ext_is_the_predeclared_extension():
    import a3_rs_ext as r
    assert r.SEEDS == list(range(21, 41))
    assert r.POLICIES == ["oracle_v3", "oracle_v3_no_memory"]
    assert r.ENV == {"deterministic_rng": True, "shock_schedule": "random"}
    assert r.FREQ == 10
