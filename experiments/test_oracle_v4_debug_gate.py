"""Tests for oracle_v4_debug_gate.

Pure helper tests use synthetic trace rows only (no Ollama/Neo4j/simulation
stack). The CLI smoke tests monkeypatch `run_v4_episode_prefix` so `main()`
can be exercised end-to-end (JSON artifact, summary line, exit code) without
touching the real simulation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import oracle_v4_debug_gate as gate


# --------------------------------------------------------------------------- #
# Synthetic row builders
# --------------------------------------------------------------------------- #

def make_proposal(agent: str, causal_confidence, price_change_pct=None) -> dict:
    actions: dict = {}
    if agent == "CFO":
        actions["hiring"] = {"hires": 1, "cost_per_employee": 10000.0}
        actions["pricing"] = {
            "price_change_pct": price_change_pct if price_change_pct is not None else -0.02,
        }
    elif agent == "CMO":
        actions["marketing"] = {"spend": 10000.0, "channel": "ppc"}
    elif agent == "CPO":
        actions["product"] = {"r_and_d_spend": 10000.0}
    return {
        "agent": agent,
        "actions": actions,
        "causal_confidence": causal_confidence,
        "base_score": 0.5,
        "final_confidence": 0.5,
    }


def make_row(
    month: int,
    stress_node: str = "Steady_State",
    confidences=(0.77, 0.77, 0.77),
    price_change_pct=None,
    rule_of_40=None,
    include_cfo: bool = True,
) -> dict:
    cfo_conf, cmo_conf, cpo_conf = confidences
    proposals = []
    if include_cfo:
        proposals.append(make_proposal("CFO", cfo_conf, price_change_pct=price_change_pct))
    proposals.append(make_proposal("CMO", cmo_conf))
    proposals.append(make_proposal("CPO", cpo_conf))

    final_action: dict = {
        "marketing": {"spend": 10000.0, "channel": "ppc"},
        "hiring": {"hires": 1, "cost_per_employee": 10000.0},
        "pricing": {"price_change_pct": price_change_pct if price_change_pct is not None else -0.02},
        "product": {"r_and_d_spend": 10000.0},
    }

    return {
        "month": month,
        "cash_before": 1_000_000.0,
        "cash_after": 1_000_000.0,
        "mrr_after": 50_000.0,
        "reward": 0.0,
        "rule_of_40": rule_of_40,
        "terminated": False,
        "truncated": False,
        "proposal_source": "llm",
        "causal_stress_node": stress_node,
        "final_action": final_action,
        "proposals": proposals,
    }


# --------------------------------------------------------------------------- #
# linear_slope
# --------------------------------------------------------------------------- #

def test_linear_slope_basic():
    xs = [0, 1, 2, 3, 4]
    ys = [1, 3, 5, 7, 9]  # y = 2x + 1
    assert gate.linear_slope(xs, ys) == pytest.approx(2.0)


def test_linear_slope_flat_is_zero():
    xs = [0, 1, 2, 3]
    ys = [-100.0, -100.0, -100.0, -100.0]
    assert gate.linear_slope(xs, ys) == pytest.approx(0.0)


def test_linear_slope_insufficient_points():
    assert gate.linear_slope([0], [1]) is None
    assert gate.linear_slope([], []) is None


# --------------------------------------------------------------------------- #
# check_cash_shortage_confidence_non_uniform
# --------------------------------------------------------------------------- #

def test_cash_shortage_flat_on_second_occurrence_fails():
    rows = [
        make_row(0, "Steady_State"),
        make_row(1, "Cash_Shortage", confidences=(0.5, 0.5, 0.5)),  # 1st occurrence, not checked
        make_row(2, "Cash_Shortage", confidences=(0.5, 0.5, 0.5)),  # 2nd occurrence, flat -> FAIL
    ]
    result = gate.check_cash_shortage_confidence_non_uniform(rows, threshold=0.05)
    assert result["status"] == "FAIL"
    assert result["occurrences"] == 2
    assert result["details"][0]["spread"] == pytest.approx(0.0)


def test_cash_shortage_differentiated_on_second_occurrence_passes():
    rows = [
        make_row(0, "Steady_State"),
        make_row(1, "Cash_Shortage", confidences=(0.5, 0.5, 0.5)),  # 1st occurrence, not checked
        make_row(2, "Cash_Shortage", confidences=(0.6, 0.7, 0.8)),  # 2nd occurrence, spread=0.2 -> PASS
    ]
    result = gate.check_cash_shortage_confidence_non_uniform(rows, threshold=0.05)
    assert result["status"] == "PASS"
    assert result["details"][0]["spread"] == pytest.approx(0.2)


def test_cash_shortage_fewer_than_two_occurrences_is_na_and_does_not_fail():
    rows = [
        make_row(0, "Steady_State"),
        make_row(1, "Cash_Shortage", confidences=(0.5, 0.5, 0.5)),
        make_row(2, "Churn_Spike", confidences=(0.7, 0.8, 0.9)),
    ]
    result = gate.check_cash_shortage_confidence_non_uniform(rows, threshold=0.05)
    assert result["status"] == "NA"
    assert result["occurrences"] == 1

    overall = gate.evaluate_all_checks(rows + [make_row(m) for m in range(3, 12)])
    # NA must not appear in the failing list
    assert "cash_shortage_confidence_non_uniform" not in overall["failing"]


def test_cash_shortage_no_occurrences_is_na():
    rows = [make_row(m, "Steady_State") for m in range(5)]
    result = gate.check_cash_shortage_confidence_non_uniform(rows)
    assert result["status"] == "NA"
    assert result["occurrences"] == 0


def test_cash_shortage_missing_confidence_on_second_occurrence_fails():
    rows = [
        make_row(0, "Cash_Shortage", confidences=(0.5, 0.5, 0.5)),
        make_row(1, "Cash_Shortage", confidences=(0.5, 0.5, 0.5), include_cfo=False),
    ]
    result = gate.check_cash_shortage_confidence_non_uniform(rows, threshold=0.05)
    assert result["status"] == "FAIL"
    assert result["details"][0]["reason"] == "missing_confidence"
    assert "CFO" in result["details"][0]["missing_agents"]


# --------------------------------------------------------------------------- #
# check_positive_pricing_seen
# --------------------------------------------------------------------------- #

def test_positive_pricing_seen_passes_with_one_positive_value():
    rows = [make_row(m, price_change_pct=-0.02) for m in range(29)]
    rows.append(make_row(29, price_change_pct=0.03))
    result = gate.check_positive_pricing_seen(rows)
    assert result["status"] == "PASS"
    assert result["max_price_change_pct"] == pytest.approx(0.03)
    assert result["source_used"] == "cfo_proposal"


def test_positive_pricing_seen_fails_when_all_zero_or_negative():
    rows = [make_row(m, price_change_pct=-0.05 if m % 2 == 0 else 0.0) for m in range(30)]
    result = gate.check_positive_pricing_seen(rows)
    assert result["status"] == "FAIL"
    assert result["max_price_change_pct"] == pytest.approx(0.0)


def test_positive_pricing_seen_falls_back_to_final_action_when_cfo_missing():
    rows = [make_row(m, price_change_pct=-0.02, include_cfo=False) for m in range(10)]
    rows[5]["final_action"]["pricing"]["price_change_pct"] = 0.04
    result = gate.check_positive_pricing_seen(rows)
    assert result["status"] == "PASS"
    assert result["source_used"] == "final_action"
    assert result["max_price_change_pct"] == pytest.approx(0.04)


def test_positive_pricing_seen_ignores_months_past_29():
    rows = [make_row(m, price_change_pct=-0.02) for m in range(30)]
    rows.append(make_row(30, price_change_pct=0.05))  # month 30 is out of window
    result = gate.check_positive_pricing_seen(rows)
    assert result["status"] == "FAIL"


# --------------------------------------------------------------------------- #
# check_rule40_slope_up
# --------------------------------------------------------------------------- #

def test_rule40_slope_up_passes_for_rising_trend():
    rows = [make_row(m, rule_of_40=-100.0 + 0.5 * m) for m in range(30)]
    result = gate.check_rule40_slope_up(rows)
    assert result["status"] == "PASS"
    assert result["slope"] == pytest.approx(0.5)
    assert result["num_points"] == 30


def test_rule40_slope_up_fails_for_flat_trend():
    rows = [make_row(m, rule_of_40=-100.0) for m in range(30)]
    result = gate.check_rule40_slope_up(rows)
    assert result["status"] == "FAIL"
    assert result["slope"] == pytest.approx(0.0)


def test_rule40_slope_up_fails_for_negative_trend():
    rows = [make_row(m, rule_of_40=-50.0 - 0.3 * m) for m in range(30)]
    result = gate.check_rule40_slope_up(rows)
    assert result["status"] == "FAIL"
    assert result["slope"] < 0


def test_rule40_slope_up_fails_with_fewer_than_ten_points():
    rows = [make_row(m, rule_of_40=-100.0 + m) for m in range(5)]
    result = gate.check_rule40_slope_up(rows)
    assert result["status"] == "FAIL"
    assert result["reason"] == "insufficient_points"
    assert result["num_points"] == 5


# --------------------------------------------------------------------------- #
# compact_row
# --------------------------------------------------------------------------- #

def test_compact_row_extracts_expected_fields():
    decision_trace = {
        "proposal_source": "llm",
        "causal_stress_node": "Churn_Spike",
        "final_action": {"pricing": {"price_change_pct": -0.03}},
        "proposals": [
            {
                "agent": "CFO",
                "actions": {"pricing": {"price_change_pct": -0.03}},
                "causal_confidence": 0.7,
                "base_score": 0.4,
                "final_confidence": 0.42,
            },
            {
                "agent": "CMO",
                "actions": {"marketing": {"spend": 25000.0}},
                "causal_confidence": 0.8,
                "base_score": 0.4,
                "final_confidence": 0.43,
            },
        ],
    }
    row = gate.compact_row(
        month=25,
        cash_before=571726.0,
        cash_after=567226.0,
        mrr_after=313398.0,
        reward=-1.5,
        rule_of_40=-100.3,
        terminated=False,
        truncated=False,
        decision_trace=decision_trace,
    )
    assert row["month"] == 25
    assert row["causal_stress_node"] == "Churn_Spike"
    assert row["proposal_source"] == "llm"
    assert len(row["proposals"]) == 2
    assert row["proposals"][0]["agent"] == "CFO"
    assert row["proposals"][0]["causal_confidence"] == 0.7
    assert row["final_action"]["pricing"]["price_change_pct"] == -0.03


def test_compact_row_handles_missing_decision_trace_fields():
    row = gate.compact_row(
        month=0, cash_before=1.0, cash_after=2.0, mrr_after=3.0,
        reward=0.0, rule_of_40=None, terminated=False, truncated=False,
        decision_trace={},
    )
    assert row["proposal_source"] is None
    assert row["causal_stress_node"] is None
    assert row["stress_persistence_months"] is None
    assert row["final_action"] is None
    assert row["proposals"] == []
    assert row["rule_of_40"] is None


def test_compact_row_preserves_stress_persistence_months():
    row = gate.compact_row(
        month=12, cash_before=1.0, cash_after=2.0, mrr_after=3.0,
        reward=0.0, rule_of_40=None, terminated=False, truncated=False,
        decision_trace={"stress_persistence_months": 7},
    )
    assert row["stress_persistence_months"] == 7


# --------------------------------------------------------------------------- #
# evaluate_all_checks aggregation
# --------------------------------------------------------------------------- #

def test_evaluate_all_checks_overall_pass():
    rows = [
        make_row(m, "Steady_State", rule_of_40=-100.0 + 0.5 * m,
                 price_change_pct=(0.02 if m == 10 else -0.02))
        for m in range(30)
    ]
    result = gate.evaluate_all_checks(rows)
    assert result["overall"] == "PASS"
    assert result["failing"] == []
    assert result["checks"]["cash_shortage_confidence_non_uniform"]["status"] == "NA"


def test_evaluate_all_checks_overall_fail_lists_failing_check():
    # Flat rule_of_40 -> rule40_slope_up FAILs -> overall FAIL.
    rows = [
        make_row(m, "Steady_State", rule_of_40=-100.0,
                 price_change_pct=(0.02 if m == 10 else -0.02))
        for m in range(30)
    ]
    result = gate.evaluate_all_checks(rows)
    assert result["overall"] == "FAIL"
    assert "rule40_slope_up" in result["failing"]
    assert "positive_pricing_seen" not in result["failing"]


# --------------------------------------------------------------------------- #
# _find_repo_root
# --------------------------------------------------------------------------- #

def test_find_repo_root_finds_sentinel(tmp_path):
    repo = tmp_path / "myrepo"
    (repo / "experiments").mkdir(parents=True)
    (repo / "simulation_runner.py").write_text("# sentinel\n")
    script_path = repo / "experiments" / "oracle_v4_debug_gate.py"
    script_path.write_text("# placeholder\n")

    found = gate._find_repo_root(start=script_path)
    assert found == repo


def test_find_repo_root_fallback_when_sentinel_absent(tmp_path):
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    script_path = deep / "oracle_v4_debug_gate.py"
    script_path.write_text("# placeholder\n")

    found = gate._find_repo_root(start=script_path, max_levels=1)
    assert found == (tmp_path / "a")


# --------------------------------------------------------------------------- #
# CLI smoke tests (monkeypatched run_v4_episode_prefix; no Ollama/Neo4j)
# --------------------------------------------------------------------------- #

def _fake_run_meta(**overrides) -> dict:
    meta = {
        "policy": "oracle_v4_causal_hetero",
        "seed": 0,
        "max_months": 30,
        "oracle_frequency": 5,
        "months_elapsed": 31,
        "terminated": False,
        "truncated": False,
        "final_cash": 500_000.0,
        "final_mrr": 400_000.0,
        "survived_past_30": True,
        "episode_stats": {"llm_calls": 12, "proposal_llm_calls": 12},
    }
    meta.update(overrides)
    return meta


def test_cli_smoke_pass(monkeypatch, tmp_path, capsys):
    rows = [
        make_row(m, "Steady_State", rule_of_40=-100.0 + 0.5 * m,
                 price_change_pct=(0.03 if m == 20 else -0.02))
        for m in range(30)
    ]
    meta = _fake_run_meta()

    def fake_run(**kwargs):
        return rows, meta

    monkeypatch.setattr(gate, "run_v4_episode_prefix", lambda *a, **k: fake_run(**k))

    out_file = tmp_path / "result.json"
    exit_code = gate.main(["--output", str(out_file)])
    captured = capsys.readouterr()

    assert exit_code == 0
    first_line = captured.out.splitlines()[0]
    assert first_line.startswith("ORACLE_V4_DEBUG_GATE PASS")
    assert "cash_shortage_confidence_non_uniform=NA" in first_line
    assert "positive_pricing_seen=PASS" in first_line
    assert "rule40_slope_up=PASS" in first_line

    assert out_file.exists()
    artifact = json.loads(out_file.read_text())
    assert artifact["overall"] == "PASS"
    assert len(artifact["trace"]) == 30


def test_cli_smoke_fail(monkeypatch, tmp_path, capsys):
    # Flat rule_of_40 and no positive pricing, plus a non-uniform-but-flat
    # Cash_Shortage streak whose 2nd occurrence is flat -> three different
    # failure reasons surfacing together.
    rows = []
    for m in range(30):
        if 10 <= m <= 15:
            rows.append(make_row(m, "Cash_Shortage", confidences=(0.5, 0.5, 0.5),
                                  rule_of_40=-100.0, price_change_pct=-0.02))
        else:
            rows.append(make_row(m, "Steady_State", rule_of_40=-100.0, price_change_pct=-0.02))

    meta = _fake_run_meta(months_elapsed=16, terminated=True, survived_past_30=False,
                           final_cash=-50_000.0)

    monkeypatch.setattr(gate, "run_v4_episode_prefix", lambda *a, **k: (rows, meta))

    out_file = tmp_path / "result.json"
    exit_code = gate.main(["--output", str(out_file), "--verbose"])
    captured = capsys.readouterr()
    lines = captured.out.splitlines()

    assert exit_code == 1
    assert lines[0].startswith("ORACLE_V4_DEBUG_GATE FAIL")
    assert "cash_shortage_confidence_non_uniform=FAIL" in lines[0]
    assert "positive_pricing_seen=FAIL" in lines[0]
    assert "rule40_slope_up=FAIL" in lines[0]

    # Verbose mode dumps full per-check JSON too.
    assert any("cash_shortage_confidence_non_uniform_full=" in line for line in lines)

    artifact = json.loads(out_file.read_text())
    assert artifact["overall"] == "FAIL"
    assert set(artifact["failing_checks"]) == {
        "cash_shortage_confidence_non_uniform",
        "positive_pricing_seen",
        "rule40_slope_up",
    }


def test_cli_smoke_error_path_returns_exit_code_two(monkeypatch, tmp_path, capsys):
    def boom(*a, **k):
        raise RuntimeError("Ollama unreachable")

    monkeypatch.setattr(gate, "run_v4_episode_prefix", boom)

    out_file = tmp_path / "result.json"
    exit_code = gate.main(["--output", str(out_file)])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert captured.out.startswith("ORACLE_V4_DEBUG_GATE ERROR")
    assert not out_file.exists()


def test_relative_output_path_resolves_against_cwd(monkeypatch, tmp_path, capsys):
    # Pass/fail status is irrelevant here; this test only checks *where*
    # a relative --output path lands.
    rows = [
        make_row(m, "Steady_State", rule_of_40=-100.0 + 0.5 * m,
                 price_change_pct=(0.03 if m == 20 else -0.02))
        for m in range(30)
    ]
    meta = _fake_run_meta()
    monkeypatch.setattr(gate, "run_v4_episode_prefix", lambda *a, **k: (rows, meta))
    monkeypatch.chdir(tmp_path)

    exit_code = gate.main(["--output", "outputs/oracle_v4_debug_gate/seed0_latest.json"])
    assert exit_code == 0
    assert (tmp_path / "outputs" / "oracle_v4_debug_gate" / "seed0_latest.json").exists()
