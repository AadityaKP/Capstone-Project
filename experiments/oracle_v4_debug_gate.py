"""Oracle v4 causal-hetero single-seed stabilization gate.

Runs `oracle_v4_causal_hetero` for one seed up to N months, then evaluates
three pass/fail/NA checks against the resulting trace:

  - cash_shortage_confidence_non_uniform
        From the 2nd Cash_Shortage occurrence onward, CFO/CMO/CPO
        causal_confidence must be present and spread by at least
        --confidence-spread-threshold. <2 occurrences => NA (does not
        fail the overall gate).

  - positive_pricing_seen
        Across months 0..29, at least one CFO pricing.price_change_pct
        (falling back to final_action.pricing.price_change_pct) must be
        strictly > 0.

  - rule40_slope_up
        Linear slope of rule_of_40 over months 0..29 must be
        >= --rule40-min-slope. <10 valid points => FAIL.

Overall = PASS iff no check is FAIL (NA does not fail).

Usage:
    python experiments/oracle_v4_debug_gate.py [options]

Output:
    One line: "ORACLE_V4_DEBUG_GATE PASS|FAIL ..."
    Three "ORACLE_V4_DEBUG_GATE_DETAIL ..." lines, one per check.
    A JSON artifact at --output containing run metadata, the compact
    per-month trace, and full check results.

Exit codes:
    0 = overall PASS
    1 = overall FAIL (at least one check failed)
    2 = the run itself errored before checks could be evaluated

Per-month compact trace row schema:
    {
      "month": int,
      "cash_before": float, "cash_after": float, "mrr_after": float,
      "reward": float, "rule_of_40": float | None,
      "terminated": bool, "truncated": bool,
      "proposal_source": str | None,
      "causal_stress_node": str | None,
      "stress_persistence_months": int | None,
      "final_action": dict | None,
      "proposals": [
        {"agent": str, "actions": dict, "causal_confidence": float | None,
         "base_score": float | None, "final_confidence": float | None},
        ...
      ],
    }

Heavy/project-specific imports (simulation_runner, env, agents, config) are
local to run_v4_episode_prefix() so this module can be imported (and its
pure helpers unit tested) without Ollama/Neo4j/the full simulation stack
being importable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Repo-root resolution
# --------------------------------------------------------------------------- #

def _find_repo_root(start: Path | None = None, sentinel: str = "simulation_runner.py",
                     max_levels: int = 6) -> Path:
    """Walk upward from this file looking for `sentinel`.

    Falls back to the grandparent of this file (i.e. assumes this script
    lives at <repo_root>/experiments/oracle_v4_debug_gate.py) if the
    sentinel isn't found, so callers still get *a* path rather than a
    confusing error deep inside an import.
    """
    here = (start or Path(__file__)).resolve()
    candidates = [here.parent, *here.parent.parents]
    for parent in candidates[:max_levels]:
        if (parent / sentinel).exists():
            return parent
    return here.parent.parent


# --------------------------------------------------------------------------- #
# Compact row construction
# --------------------------------------------------------------------------- #

def compact_row(
    month: int,
    cash_before: float,
    cash_after: float,
    mrr_after: float,
    reward: Any,
    rule_of_40: Any,
    terminated: bool,
    truncated: bool,
    decision_trace: dict[str, Any],
) -> dict[str, Any]:
    """Build one compact per-month trace row from a raw decision_trace."""
    proposals = []
    for p in decision_trace.get("proposals") or []:
        proposals.append({
            "agent": p.get("agent"),
            "actions": p.get("actions"),
            "causal_confidence": p.get("causal_confidence"),
            "base_score": p.get("base_score"),
            "final_confidence": p.get("final_confidence"),
        })
    return {
        "month": int(month),
        "cash_before": float(cash_before),
        "cash_after": float(cash_after),
        "mrr_after": float(mrr_after),
        "reward": float(reward) if reward is not None else None,
        "rule_of_40": float(rule_of_40) if rule_of_40 is not None else None,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "proposal_source": decision_trace.get("proposal_source"),
        "causal_stress_node": decision_trace.get("causal_stress_node"),
        "stress_persistence_months": decision_trace.get("stress_persistence_months"),
        "final_action": decision_trace.get("final_action"),
        "proposals": proposals,
    }


# --------------------------------------------------------------------------- #
# Pure math helper
# --------------------------------------------------------------------------- #

def linear_slope(xs: list[float], ys: list[float]) -> float | None:
    """Least-squares slope of ys vs xs. None if <2 points or zero variance in x."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return None
    return num / den


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

_AGENTS = ("CFO", "CMO", "CPO")


def check_cash_shortage_confidence_non_uniform(
    rows: list[dict[str, Any]],
    threshold: float = 0.05,
) -> dict[str, Any]:
    """PASS/FAIL/NA on causal_confidence spread during Cash_Shortage.

    <2 Cash_Shortage rows => NA (does not fail the overall gate).
    From the 2nd occurrence onward, every occurrence must have all three
    agents' causal_confidence present, with max-min spread >= threshold.
    """
    cs_rows = [r for r in rows if r.get("causal_stress_node") == "Cash_Shortage"]
    if len(cs_rows) < 2:
        return {
            "status": "NA",
            "occurrences": len(cs_rows),
            "threshold": threshold,
            "details": [],
        }

    details = []
    overall_pass = True
    for occurrence_idx, row in enumerate(cs_rows[1:], start=2):
        confs: dict[str, Any] = {}
        for p in row.get("proposals") or []:
            agent = p.get("agent")
            if agent in _AGENTS:
                confs[agent] = p.get("causal_confidence")

        missing = [a for a in _AGENTS if confs.get(a) is None]
        if missing:
            details.append({
                "occurrence": occurrence_idx,
                "month": row.get("month"),
                "status": "FAIL",
                "reason": "missing_confidence",
                "missing_agents": missing,
                "confidences": confs,
            })
            overall_pass = False
            continue

        values = [confs[a] for a in _AGENTS]
        spread = max(values) - min(values)
        row_pass = spread >= threshold
        details.append({
            "occurrence": occurrence_idx,
            "month": row.get("month"),
            "status": "PASS" if row_pass else "FAIL",
            "spread": spread,
            "confidences": confs,
        })
        if not row_pass:
            overall_pass = False

    return {
        "status": "PASS" if overall_pass else "FAIL",
        "occurrences": len(cs_rows),
        "threshold": threshold,
        "details": details,
    }


def check_positive_pricing_seen(
    rows: list[dict[str, Any]],
    max_month: int = 29,
) -> dict[str, Any]:
    """PASS if any CFO (or final_action fallback) price_change_pct > 0
    across months 0..max_month."""
    details = []
    best_value: float | None = None
    best_source: str | None = None

    for row in rows:
        month = row.get("month")
        if month is None or month > max_month:
            continue

        value = None
        source = None
        for p in row.get("proposals") or []:
            if p.get("agent") == "CFO":
                pricing = (p.get("actions") or {}).get("pricing") or {}
                if "price_change_pct" in pricing:
                    value = pricing["price_change_pct"]
                    source = "cfo_proposal"
                break

        if value is None:
            pricing = (row.get("final_action") or {}).get("pricing") or {}
            if "price_change_pct" in pricing:
                value = pricing["price_change_pct"]
                source = "final_action"

        if value is None:
            continue

        details.append({"month": month, "price_change_pct": value, "source": source})
        if best_value is None or value > best_value:
            best_value = value
            best_source = source

    status = "PASS" if (best_value is not None and best_value > 0.0) else "FAIL"
    return {
        "status": status,
        "max_price_change_pct": best_value,
        "source_used": best_source or "none",
        "details": details,
    }


def check_rule40_slope_up(
    rows: list[dict[str, Any]],
    min_slope: float = 0.10,
    max_month: int = 29,
    min_points: int = 10,
) -> dict[str, Any]:
    """PASS if the rule_of_40 slope over months 0..max_month >= min_slope.

    FAIL (not NA) if fewer than min_points valid points exist.
    """
    xs, ys = [], []
    for row in rows:
        month = row.get("month")
        r40 = row.get("rule_of_40")
        if month is None or month > max_month or r40 is None:
            continue
        xs.append(float(month))
        ys.append(float(r40))

    if len(xs) < min_points:
        return {
            "status": "FAIL",
            "slope": None,
            "num_points": len(xs),
            "threshold": min_slope,
            "reason": "insufficient_points",
        }

    slope = linear_slope(xs, ys)
    if slope is None:
        return {
            "status": "FAIL",
            "slope": None,
            "num_points": len(xs),
            "threshold": min_slope,
            "reason": "degenerate_slope",
        }

    return {
        "status": "PASS" if slope >= min_slope else "FAIL",
        "slope": slope,
        "num_points": len(xs),
        "threshold": min_slope,
    }


def evaluate_all_checks(
    rows: list[dict[str, Any]],
    confidence_spread_threshold: float = 0.05,
    rule40_min_slope: float = 0.10,
) -> dict[str, Any]:
    checks = {
        "cash_shortage_confidence_non_uniform": check_cash_shortage_confidence_non_uniform(
            rows, confidence_spread_threshold,
        ),
        "positive_pricing_seen": check_positive_pricing_seen(rows),
        "rule40_slope_up": check_rule40_slope_up(rows, rule40_min_slope),
    }
    failing = [name for name, result in checks.items() if result["status"] == "FAIL"]
    return {
        "overall": "FAIL" if failing else "PASS",
        "failing": failing,
        "checks": checks,
    }


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #

_CHECK_ORDER = ("cash_shortage_confidence_non_uniform", "positive_pricing_seen", "rule40_slope_up")


def format_summary_line(eval_result: dict[str, Any], run_meta: dict[str, Any]) -> str:
    parts = [
        f"ORACLE_V4_DEBUG_GATE {eval_result['overall']}",
        f"seed={run_meta.get('seed')}",
        f"months_elapsed={run_meta.get('months_elapsed')}",
        f"survived_past_30={run_meta.get('survived_past_30')}",
    ]
    for name in _CHECK_ORDER:
        parts.append(f"{name}={eval_result['checks'][name]['status']}")
    return " ".join(parts)


def format_detail_lines(eval_result: dict[str, Any]) -> list[str]:
    checks = eval_result["checks"]
    lines = []

    cs = checks["cash_shortage_confidence_non_uniform"]
    lines.append(
        "ORACLE_V4_DEBUG_GATE_DETAIL cash_shortage_confidence_non_uniform="
        f"{cs['status']} occurrences={cs['occurrences']} threshold={cs['threshold']}"
    )

    pp = checks["positive_pricing_seen"]
    lines.append(
        "ORACLE_V4_DEBUG_GATE_DETAIL positive_pricing_seen="
        f"{pp['status']} max_price_change_pct={pp['max_price_change_pct']} "
        f"source={pp['source_used']}"
    )

    r40 = checks["rule40_slope_up"]
    lines.append(
        "ORACLE_V4_DEBUG_GATE_DETAIL rule40_slope_up="
        f"{r40['status']} slope={r40['slope']} points={r40['num_points']} "
        f"threshold={r40['threshold']}"
    )

    return lines


def build_artifact(
    run_meta: dict[str, Any],
    rows: list[dict[str, Any]],
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run": run_meta,
        "overall": eval_result["overall"],
        "failing_checks": eval_result["failing"],
        "checks": eval_result["checks"],
        "trace": rows,
    }


# --------------------------------------------------------------------------- #
# Simulation runner (heavy imports are local to this function)
# --------------------------------------------------------------------------- #

def run_v4_episode_prefix(
    seed: int,
    max_months: int,
    oracle_frequency: int,
    policy: str = "oracle_v4_causal_hetero",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run `policy` for one episode, stopping once months_elapsed > max_months
    (or the episode terminates early). Returns (compact_rows, run_meta).

    Mirrors the real v4 step loop: ActionAdapter.translate_action, full
    decision_trace capture, and the closed-loop causal write-back call —
    so this exercises the actual production code path, not a stub.
    """
    import random

    import numpy as np

    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from agents.adapter import ActionAdapter
    from env.startup_env import StartupEnv
    from simulation_runner import (
        _build_agent_for_policy,
        _capture_causal_metrics,
        _write_causal_step_outcome,
    )

    env = StartupEnv()
    agent = _build_agent_for_policy(policy, oracle_frequency=oracle_frequency)

    env.reset(seed=seed)
    agent.start_episode(seed)
    agent.set_shock_label(None)
    random.seed(seed)
    np.random.seed(seed)

    terminated = False
    truncated = False
    rows: list[dict[str, Any]] = []

    while not (terminated or truncated) and env.state.months_elapsed <= max_months:
        current_month = env.state.months_elapsed
        cash_before = float(env.state.cash)

        raw_action = agent.get_action(env.state)
        decision_trace = agent.get_last_decision_trace() or {}
        clean_action = ActionAdapter.translate_action(raw_action)
        before_metrics = _capture_causal_metrics(env.state, agent)

        _, reward, terminated, truncated, info = env.step(clean_action)

        _write_causal_step_outcome(
            agent=agent,
            clean_action=clean_action,
            before_metrics=before_metrics,
            after_state=env.state,
            episode_seed=seed,
            month=current_month,
        )
        agent.set_shock_label(info.get("shock_label"))

        rows.append(compact_row(
            month=current_month,
            cash_before=cash_before,
            cash_after=float(env.state.cash),
            mrr_after=float(env.state.mrr),
            reward=reward,
            rule_of_40=info.get("rule_of_40"),
            terminated=terminated,
            truncated=truncated,
            decision_trace=decision_trace,
        ))

    run_meta = {
        "policy": policy,
        "seed": seed,
        "max_months": max_months,
        "oracle_frequency": oracle_frequency,
        "months_elapsed": int(env.state.months_elapsed),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_cash": float(env.state.cash),
        "final_mrr": float(env.state.mrr),
        "survived_past_30": (not terminated) and env.state.months_elapsed > max_months,
        "episode_stats": agent.get_episode_stats(),
    }
    return rows, run_meta


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run oracle_v4_causal_hetero for one seed up to N months and "
            "check the three current stabilization gates."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--months", type=int, default=30)
    parser.add_argument("--oracle-frequency", type=int, default=5)
    parser.add_argument("--policy", type=str, default="oracle_v4_causal_hetero")
    parser.add_argument(
        "--output", type=str, default="outputs/oracle_v4_debug_gate/seed0_latest.json",
        help="Path to write the JSON artifact (relative paths resolve against the repo root).",
    )
    # Recalibrated post predicate+object-matching fix, which removed the
    # absorber effect the original 0.05 threshold was set against.
    parser.add_argument("--confidence-spread-threshold", type=float, default=0.02)
    parser.add_argument("--rule40-min-slope", type=float, default=0.10)
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Also dump each check's full result dict as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        rows, run_meta = run_v4_episode_prefix(
            seed=args.seed,
            max_months=args.months,
            oracle_frequency=args.oracle_frequency,
            policy=args.policy,
        )
    except Exception as exc:  # pragma: no cover - exercised only with full stack
        print(f"ORACLE_V4_DEBUG_GATE ERROR seed={args.seed} error={exc!r}")
        return 2

    eval_result = evaluate_all_checks(
        rows,
        confidence_spread_threshold=args.confidence_spread_threshold,
        rule40_min_slope=args.rule40_min_slope,
    )
    artifact = build_artifact(run_meta, rows, eval_result)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")

    print(format_summary_line(eval_result, run_meta))
    for line in format_detail_lines(eval_result):
        print(line)
    if args.verbose:
        for name in _CHECK_ORDER:
            print(f"ORACLE_V4_DEBUG_GATE_DETAIL {name}_full={json.dumps(eval_result['checks'][name], default=str)}")
    print(f"ORACLE_V4_DEBUG_GATE_DETAIL output={output_path}")

    return 0 if eval_result["overall"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
