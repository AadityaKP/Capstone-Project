import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oracle.context import compute_trend_context
from oracle.schemas import StateSnapshot, TrendDirection


def make_snapshot(global_month: int, mrr: float, churn: float, innovation: float) -> StateSnapshot:
    return StateSnapshot(
        global_month=global_month,
        source_month=global_month,
        episode_seed=1,
        mrr=mrr,
        avg_churn=churn,
        innovation=innovation,
    )


def test_trend_context_defaults_to_flat_with_insufficient_history():
    trend_context = compute_trend_context([make_snapshot(0, 100_000, 0.03, 0.50)])

    assert trend_context.history_points == 1
    assert trend_context.mrr_trend == TrendDirection.FLAT
    assert trend_context.innovation_trend == TrendDirection.FLAT
    assert trend_context.churn_trend == TrendDirection.FLAT


def test_trend_context_detects_increasing_signals():
    trend_context = compute_trend_context(
        [
            make_snapshot(0, 100_000, 0.030, 0.50),
            make_snapshot(4, 111_000, 0.033, 0.53),
        ]
    )

    assert trend_context.history_points == 2
    assert trend_context.mrr_trend == TrendDirection.INCREASING
    assert trend_context.innovation_trend == TrendDirection.INCREASING
    assert trend_context.churn_trend == TrendDirection.INCREASING
    assert trend_context.previous_mrr == 100_000
    assert trend_context.current_mrr == 111_000
    assert round(trend_context.mrr_delta_pct, 1) == 11.0
    assert round(trend_context.churn_delta, 3) == 0.003


def test_trend_context_detects_decreasing_and_flat_signals():
    decreasing = compute_trend_context(
        [
            make_snapshot(0, 100_000, 0.030, 0.50),
            make_snapshot(4, 90_000, 0.027, 0.45),
        ]
    )
    flat = compute_trend_context(
        [
            make_snapshot(0, 100_000, 0.030, 0.50),
            make_snapshot(4, 104_000, 0.031, 0.51),
        ]
    )

    assert decreasing.mrr_trend == TrendDirection.DECREASING
    assert decreasing.innovation_trend == TrendDirection.DECREASING
    assert decreasing.churn_trend == TrendDirection.DECREASING

    assert flat.mrr_trend == TrendDirection.FLAT
    assert flat.innovation_trend == TrendDirection.FLAT
    assert flat.churn_trend == TrendDirection.FLAT
