from typing import Sequence

from env.schemas import EnvState
from oracle.schemas import StateSnapshot, TrendContext, TrendDirection

MRR_TREND_THRESHOLD = 0.05
INNOVATION_TREND_THRESHOLD = 0.02
CHURN_TREND_THRESHOLD = 0.002


def snapshot_state(state: EnvState, global_month: int, episode_seed: int | None = None) -> StateSnapshot:
    avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0
    return StateSnapshot(
        global_month=global_month,
        source_month=state.months_elapsed,
        episode_seed=episode_seed,
        mrr=state.mrr,
        avg_churn=avg_churn,
        innovation=state.innovation_factor,
    )


def _classify_relative_delta(current: float, baseline: float, threshold: float) -> TrendDirection:
    if abs(baseline) < 1e-9:
        if abs(current) < 1e-9:
            return TrendDirection.FLAT
        return TrendDirection.INCREASING if current > 0 else TrendDirection.DECREASING

    delta = (current - baseline) / abs(baseline)
    if delta > threshold:
        return TrendDirection.INCREASING
    if delta < (-threshold):
        return TrendDirection.DECREASING
    return TrendDirection.FLAT


def _classify_absolute_delta(current: float, baseline: float, threshold: float) -> TrendDirection:
    delta = current - baseline
    if delta > threshold:
        return TrendDirection.INCREASING
    if delta < (-threshold):
        return TrendDirection.DECREASING
    return TrendDirection.FLAT


def compute_trend_context(history: Sequence[StateSnapshot]) -> TrendContext:
    history_points = max(len(history), 1)
    if len(history) < 2:
        return TrendContext(history_points=history_points)

    oldest = history[0]
    previous = history[-2]
    current = history[-1]
    previous_mrr = previous.mrr
    current_mrr = current.mrr
    baseline_mrr = max(abs(previous_mrr), 1.0)
    mrr_delta_pct = ((current_mrr - previous_mrr) / baseline_mrr) * 100.0
    previous_avg_churn = previous.avg_churn
    current_avg_churn = current.avg_churn
    churn_delta = current_avg_churn - previous_avg_churn

    return TrendContext(
        mrr_trend=_classify_relative_delta(current.mrr, oldest.mrr, MRR_TREND_THRESHOLD),
        innovation_trend=_classify_absolute_delta(
            current.innovation,
            oldest.innovation,
            INNOVATION_TREND_THRESHOLD,
        ),
        churn_trend=_classify_absolute_delta(
            current.avg_churn,
            oldest.avg_churn,
            CHURN_TREND_THRESHOLD,
        ),
        history_points=history_points,
        previous_mrr=previous_mrr,
        current_mrr=current_mrr,
        mrr_delta_pct=mrr_delta_pct,
        previous_avg_churn=previous_avg_churn,
        current_avg_churn=current_avg_churn,
        churn_delta=churn_delta,
    )
