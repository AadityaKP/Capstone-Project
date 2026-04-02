import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.schemas import EnvState
from oracle.prompt_builder import build_prompt
from oracle.schemas import RetrievedMemoryCandidate, TrendContext, TrendDirection


def make_state() -> EnvState:
    return EnvState(
        mrr=100_000.0,
        cash=1_000_000.0,
        cac=120.0,
        ltv=700.0,
        churn_enterprise=0.02,
        churn_smb=0.03,
        churn_b2c=0.04,
        interest_rate=4.0,
        consumer_confidence=98.0,
        competitors=6,
        product_quality=0.5,
        price=50.0,
        months_elapsed=3,
        headcount=6,
        valuation_multiple=10.0,
        unemployment=4.0,
        innovation_factor=0.6,
        months_in_depression=0,
    )


def test_prompt_modes_include_expected_sections():
    state = make_state()
    trends = TrendContext(
        mrr_trend=TrendDirection.DECREASING,
        innovation_trend=TrendDirection.FLAT,
        churn_trend=TrendDirection.INCREASING,
        history_points=5,
        previous_mrr=120_000.0,
        current_mrr=100_000.0,
        mrr_delta_pct=-16.7,
        previous_avg_churn=0.025,
        current_avg_churn=0.030,
        churn_delta=0.005,
    )
    memories = [
        RetrievedMemoryCandidate(
            document="Past decline after churn spike.",
            memory_weight=0.81,
            similarity_score=0.9,
            recency_factor=0.9,
        )
    ]

    v1_prompt = build_prompt(state, mode="oracle_v1")
    v2_prompt = build_prompt(state, mode="oracle_v2", trend_context=trends)
    v3_prompt = build_prompt(state, mode="oracle_v3", trend_context=trends, memories=memories, shock_label="RATE_HIKE")

    assert "Recent Trends:" not in v1_prompt
    assert "Similar Past Situations:" not in v1_prompt
    assert '"expected_outcome"' not in v1_prompt

    assert "Recent Trends:" in v2_prompt
    assert "Similar Past Situations:" not in v2_prompt
    assert '"expected_outcome"' not in v2_prompt
    assert "--- MARKET CONDITIONS ---" in v2_prompt
    assert "--- TREND SIGNALS ---" in v2_prompt
    assert "MRR last period: $120,000" in v2_prompt
    assert "MoM change: -16.7%" in v2_prompt
    assert "Average churn delta: +0.005" in v2_prompt
    assert "SHOCK_ALERT: No active shocks detected" in v2_prompt

    assert "Recent Trends:" in v3_prompt
    assert "Similar Past Situations:" in v3_prompt
    assert '"expected_outcome"' in v3_prompt
    assert "Past decline after churn spike." in v3_prompt
    assert "SHOCK_ALERT: ACTIVE SHOCK: RATE_HIKE" in v3_prompt
