import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oracle.schemas import (
    ExpectedOutcome,
    GrowthOutlook,
    MacroCondition,
    OracleBrief,
    UrgencyLevel,
)
from oracle.weight_adapter import WeightAdapter


def make_brief(expected_outcome=None):
    return OracleBrief(
        risk_level=UrgencyLevel.HIGH,
        growth_outlook=GrowthOutlook.DECLINING,
        efficiency_pressure=UrgencyLevel.HIGH,
        innovation_urgency=UrgencyLevel.CRITICAL,
        macro_condition=MacroCondition.RECESSION,
        expected_outcome=expected_outcome,
        confidence=1.0,
    )


def test_weight_adapter_preserves_normalization_and_smoothing():
    base_weights = {"efficiency": 0.30, "growth": 0.20, "innovation": 0.40, "macro": 0.10}
    adjusted = WeightAdapter().adjust_weights(base_weights, make_brief(), oracle_mode="oracle_v1")

    assert abs(sum(adjusted.values()) - 1.0) < 1e-9
    assert adjusted["innovation"] > base_weights["innovation"]
    assert adjusted["macro"] < 0.20


def test_only_oracle_v3_gets_stronger_decline_boost():
    base_weights = {"efficiency": 0.30, "growth": 0.20, "innovation": 0.40, "macro": 0.10}
    v1 = WeightAdapter().adjust_weights(
        base_weights,
        make_brief(expected_outcome=ExpectedOutcome.DECLINE),
        oracle_mode="oracle_v1",
    )
    v3 = WeightAdapter().adjust_weights(
        base_weights,
        make_brief(expected_outcome=ExpectedOutcome.DECLINE),
        oracle_mode="oracle_v3",
    )

    v1_delta = sum(abs(v1[key] - base_weights[key]) for key in base_weights)
    v3_delta = sum(abs(v3[key] - base_weights[key]) for key in base_weights)

    assert v3["innovation"] > v1["innovation"]
    assert v3["efficiency"] > v1["efficiency"]
    assert v3_delta > v1_delta
