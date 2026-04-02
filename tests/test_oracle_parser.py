import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from oracle.parser import parse_llm_response
from oracle.schemas import ExpectedOutcome


def test_parser_accepts_v1_shape_without_expected_outcome():
    brief = parse_llm_response(
        '{"risk_level":"MEDIUM","growth_outlook":"STABLE","efficiency_pressure":"MEDIUM",'
        '"innovation_urgency":"HIGH","macro_condition":"NEUTRAL","confidence":0.9}'
    )

    assert brief.expected_outcome is None
    assert brief.innovation_urgency.value == "HIGH"


def test_parser_accepts_v3_shape_with_expected_outcome():
    brief = parse_llm_response(
        '```json {"risk_level":"HIGH","growth_outlook":"DECLINING","efficiency_pressure":"HIGH",'
        '"innovation_urgency":"CRITICAL","macro_condition":"RECESSION",'
        '"expected_outcome":"DECLINE","confidence":0.7} ```'
    )

    assert brief.expected_outcome == ExpectedOutcome.DECLINE
    assert brief.macro_condition.value == "RECESSION"
