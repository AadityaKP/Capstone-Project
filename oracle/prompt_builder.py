from env.schemas import EnvState
from oracle.schemas import RetrievedMemoryCandidate, TrendContext


def build_prompt(
    state: EnvState,
    mode: str = "oracle_v1",
    trend_context: TrendContext | None = None,
    memories: list[RetrievedMemoryCandidate] | None = None,
    shock_label: str | None = None,
) -> str:
    avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0
    previous_mrr = f"${trend_context.previous_mrr:,.0f}" if trend_context and trend_context.previous_mrr is not None else "N/A"
    current_mrr = f"${trend_context.current_mrr:,.0f}" if trend_context and trend_context.current_mrr is not None else f"${state.mrr:,.0f}"
    mrr_delta_pct = f"{trend_context.mrr_delta_pct:+.1f}%" if trend_context and trend_context.mrr_delta_pct is not None else "N/A"
    previous_churn = f"{trend_context.previous_avg_churn:.3f}" if trend_context and trend_context.previous_avg_churn is not None else "N/A"
    current_churn = f"{trend_context.current_avg_churn:.3f}" if trend_context and trend_context.current_avg_churn is not None else f"{avg_churn:.3f}"
    churn_delta = f"{trend_context.churn_delta:+.3f}" if trend_context and trend_context.churn_delta is not None else "N/A"
    shock_alert = f"ACTIVE SHOCK: {shock_label}" if shock_label and shock_label != "NO_SHOCK" else "No active shocks detected"

    sections = [
        "You are a strategic advisor for a SaaS company.",
        "",
        "Current State:",
        f"- MRR: {state.mrr:,.0f}",
        f"- Cash: {state.cash:,.0f}",
        f"- CAC: {state.cac:.1f}",
        f"- LTV: {state.ltv:,.0f}",
        f"- Churn (Avg): {avg_churn:.3f}",
        f"- Innovation: {state.innovation_factor:.3f}",
        f"- Unemployment: {state.unemployment:.1f}%",
        f"- Confidence: {state.consumer_confidence:.1f}",
        f"- Competitors: {state.competitors}",
        f"- Interest Rate: {state.interest_rate:.1f}%",
    ]

    sections.extend(
        [
            "",
            "--- MARKET CONDITIONS ---",
            f"Competitors in market: {state.competitors}",
            f"Consumer confidence index: {state.consumer_confidence:.1f}",
            f"Interest rate: {state.interest_rate:.1f}%",
            f"Unemployment rate: {state.unemployment:.1f}%",
            f"Months in depression: {state.months_in_depression}",
            "",
            "--- TREND SIGNALS ---",
            f"MRR last period: {previous_mrr}  |  MRR this period: {current_mrr}",
            f"MoM change: {mrr_delta_pct}",
            f"Average churn last period: {previous_churn}  |  Average churn this period: {current_churn}",
            f"Average churn delta: {churn_delta}",
            f"SHOCK_ALERT: {shock_alert}",
        ]
    )

    if mode in {"oracle_v2", "oracle_v3"} and trend_context is not None:
        sections.extend(["", "Recent Trends:"])
        if trend_context.history_points < 2:
            sections.append("- Insufficient history yet; treat current trends as flat.")
        else:
            sections.extend(
                [
                    f"- MRR trend: {trend_context.mrr_trend.value}",
                    f"- Innovation trend: {trend_context.innovation_trend.value}",
                    f"- Churn trend: {trend_context.churn_trend.value}",
                ]
            )

    if mode == "oracle_v3":
        sections.extend(["", "Similar Past Situations:"])
        if memories:
            for index, memory in enumerate(memories, start=1):
                sections.append(
                    f"{index}. weight={memory.memory_weight:.3f}, "
                    f"similarity={memory.similarity_score:.3f}, recency={memory.recency_factor:.3f} :: "
                    f"{memory.document}"
                )
        else:
            sections.append("No similar past situations found yet.")

    sections.extend(
        [
            "",
            "Task:",
            "Analyze the situation and return a JSON object with:",
            "",
            "{",
            '  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",',
            '  "growth_outlook": "ACCELERATING|STABLE|DECLINING|COLLAPSING",',
            '  "efficiency_pressure": "LOW|MEDIUM|HIGH|CRITICAL",',
            '  "innovation_urgency": "LOW|MEDIUM|HIGH|CRITICAL",',
            '  "macro_condition": "EXPANSION|NEUTRAL|RECESSION",',
        ]
    )

    if mode == "oracle_v3":
        sections.append('  "expected_outcome": "GROWTH|STAGNATION|DECLINE",')

    sections.extend(
        [
            '  "key_risks": ["..."],',
            '  "key_opportunities": ["..."],',
            '  "recommended_focus": ["..."],',
            '  "confidence": 0.9',
            "}",
            "",
        ]
    )

    if mode == "oracle_v3":
        sections.append(
            "For expected_outcome, predict whether the next 6-12 months most likely look like "
            "GROWTH, STAGNATION, or DECLINE based on the current state plus similar past situations."
        )

    sections.extend(
        [
            "Memory must add historical evidence, not duplicate the current state.",
            "Ensure valid enums and exactly 0.0 to 1.0 for confidence.",
            "Only output JSON. Do not output markdown blocks or surrounding text.",
        ]
    )

    return "\n".join(sections)
