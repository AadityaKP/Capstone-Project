from env import business_logic
from env.schemas import EnvState
from oracle.levels import compute_level_assessment, level_block
from oracle.schemas import GraphContext, RetrievedMemoryCandidate, TrendContext


def _level_sections(state: EnvState, churn_benchmark_pct: float | None) -> list[str]:
    # churn_benchmark_pct arrives as a percentage (e.g. 2.7) when a product
    # surface set it; the level block works in fractions.
    band_median = None if churn_benchmark_pct is None else churn_benchmark_pct / 100.0
    return level_block(compute_level_assessment(state, band_median))


def _runway_clause(state: EnvState) -> str:
    """Months of cash left at the company's monthly burn, as a whole clause.

    Matches Boardroom._estimate_runway_months so the prompt and the guards
    reason about the same number.

    Two things used to be wrong here. The burn was computed as headcount slots
    of $8k inline, so the prompt told the model a founder burning $500/month
    was burning $8,000 and then asked it why the company was in trouble. And
    the caller appended "months of cash" to whatever came back, so a
    cash-flow-positive company produced "Runway: cash-flow positive months of
    cash" - rare while every company was charged $8k/month, and the common case
    now that real costs arrive.
    """
    burn = max(1.0, business_logic.monthly_burn(state))
    net_burn = burn - state.mrr
    if net_burn <= 0:
        return "cash-flow positive - revenue covers the current burn"
    return f"{state.cash / net_burn:.1f} months of cash at the current burn"


def build_prompt(
    state: EnvState,
    mode: str = "oracle_v1",
    trend_context: TrendContext | None = None,
    memories: list[RetrievedMemoryCandidate] | None = None,
    shock_label: str | None = None,
    graph_context: GraphContext | None = None,
    include_burn_context: bool = False,
    churn_benchmark_pct: float | None = None,
    brief_version: str = "v1",
) -> str:
    avg_churn = (state.churn_enterprise + state.churn_smb + state.churn_b2c) / 3.0
    previous_mrr = (
        f"${trend_context.previous_mrr:,.0f}"
        if trend_context and trend_context.previous_mrr is not None
        else "N/A"
    )
    current_mrr = (
        f"${trend_context.current_mrr:,.0f}"
        if trend_context and trend_context.current_mrr is not None
        else f"${state.mrr:,.0f}"
    )
    mrr_delta_pct = (
        f"{trend_context.mrr_delta_pct:+.1f}%"
        if trend_context and trend_context.mrr_delta_pct is not None
        else "N/A"
    )
    previous_churn = (
        f"{trend_context.previous_avg_churn:.3f}"
        if trend_context and trend_context.previous_avg_churn is not None
        else "N/A"
    )
    current_churn = (
        f"{trend_context.current_avg_churn:.3f}"
        if trend_context and trend_context.current_avg_churn is not None
        else f"{avg_churn:.3f}"
    )
    churn_delta = (
        f"{trend_context.churn_delta:+.3f}"
        if trend_context and trend_context.churn_delta is not None
        else "N/A"
    )
    shock_alert = (
        f"ACTIVE SHOCK: {shock_label}"
        if shock_label and shock_label != "NO_SHOCK"
        else "No active shocks detected"
    )

    sections = [
        "You are a strategic advisor for a SaaS company.",
        "",
        "Current State:",
        f"- MRR: {state.mrr:,.0f}",
        f"- Cash: {state.cash:,.0f}",
        # Without burn the model cannot derive runway, and cash alone reads as
        # healthy at any size: $45k looks fine until you see $70k/month of costs.
        # Off by default so research prompts stay byte-identical.
        *(
            [
                f"- Monthly burn: {business_logic.monthly_burn(state):,.0f}",
                f"- Runway: {_runway_clause(state)}",
                f"- Company age: {state.months_elapsed} months",
            ]
            if include_burn_context
            else []
        ),
        # Churn in isolation is uninterpretable: 5%/month is poor at $500 ARPA
        # and unremarkable at $10. The published median for this company's own
        # price point turns an absolute number into a judgement.
        *(
            [
                f"- Median monthly churn for this price point: {churn_benchmark_pct:.1f}%"
                f" (published benchmark; this company is at {avg_churn * 100:.1f}%)"
            ]
            if churn_benchmark_pct is not None
            else []
        ),
        f"- CAC: {state.cac:.1f}",
        f"- LTV: {state.ltv:,.0f}",
        f"- Churn (Avg): {avg_churn:.3f}",
        f"- Innovation: {state.innovation_factor:.3f}",
        f"- Unemployment: {state.unemployment:.1f}%",
        f"- Confidence: {state.consumer_confidence:.1f}",
        f"- Competitors: {state.competitors}",
        f"- Interest Rate: {state.interest_rate:.1f}%",
        # brief v2 (round 2, BRIEF_V2_SPEC.md change 1): a deterministic level
        # block so the model has something to judge levels AGAINST - A4 showed
        # raw levels alone are ignored in favour of trends and the shock line.
        # Off by default ("v1"): research prompts stay byte-identical.
        *(
            ["", *_level_sections(state, churn_benchmark_pct)]
            if brief_version == "v2"
            else []
        ),
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

    if mode in {"oracle_v2", "oracle_v3", "oracle_v4", "oracle_v4_causal"} and trend_context is not None:
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

    if mode == "oracle_v4_causal" and graph_context and graph_context.active_shock_type:
        sections.extend(["", "--- CAUSAL GRAPH CONTEXT ---"])
        sections.append(f"Active shock type: {graph_context.active_shock_type}")

        summary = graph_context.causal_summary
        if summary and summary.total_occurrences > 0:
            sections.extend(
                [
                    f"Historical data for {summary.shock_type} across {summary.total_occurrences} prior episodes:",
                    f"  Mean recovery time: {summary.mean_recovery_months:.1f} months",
                    f"  Recovery rate: {summary.recovery_rate * 100:.0f}%",
                    f"  Mean post-shock Rule-of-40: {summary.mean_post_shock_rule40:.1f}",
                ]
            )
            if summary.best_risk_level:
                sections.append(
                    f"  Fastest recovery observed when risk_level was assessed as: {summary.best_risk_level}"
                )
            if summary.worst_risk_level:
                sections.append(
                    f"  Slowest recovery observed when risk_level was assessed as: {summary.worst_risk_level}"
                )
        else:
            sections.append("  No historical data yet for this shock type.")

        if graph_context.similar_shocks:
            sections.append("Similar past shock events:")
            for rec in graph_context.similar_shocks[:3]:
                sections.append(
                    f"  - Episode {rec.episode_id} month {rec.shock_month} "
                    f"({rec.mrr_tier} tier, risk={rec.brief_risk_level}): "
                    f"recovered={rec.recovered}, recovery_months={rec.recovery_months}, "
                    f"mrr_change={rec.mrr_change_pct:+.1%}"
                )

    if mode in {"oracle_v3", "oracle_v4", "oracle_v4_causal"}:
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

    if mode in {"oracle_v3", "oracle_v4", "oracle_v4_causal"}:
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

    if mode in {"oracle_v3", "oracle_v4", "oracle_v4_causal"}:
        sections.append(
            "For expected_outcome, predict whether the next 6-12 months most likely look like "
            "GROWTH, STAGNATION, or DECLINE based on the current state plus similar past situations."
        )

    if mode == "oracle_v4_causal" and graph_context and graph_context.causal_summary:
        sections.append(
            "Use the CAUSAL GRAPH CONTEXT to inform your risk_level assessment - "
            "historical recovery data is empirically derived, not estimated."
        )

    if brief_version == "v2":
        sections.append(
            "risk_level must be at least as severe as the Runway band and the "
            "LTV:CAC band. macro_condition must agree with the computed Macro regime."
        )

    sections.extend(
        [
            "Memory must add historical evidence, not duplicate the current state.",
            "Ensure valid enums and exactly 0.0 to 1.0 for confidence.",
            "Only output JSON. Do not output markdown blocks or surrounding text.",
        ]
    )

    return "\n".join(sections)
