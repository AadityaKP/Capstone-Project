from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

from simulation_runner import run_simulation

POST_SHOCK_START_MONTH = 25
POST_SHOCK_END_MONTH = 60
RECOVERY_REFERENCE_OFFSET = 1
OUTPUT_DIR = Path("results") / "future_experiments" / "thesis_analysis"

PRIMARY_SCENARIOS = [
    {
        "scenario_id": "boardroom",
        "policy": "boardroom",
        "label": "Boardroom Baseline",
        "oracle_overrides": {},
    },
    {
        "scenario_id": "oracle_v1",
        "policy": "oracle_v1",
        "label": "Oracle v1",
        "oracle_overrides": {},
    },
    {
        "scenario_id": "oracle_v3",
        "policy": "oracle_v3",
        "label": "Oracle v3",
        "oracle_overrides": {},
    },
]

ABLATION_SCENARIOS = [
    {
        "scenario_id": "oracle_v1_action_modifier",
        "policy": "oracle_v1",
        "label": "Oracle v1 + ActionModifier",
        "oracle_overrides": {},
    },
    {
        "scenario_id": "oracle_v1_weights_only",
        "policy": "oracle_v1_no_modifier",
        "label": "Oracle v1 Weights Only",
        "oracle_overrides": {},
    },
    {
        "scenario_id": "oracle_v3_no_memory",
        "policy": "oracle_v3_no_memory",
        "label": "Oracle v3 No Memory",
        "oracle_overrides": {},
    },
    {
        "scenario_id": "oracle_v3_full",
        "policy": "oracle_v3",
        "label": "Oracle v3 Full",
        "oracle_overrides": {},
    },
]

# V3 attribution: does the MEMORY ARCHITECTURE cause the improvement?
#
# ABLATION_SCENARIOS above varies Oracle versions, which answers a different
# question - it compares releases, not components. Isolating the architecture
# needs the two memory systems crossed against each other, because the project
# has two and they are not the same thing:
#
#   episodic  - OracleMemoryStore, Chroma, retrieved past run snapshots
#   semantic  - CausalGraphStore, Neo4j, CONFIRMED_CAUSE / MAY_CAUSE edges
#
# Run this suite with environment_config={"deterministic_rng": True} and one
# seed list, or the arms are not comparable.
#
# Note on the semantic arm: CausalGraphStore seeds MAY_CAUSE priors by hand
# before any live evidence exists, so "semantic only" is partly a hand-authored
# prior rather than something learned. Any claim from this arm has to say so.
MEMORY_ABLATION_SCENARIOS = [
    {
        "scenario_id": "memory_none",
        "policy": "boardroom",
        "label": "No memory (boardroom only)",
        "oracle_overrides": {},
    },
    {
        "scenario_id": "memory_episodic_only",
        "policy": "oracle_v3",
        "label": "Episodic only (Chroma)",
        "oracle_overrides": {},
    },
    {
        "scenario_id": "memory_semantic_only",
        "policy": "oracle_v4_causal_no_memory",
        "label": "Semantic only (causal graph)",
        "oracle_overrides": {},
    },
    {
        "scenario_id": "memory_full",
        "policy": "oracle_v4_causal",
        "label": "Full memory (episodic + semantic)",
        "oracle_overrides": {},
    },
]


def ensure_output_dir(output_dir: str | Path = OUTPUT_DIR) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _nested_get(payload: dict[str, Any] | None, path: list[str], default=None):
    current = payload or {}
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _normalize_trace_rows(rows: list[dict[str, Any]], scenario: dict[str, Any]) -> list[dict[str, Any]]:
    normalized = []
    for row in rows:
        enriched = dict(row)
        enriched["scenario_id"] = scenario["scenario_id"]
        enriched["scenario_label"] = scenario["label"]
        normalized.append(enriched)
    return normalized


def flatten_action_trace(action_trace: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in action_trace:
        decision_trace = row.get("decision_trace") or {}
        brief = decision_trace.get("brief") or row.get("brief") or {}
        pre_modifier = decision_trace.get("pre_modifier_action") or {}
        post_modifier = decision_trace.get("post_modifier_action") or {}
        final_action = decision_trace.get("final_action") or row.get("action") or {}

        rows.append(
            {
                "scenario_id": row.get("scenario_id"),
                "scenario_label": row.get("scenario_label"),
                "policy": row.get("policy"),
                "episode": row.get("episode"),
                "seed": row.get("seed"),
                "month": row.get("month"),
                "used_oracle": decision_trace.get("used_oracle", False),
                "oracle_mode": decision_trace.get("oracle_mode"),
                "refresh_reason": decision_trace.get("refresh_reason"),
                "brief_source": decision_trace.get("brief_source"),
                "shock_label": decision_trace.get("shock_label"),
                "brief_risk_level": brief.get("risk_level"),
                "brief_growth_outlook": brief.get("growth_outlook"),
                "brief_efficiency_pressure": brief.get("efficiency_pressure"),
                "brief_innovation_urgency": brief.get("innovation_urgency"),
                "brief_macro_condition": brief.get("macro_condition"),
                "brief_expected_outcome": brief.get("expected_outcome"),
                "brief_confidence": brief.get("confidence"),
                "action_modifier_applied": decision_trace.get("action_modifier_applied", False),
                "memory_count": decision_trace.get("memory_count", 0),
                "marketing_spend_pre_modifier": _nested_get(pre_modifier, ["marketing", "spend"]),
                "marketing_spend_post_modifier": _nested_get(post_modifier, ["marketing", "spend"]),
                "marketing_spend_final": _nested_get(final_action, ["marketing", "spend"]),
                "rd_spend_pre_modifier": _nested_get(pre_modifier, ["product", "r_and_d_spend"]),
                "rd_spend_post_modifier": _nested_get(post_modifier, ["product", "r_and_d_spend"]),
                "rd_spend_final": _nested_get(final_action, ["product", "r_and_d_spend"]),
                "hires_pre_modifier": _nested_get(pre_modifier, ["hiring", "hires"], 0),
                "hires_post_modifier": _nested_get(post_modifier, ["hiring", "hires"], 0),
                "hires_final": _nested_get(final_action, ["hiring", "hires"], 0),
                "marketing_spend_change_pct": decision_trace.get("marketing_spend_change_pct"),
                "rd_spend_change_pct": decision_trace.get("rd_spend_change_pct"),
                "hires_change": decision_trace.get("hires_change"),
            }
        )

    return pd.DataFrame(rows)


def flatten_monthly_trace(monthly_trace: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in monthly_trace:
        decision_trace = row.get("decision_trace") or {}
        brief = decision_trace.get("brief") or row.get("brief") or {}
        rows.append(
            {
                "scenario_id": row.get("scenario_id"),
                "scenario_label": row.get("scenario_label"),
                "policy": row.get("policy"),
                "episode": row.get("episode"),
                "seed": row.get("seed"),
                "month": row.get("month"),
                "reward": row.get("reward"),
                "rule_of_40": row.get("rule_of_40"),
                "shock_label": row.get("shock_label", "NO_SHOCK"),
                "terminated": row.get("terminated", False),
                "truncated": row.get("truncated", False),
                "mrr": row.get("mrr"),
                "cash": row.get("cash"),
                "innovation_factor": row.get("innovation_factor"),
                "unemployment": row.get("unemployment"),
                "months_in_depression": row.get("months_in_depression"),
                "brief_risk_level": brief.get("risk_level"),
                "brief_source": decision_trace.get("brief_source"),
            }
        )
    return pd.DataFrame(rows)


def flatten_retrieval_trace(action_trace: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in action_trace:
        decision_trace = row.get("decision_trace") or {}
        for index, memory in enumerate(decision_trace.get("retrieved_memories") or [], start=1):
            metadata = memory.get("metadata") or {}
            realized_outcome = metadata.get("realized_outcome")
            rows.append(
                {
                    "scenario_id": row.get("scenario_id"),
                    "scenario_label": row.get("scenario_label"),
                    "policy": row.get("policy"),
                    "episode": row.get("episode"),
                    "seed": row.get("seed"),
                    "month": row.get("month"),
                    "retrieval_rank": index,
                    "document": memory.get("document"),
                    "source_month": metadata.get("source_month"),
                    "stored_global_month": metadata.get("stored_global_month"),
                    "realized_outcome": realized_outcome,
                    "outcome_bucket": map_outcome_bucket(realized_outcome),
                    "memory_weight": memory.get("memory_weight"),
                    "similarity_score": memory.get("similarity_score"),
                    "recency_factor": memory.get("recency_factor"),
                }
            )
    return pd.DataFrame(rows)


def map_outcome_bucket(realized_outcome: str | None) -> str:
    if realized_outcome == "GROWTH":
        return "POSITIVE"
    if realized_outcome == "DECLINE":
        return "NEGATIVE"
    return "NEUTRAL"


def compute_recovery_events(monthly_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_df.empty:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "scenario_label",
                "policy",
                "episode",
                "seed",
                "shock_month",
                "shock_label",
                "pre_shock_rule_40",
                "recovery_month",
                "recovery_time_months",
                "recovered",
            ]
        )

    records = []
    group_cols = ["scenario_id", "scenario_label", "policy", "episode", "seed"]
    for keys, group in monthly_df.groupby(group_cols):
        ordered = group.sort_values("month").reset_index(drop=True)
        by_month = {int(row["month"]): row for _, row in ordered.iterrows()}
        shock_rows = ordered[ordered["shock_label"].fillna("NO_SHOCK") != "NO_SHOCK"]
        for _, shock_row in shock_rows.iterrows():
            shock_month = int(shock_row["month"])
            baseline_month = shock_month - RECOVERY_REFERENCE_OFFSET
            baseline_row = by_month.get(baseline_month)
            pre_shock_rule = baseline_row["rule_of_40"] if baseline_row is not None else np.nan

            future_rows = ordered[ordered["month"] > shock_month]
            recovered_rows = future_rows[future_rows["rule_of_40"] >= pre_shock_rule] if pd.notna(pre_shock_rule) else pd.DataFrame()
            if recovered_rows.empty:
                recovery_month = np.nan
                recovery_time = np.nan
                recovered = False
            else:
                recovery_month = float(recovered_rows.iloc[0]["month"])
                recovery_time = recovery_month - shock_month
                recovered = True

            records.append(
                {
                    "scenario_id": keys[0],
                    "scenario_label": keys[1],
                    "policy": keys[2],
                    "episode": keys[3],
                    "seed": keys[4],
                    "shock_month": shock_month,
                    "shock_label": shock_row["shock_label"],
                    "pre_shock_rule_40": pre_shock_rule,
                    "recovery_month": recovery_month,
                    "recovery_time_months": recovery_time,
                    "recovered": recovered,
                }
            )

    return pd.DataFrame(records)


def compute_reward_curve(monthly_df: pd.DataFrame) -> pd.DataFrame:
    if monthly_df.empty:
        return pd.DataFrame()

    grouped = (
        monthly_df.groupby(["scenario_id", "scenario_label", "policy", "month"])
        .agg(
            reward_mean=("reward", "mean"),
            reward_std=("reward", "std"),
            rule40_mean=("rule_of_40", "mean"),
            rule40_std=("rule_of_40", "std"),
            n=("reward", "count"),
        )
        .reset_index()
    )
    grouped["reward_std"] = grouped["reward_std"].fillna(0.0)
    grouped["rule40_std"] = grouped["rule40_std"].fillna(0.0)
    grouped["reward_ci95"] = 1.96 * grouped["reward_std"] / np.sqrt(grouped["n"].clip(lower=1))
    grouped["rule40_ci95"] = 1.96 * grouped["rule40_std"] / np.sqrt(grouped["n"].clip(lower=1))
    return grouped


def compute_decision_divergence(action_df: pd.DataFrame, reference_scenario_id: str = "boardroom") -> tuple[pd.DataFrame, pd.DataFrame]:
    if action_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    reference = action_df[action_df["scenario_id"] == reference_scenario_id][
        [
            "seed",
            "month",
            "marketing_spend_final",
            "rd_spend_final",
            "hires_final",
        ]
    ].rename(
        columns={
            "marketing_spend_final": "reference_marketing_spend",
            "rd_spend_final": "reference_rd_spend",
            "hires_final": "reference_hires",
        }
    )

    detail_frames = []
    summary_rows = []
    for scenario_id, group in action_df.groupby("scenario_id"):
        if scenario_id == reference_scenario_id:
            continue

        merged = group.merge(reference, on=["seed", "month"], how="inner")
        if merged.empty:
            continue

        merged["marketing_diff_pct"] = (
            (merged["marketing_spend_final"] - merged["reference_marketing_spend"]).abs()
            / merged["reference_marketing_spend"].abs().clip(lower=1.0)
        )
        merged["rd_diff_pct"] = (
            (merged["rd_spend_final"] - merged["reference_rd_spend"]).abs()
            / merged["reference_rd_spend"].abs().clip(lower=1.0)
        )
        merged["hires_diff"] = (merged["hires_final"] - merged["reference_hires"]).abs()
        merged["decision_diff"] = (
            (merged["marketing_diff_pct"] > 0.05)
            | (merged["rd_diff_pct"] > 0.05)
            | (merged["hires_diff"] > 0)
        )
        detail_frames.append(merged)
        summary_rows.append(
            {
                "scenario_id": scenario_id,
                "scenario_label": merged["scenario_label"].iloc[0],
                "decision_difference_rate_pct": merged["decision_diff"].mean() * 100.0,
                "avg_marketing_diff_pct": merged["marketing_diff_pct"].mean() * 100.0,
                "avg_rd_diff_pct": merged["rd_diff_pct"].mean() * 100.0,
                "avg_hires_diff": merged["hires_diff"].mean(),
            }
        )

    detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()
    summary_df = pd.DataFrame(summary_rows)
    return summary_df, detail_df


def compute_retrieval_quality(retrieval_df: pd.DataFrame) -> pd.DataFrame:
    if retrieval_df.empty:
        return pd.DataFrame()

    quality_df = (
        retrieval_df.groupby(["scenario_id", "scenario_label", "episode", "outcome_bucket"])
        .size()
        .reset_index(name="retrieval_count")
    )
    totals = (
        quality_df.groupby(["scenario_id", "scenario_label", "episode"])["retrieval_count"]
        .sum()
        .reset_index(name="episode_total")
    )
    quality_df = quality_df.merge(totals, on=["scenario_id", "scenario_label", "episode"], how="left")
    quality_df["retrieval_share"] = quality_df["retrieval_count"] / quality_df["episode_total"].clip(lower=1)
    return quality_df


def compute_decision_map(action_df: pd.DataFrame) -> pd.DataFrame:
    if action_df.empty:
        return pd.DataFrame()

    return action_df[
        (action_df["used_oracle"])
        & (action_df["brief_risk_level"].notna())
        & (action_df["action_modifier_applied"])
    ][
        [
            "scenario_id",
            "scenario_label",
            "policy",
            "episode",
            "seed",
            "month",
            "brief_risk_level",
            "brief_confidence",
            "marketing_spend_change_pct",
            "rd_spend_change_pct",
            "memory_count",
            "brief_source",
        ]
    ].copy()


def compute_primary_summary(
    episode_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
    decision_summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows = []
    decision_summary_df = decision_summary_df if decision_summary_df is not None else pd.DataFrame()
    for scenario_id, group in episode_df.groupby("scenario_id"):
        scenario_label = group["scenario_label"].iloc[0]
        monthly_group = monthly_df[monthly_df["scenario_id"] == scenario_id]
        recovery_group = recovery_df[recovery_df["scenario_id"] == scenario_id]
        decision_row = decision_summary_df[decision_summary_df["scenario_id"] == scenario_id]

        post_shock_window = monthly_group[
            monthly_group["month"].between(POST_SHOCK_START_MONTH, POST_SHOCK_END_MONTH)
        ]
        decision_diff_rate = decision_row["decision_difference_rate_pct"].iloc[0] if not decision_row.empty else np.nan

        rows.append(
            {
                "scenario_id": scenario_id,
                "Scenario": scenario_label,
                "Survival %": (group["cause"] == "Time Limit").mean() * 100.0,
                "Avg Final MRR": group["final_mrr"].mean(),
                "Median Final MRR": group["final_mrr"].median(),
                "Avg Rule-40 Post Shock (25-60)": post_shock_window["rule_of_40"].mean(),
                "Mean Recovery Time (Mo)": recovery_group["recovery_time_months"].mean(),
                "Median Recovery Time (Mo)": recovery_group["recovery_time_months"].median(),
                "Recovered %": recovery_group["recovered"].mean() * 100.0 if not recovery_group.empty else np.nan,
                "Decision Difference vs Boardroom %": decision_diff_rate,
                "Avg LLM Calls": group["llm_calls"].mean() if "llm_calls" in group else 0.0,
                "Avg Cache Hits": group["cache_hits"].mean() if "cache_hits" in group else 0.0,
            }
        )

    return pd.DataFrame(rows)


def compute_episode_level_primary_metrics(
    episode_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame()

    post_shock_window = monthly_df[
        monthly_df["month"].between(POST_SHOCK_START_MONTH, POST_SHOCK_END_MONTH)
    ]
    post_shock_summary = (
        post_shock_window.groupby(["scenario_id", "scenario_label", "policy", "episode", "seed"])
        .agg(post_shock_avg_rule40=("rule_of_40", "mean"))
        .reset_index()
    )

    merged = episode_df.merge(
        post_shock_summary,
        on=["scenario_id", "scenario_label", "policy", "episode", "seed"],
        how="left",
    )
    return merged


def significance_test(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = "post_shock_avg_rule40") -> dict[str, float | bool | str]:
    sample_a = pd.to_numeric(df_a.get(metric), errors="coerce").dropna().to_numpy()
    sample_b = pd.to_numeric(df_b.get(metric), errors="coerce").dropna().to_numpy()
    if len(sample_a) == 0 or len(sample_b) == 0:
        return {
            "metric": metric,
            "U": np.nan,
            "p_value": np.nan,
            "significant": False,
            "n_a": int(len(sample_a)),
            "n_b": int(len(sample_b)),
            "method": "insufficient_data",
        }

    if stats is not None:
        result = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
        p_value = float(result.pvalue)
        u_stat = float(result.statistic)
        method = "scipy_mannwhitneyu"
    else:
        u_stat, p_value = _mann_whitney_fallback(sample_a, sample_b)
        method = "fallback_normal_approx"

    return {
        "metric": metric,
        "U": u_stat,
        "p_value": p_value,
        "significant": bool(p_value < 0.05),
        "n_a": int(len(sample_a)),
        "n_b": int(len(sample_b)),
        "method": method,
    }


def cohens_d(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    """Standardised mean difference, pooled SD, Hedges-corrected for small n.

    A p-value says an effect is unlikely to be chance; it says nothing about
    whether the effect is large enough to matter. With 200 seeds a trivial
    difference clears p<0.05 easily, so an ablation reporting only p-values
    cannot distinguish "the memory architecture helps" from "the memory
    architecture helps by an amount no one would notice".
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    var_a, var_b = np.var(sample_a, ddof=1), np.var(sample_b, ddof=1)
    pooled = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled == 0:
        return 0.0
    d = (np.mean(sample_b) - np.mean(sample_a)) / pooled
    # Hedges' g: d is biased upward at small n.
    correction = 1.0 - (3.0 / (4.0 * (n_a + n_b) - 9.0))
    return float(d * correction)


def effect_magnitude(d: float) -> str:
    """Cohen's conventional bands. Reported as a word so a reader does not have
    to remember what 0.5 means."""
    if math.isnan(d):
        return "unknown"
    magnitude = abs(d)
    if magnitude < 0.2:
        return "negligible"
    if magnitude < 0.5:
        return "small"
    if magnitude < 0.8:
        return "medium"
    return "large"


def mean_difference_ci(
    sample_a: np.ndarray, sample_b: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Welch confidence interval on the difference in means (b - a).

    Welch rather than Student because ablation arms routinely differ in
    variance - an arm that goes bankrupt more often has a wider spread by
    construction - and pooling that variance would understate the interval.
    Returns (difference, low, high).
    """
    n_a, n_b = len(sample_a), len(sample_b)
    if n_a < 2 or n_b < 2:
        return float("nan"), float("nan"), float("nan")
    mean_a, mean_b = np.mean(sample_a), np.mean(sample_b)
    var_a, var_b = np.var(sample_a, ddof=1), np.var(sample_b, ddof=1)
    se = math.sqrt(var_a / n_a + var_b / n_b)
    difference = float(mean_b - mean_a)
    if se == 0:
        return difference, difference, difference

    # Welch-Satterthwaite degrees of freedom.
    df = (var_a / n_a + var_b / n_b) ** 2 / (
        (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    )
    if stats is not None:
        critical = float(stats.t.ppf(0.5 + confidence / 2.0, df))
    else:
        critical = 1.96  # normal approximation; only reached without scipy
    return difference, difference - critical * se, difference + critical * se


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values.

    An ablation compares several arms against one baseline on several metrics,
    so the family-wise error rate is not the per-test alpha. Holm is used rather
    than plain Bonferroni because it is uniformly more powerful at the same
    guarantee, and rather than Benjamini-Hochberg because with a handful of
    planned comparisons controlling the family-wise rate is the stricter and
    more defensible choice.
    """
    indexed = sorted(
        ((p, i) for i, p in enumerate(p_values) if not math.isnan(p)),
        key=lambda pair: pair[0],
    )
    adjusted = [float("nan")] * len(p_values)
    m = len(indexed)
    running = 0.0
    for rank, (p, index) in enumerate(indexed):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)  # enforce monotonicity
        adjusted[index] = running
    return adjusted


def _mann_whitney_fallback(sample_a: np.ndarray, sample_b: np.ndarray) -> tuple[float, float]:
    combined = np.concatenate([sample_a, sample_b])
    ranks = pd.Series(combined).rank(method="average").to_numpy()
    n_a = len(sample_a)
    n_b = len(sample_b)
    rank_sum_a = ranks[:n_a].sum()
    u_a = rank_sum_a - (n_a * (n_a + 1) / 2.0)
    mean_u = n_a * n_b / 2.0
    sigma_u = math.sqrt(max((n_a * n_b * (n_a + n_b + 1)) / 12.0, 1e-9))
    z_score = (u_a - mean_u) / sigma_u
    p_value = math.erfc(abs(z_score) / math.sqrt(2.0))
    return float(u_a), float(p_value)


def compute_pairwise_significance(
    episode_metric_df: pd.DataFrame,
    baseline_scenario_id: str = "boardroom",
    metrics: tuple[str, ...] = ("post_shock_avg_rule40", "final_mrr"),
) -> pd.DataFrame:
    if episode_metric_df.empty:
        return pd.DataFrame()

    baseline_df = episode_metric_df[episode_metric_df["scenario_id"] == baseline_scenario_id]
    if baseline_df.empty:
        return pd.DataFrame()

    rows = []
    for scenario_id, candidate_df in episode_metric_df.groupby("scenario_id"):
        if scenario_id == baseline_scenario_id:
            continue
        for metric in metrics:
            result = significance_test(baseline_df, candidate_df, metric=metric)

            sample_a = pd.to_numeric(baseline_df.get(metric), errors="coerce").dropna().to_numpy()
            sample_b = pd.to_numeric(candidate_df.get(metric), errors="coerce").dropna().to_numpy()
            d = cohens_d(sample_a, sample_b)
            difference, ci_low, ci_high = mean_difference_ci(sample_a, sample_b)

            result.update(
                {
                    "baseline_scenario_id": baseline_scenario_id,
                    "comparison_scenario_id": scenario_id,
                    "comparison_scenario_label": candidate_df["scenario_label"].iloc[0],
                    "baseline_mean": float(np.mean(sample_a)) if len(sample_a) else float("nan"),
                    "comparison_mean": float(np.mean(sample_b)) if len(sample_b) else float("nan"),
                    "mean_difference": difference,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "cohens_d": d,
                    "effect_magnitude": effect_magnitude(d),
                }
            )
            rows.append(result)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    # Every row is one comparison in a single planned family, so the reported
    # p-value has to be corrected before any of them is called significant.
    frame["p_value_adjusted"] = holm_bonferroni(frame["p_value"].tolist())
    frame["significant"] = frame["p_value_adjusted"] < 0.05
    frame["correction"] = "holm-bonferroni"
    frame["family_size"] = len(frame)
    return frame


def compute_ablation_summary(
    episode_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    recovery_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for scenario_id, group in episode_df.groupby("scenario_id"):
        monthly_group = monthly_df[monthly_df["scenario_id"] == scenario_id]
        recovery_group = recovery_df[recovery_df["scenario_id"] == scenario_id]
        post_shock_window = monthly_group[
            monthly_group["month"].between(POST_SHOCK_START_MONTH, POST_SHOCK_END_MONTH)
        ]
        rows.append(
            {
                "scenario_id": scenario_id,
                "Scenario": group["scenario_label"].iloc[0],
                "Avg Rule-40 Post Shock (25-60)": post_shock_window["rule_of_40"].mean(),
                "Mean Recovery Time (Mo)": recovery_group["recovery_time_months"].mean(),
                "Survival %": (group["cause"] == "Time Limit").mean() * 100.0,
                "Avg Final MRR": group["final_mrr"].mean(),
            }
        )
    return pd.DataFrame(rows)


def write_case_study_report(
    action_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    if action_df.empty or monthly_df.empty:
        return

    case_episode = int(action_df["episode"].max())
    case_actions = action_df[action_df["episode"] == case_episode].sort_values("month")
    case_monthly = monthly_df[monthly_df["episode"] == case_episode].sort_values("month")
    shock_rows = case_monthly[case_monthly["shock_label"] != "NO_SHOCK"]

    lines = [
        f"# Oracle v3 Case Study (Episode {case_episode})",
        "",
        "This report highlights shock detections, Oracle briefs, retrieved memories, and the resulting action changes.",
        "",
    ]

    if shock_rows.empty:
        lines.append("No hard shocks were detected in the selected case-study episode.")
    else:
        for _, shock_row in shock_rows.iterrows():
            month = int(shock_row["month"])
            action_row = case_actions[case_actions["month"] == month]
            lines.append(f"## Month {month} | {shock_row['shock_label']}")
            if action_row.empty:
                lines.append("No action trace was recorded for this month.")
                lines.append("")
                continue

            action_row = action_row.iloc[0]
            lines.append(
                f"- Risk level: {action_row['brief_risk_level']} | Growth outlook: {action_row['brief_growth_outlook']} | "
                f"Confidence: {action_row['brief_confidence']}"
            )
            lines.append(
                f"- Marketing change: {action_row['marketing_spend_change_pct']:.1f}% | "
                f"R&D change: {action_row['rd_spend_change_pct']:.1f}% | Hires change: {action_row['hires_change']}"
            )
            lines.append(f"- Retrieved memories: {int(action_row['memory_count'])}")
            lines.append("")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def write_summary_report(
    output_path: str | Path,
    primary_summary_df: pd.DataFrame | None = None,
    ablation_summary_df: pd.DataFrame | None = None,
    significance_df: pd.DataFrame | None = None,
) -> None:
    sections = ["# Oracle Thesis Analysis Summary", ""]

    if primary_summary_df is not None and not primary_summary_df.empty:
        sections.extend(
            [
                "## Primary Summary",
                "",
                primary_summary_df.to_markdown(index=False),
                "",
            ]
        )

    if significance_df is not None and not significance_df.empty:
        sections.extend(
            [
                "## Significance Tests",
                "",
                "Pairwise Mann-Whitney U tests compare each Oracle policy against the boardroom baseline.",
                "",
                significance_df.to_markdown(index=False),
                "",
            ]
        )

    if ablation_summary_df is not None and not ablation_summary_df.empty:
        sections.extend(
            [
                "## Ablation Summary",
                "",
                ablation_summary_df.to_markdown(index=False),
                "",
            ]
        )

    Path(output_path).write_text("\n".join(sections), encoding="utf-8")


def save_plot_pack(
    output_dir: str | Path,
    reward_curve_df: pd.DataFrame | None,
    recovery_df: pd.DataFrame | None,
    ablation_summary_df: pd.DataFrame | None,
    retrieval_quality_df: pd.DataFrame | None,
    decision_map_df: pd.DataFrame | None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[thesis_analysis] matplotlib not installed; skipping plot generation.")
        return

    output_path = ensure_output_dir(output_dir)

    if reward_curve_df is not None and not reward_curve_df.empty:
        plt.figure(figsize=(10, 6))
        for label, group in reward_curve_df.groupby("scenario_label"):
            ordered = group.sort_values("month")
            plt.plot(ordered["month"], ordered["reward_mean"], label=label)
            plt.fill_between(
                ordered["month"],
                ordered["reward_mean"] - ordered["reward_ci95"],
                ordered["reward_mean"] + ordered["reward_ci95"],
                alpha=0.15,
            )
        plt.title("Global Reward Over Time")
        plt.xlabel("Month")
        plt.ylabel("Average Reward")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "plot_1_global_reward_over_time.png", dpi=150)
        plt.close()

    if recovery_df is not None and not recovery_df.empty:
        plt.figure(figsize=(10, 6))
        for label, group in recovery_df.groupby("scenario_label"):
            values = group["recovery_time_months"].dropna()
            if values.empty:
                continue
            plt.hist(values, bins=10, alpha=0.5, label=label)
        plt.title("Time-to-Recovery Histogram")
        plt.xlabel("Recovery Time (Months)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "plot_2_time_to_recovery_histogram.png", dpi=150)
        plt.close()

    if ablation_summary_df is not None and not ablation_summary_df.empty:
        plt.figure(figsize=(10, 6))
        ordered = ablation_summary_df.sort_values("Avg Rule-40 Post Shock (25-60)", ascending=False)
        plt.bar(ordered["Scenario"], ordered["Avg Rule-40 Post Shock (25-60)"])
        plt.title("Ablation Bar Chart")
        plt.ylabel("Avg Rule-40 Post Shock (25-60)")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(output_path / "plot_3_ablation_bar_chart.png", dpi=150)
        plt.close()

    if retrieval_quality_df is not None and not retrieval_quality_df.empty:
        plt.figure(figsize=(10, 6))
        filtered = retrieval_quality_df[retrieval_quality_df["scenario_id"].isin(["oracle_v3", "oracle_v3_full"])]
        for outcome_bucket, group in filtered.groupby("outcome_bucket"):
            episode_share = (
                group.groupby("episode")["retrieval_share"]
                .mean()
                .reset_index()
                .sort_values("episode")
            )
            plt.plot(episode_share["episode"], episode_share["retrieval_share"], label=outcome_bucket)
        plt.title("Memory Retrieval Quality")
        plt.xlabel("Episode")
        plt.ylabel("Average Retrieval Share")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path / "plot_4_memory_retrieval_quality.png", dpi=150)
        plt.close()

    if decision_map_df is not None and not decision_map_df.empty:
        risk_mapping = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        plot_df = decision_map_df.copy()
        plot_df["risk_numeric"] = plot_df["brief_risk_level"].map(risk_mapping)
        plt.figure(figsize=(10, 6))
        plt.scatter(plot_df["risk_numeric"], plot_df["marketing_spend_change_pct"], alpha=0.5)
        plt.xticks(list(risk_mapping.values()), list(risk_mapping.keys()))
        plt.title("Oracle Decision Map")
        plt.xlabel("Oracle Risk Level")
        plt.ylabel("Marketing Spend Change (%)")
        plt.tight_layout()
        plt.savefig(output_path / "plot_5_oracle_decision_map.png", dpi=150)
        plt.close()


def run_policy_suite(
    scenarios: list[dict[str, Any]],
    num_episodes: int,
    seed_start: int,
    oracle_frequency: int,
    environment_config: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run every scenario over the same seed list.

    `environment_config` reaches StartupEnv unchanged and is the same for every
    scenario in the suite, which is the point: pass
    {"deterministic_rng": True} and the policies genuinely share a world, so a
    difference between them is attributable to the policy. Without it they do
    not - see StartupEnv.deterministic_rng.
    """
    episode_frames = []
    raw_action_trace = []
    raw_monthly_trace = []

    for scenario in scenarios:
        print(f"\n>>> Running {scenario['label']}...")
        episode_df, trace_payload = run_simulation(
            policy=scenario["policy"],
            num_episodes=num_episodes,
            seed_start=seed_start,
            oracle_frequency=oracle_frequency,
            oracle_overrides=scenario.get("oracle_overrides") or {},
            return_action_trace=True,
            return_monthly_trace=True,
            environment_config=environment_config,
        )
        episode_df = episode_df.copy()
        episode_df["scenario_id"] = scenario["scenario_id"]
        episode_df["scenario_label"] = scenario["label"]
        episode_frames.append(episode_df)
        raw_action_trace.extend(_normalize_trace_rows(trace_payload["action_trace"], scenario))
        raw_monthly_trace.extend(_normalize_trace_rows(trace_payload["monthly_trace"], scenario))

    episode_df = pd.concat(episode_frames, ignore_index=True) if episode_frames else pd.DataFrame()
    action_df = flatten_action_trace(raw_action_trace)
    monthly_df = flatten_monthly_trace(raw_monthly_trace)
    retrieval_df = flatten_retrieval_trace(raw_action_trace)
    return episode_df, action_df, monthly_df, retrieval_df
