from __future__ import annotations

from datetime import datetime
from pathlib import Path
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.thesis_analysis import compute_pairwise_significance, ensure_output_dir
from simulation_runner import run_simulation

CONFIRMATION_POLICIES = [
    ("boardroom", "Boardroom Baseline"),
    ("oracle_v1", "Oracle v1"),
    ("oracle_v3", "Oracle v3"),
    ("oracle_v4", "Oracle v4"),
    ("oracle_v4_causal", "Oracle v4 Causal"),
]
RETRIEVAL_TRACE_POLICIES = {"oracle_v3", "oracle_v4"}
DEFAULT_NUM_EPISODES = 50
DEFAULT_ORACLE_FREQUENCY = 5
DEFAULT_ROOT_DIR = Path("results") / "confirmation_runs"


def build_confirmation_output_dir(
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    oracle_frequency: int = DEFAULT_ORACLE_FREQUENCY,
    seed_start: int = 0,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = (
        f"oracle_v4_confirmation__episodes_{num_episodes}"
        f"__freq_{oracle_frequency}__seed_{seed_start}__{timestamp}"
    )
    return ensure_output_dir(Path(root_dir) / folder_name)


def _episode_columns() -> list[str]:
    return [
        "scenario_id",
        "scenario_label",
        "policy",
        "episode",
        "seed",
        "steps",
        "cause",
        "total_reward",
        "final_mrr",
        "final_cash",
        "final_cac",
        "final_ltv",
        "final_ltv_cac",
        "final_headcount",
        "final_valuation_multiple",
        "final_unemployment",
        "final_innovation_factor",
        "depression_months",
        "avg_rule_40",
        "pct_above_40",
        "shock_count",
        "recovered_shock_count",
        "recovered_shock_rate_pct",
        "mean_recovery_time_months",
        "median_recovery_time_months",
        "post_shock_avg_rule40_25_60",
        "oracle_refresh_requests",
        "cadence_refreshes",
        "event_refreshes",
        "cache_hits",
        "llm_calls",
    ]


def _retrieval_columns() -> list[str]:
    return [
        "scenario_id",
        "scenario_label",
        "policy",
        "episode",
        "seed",
        "month",
        "refresh_reason",
        "brief_source",
        "shock_label",
        "retrieval_rank",
        "document",
        "source_month",
        "stored_global_month",
        "realized_outcome",
        "outcome_bucket",
        "memory_weight",
        "similarity_score",
        "recency_factor",
    ]


def _normalize_episode_df(episode_df: pd.DataFrame, policy: str, label: str) -> pd.DataFrame:
    normalized = episode_df.copy()
    normalized["scenario_id"] = policy
    normalized["scenario_label"] = label
    return normalized[_episode_columns()].copy()


def _normalize_retrieval_df(retrieval_rows: list[dict], policy: str, label: str) -> pd.DataFrame:
    if not retrieval_rows:
        return pd.DataFrame(columns=_retrieval_columns())
    retrieval_df = pd.DataFrame(retrieval_rows)
    retrieval_df["scenario_id"] = policy
    retrieval_df["scenario_label"] = label
    return retrieval_df[_retrieval_columns()].copy()


def compute_confirmation_summary(episode_df: pd.DataFrame) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame(
            columns=[
                "scenario_id",
                "Scenario",
                "Episodes",
                "Survival %",
                "Avg Final MRR",
                "Median Final MRR",
                "Avg Rule-40 Post Shock (25-60)",
                "Mean Recovery Time (Mo)",
                "Median Recovery Time (Mo)",
                "Recovered Shock Rate %",
                "Avg LLM Calls",
                "Avg Cache Hits",
            ]
        )

    rows = []
    for scenario_id, group in episode_df.groupby("scenario_id", sort=False):
        rows.append(
            {
                "scenario_id": scenario_id,
                "Scenario": group["scenario_label"].iloc[0],
                "Episodes": int(len(group)),
                "Survival %": (group["cause"] == "Time Limit").mean() * 100.0,
                "Avg Final MRR": group["final_mrr"].mean(),
                "Median Final MRR": group["final_mrr"].median(),
                "Avg Rule-40 Post Shock (25-60)": group["post_shock_avg_rule40_25_60"].mean(),
                "Mean Recovery Time (Mo)": group["mean_recovery_time_months"].mean(),
                "Median Recovery Time (Mo)": group["mean_recovery_time_months"].median(),
                "Recovered Shock Rate %": group["recovered_shock_rate_pct"].mean(),
                "Avg LLM Calls": group["llm_calls"].mean(),
                "Avg Cache Hits": group["cache_hits"].mean(),
            }
        )

    return pd.DataFrame(rows)


def write_confirmation_report(
    output_path: str | Path,
    output_dir: str | Path,
    summary_df: pd.DataFrame,
    significance_df: pd.DataFrame,
    num_episodes: int,
    seed_start: int,
    oracle_frequency: int,
) -> None:
    sections = [
        "# Oracle v4 Confirmation Summary",
        "",
        "## Run Configuration",
        "",
        f"- Output folder: `{Path(output_dir)}`",
        f"- Episodes per policy: {num_episodes}",
        f"- Seed start: {seed_start}",
        f"- Oracle frequency: {oracle_frequency}",
        "- Policies: boardroom, oracle_v1, oracle_v3, oracle_v4, oracle_v4_causal",
        "- Retrieval trace export: oracle_v3 and oracle_v4 only",
        "",
        "## Output Files",
        "",
        "- `primary_summary.csv`",
        "- `primary_episode_metrics.csv`",
        "- `primary_retrieval_trace.csv`",
        "- `thesis_summary_report.md`",
        "",
    ]

    if not summary_df.empty:
        sections.extend(
            [
                "## Primary Summary",
                "",
                summary_df.to_markdown(index=False),
                "",
            ]
        )

    if not significance_df.empty:
        sections.extend(
            [
                "## Significance Tests",
                "",
                "Pairwise Mann-Whitney U tests compare each policy against the boardroom baseline.",
                "",
                significance_df.to_markdown(index=False),
                "",
            ]
        )

    Path(output_path).write_text("\n".join(sections), encoding="utf-8")


def run_oracle_v4_confirmation(
    num_episodes: int = DEFAULT_NUM_EPISODES,
    seed_start: int = 0,
    oracle_frequency: int = DEFAULT_ORACLE_FREQUENCY,
    output_dir: str | Path | None = None,
) -> Path:
    resolved_output_dir = (
        ensure_output_dir(output_dir)
        if output_dir is not None
        else build_confirmation_output_dir(
            num_episodes=num_episodes,
            seed_start=seed_start,
            oracle_frequency=oracle_frequency,
        )
    )

    episode_frames = []
    retrieval_frames = []

    print("==================================================")
    print("ORACLE V4 CONFIRMATION RUN")
    print("==================================================")
    print(f"Output folder: {resolved_output_dir}")
    print(
        f"Policies: {len(CONFIRMATION_POLICIES)} | Episodes per policy: {num_episodes} | "
        f"Oracle frequency: {oracle_frequency} | Seed start: {seed_start}"
    )

    for policy, label in CONFIRMATION_POLICIES:
        print(f"\n>>> Running {label}...")
        needs_retrieval_trace = policy in RETRIEVAL_TRACE_POLICIES
        if needs_retrieval_trace:
            episode_df, retrieval_rows = run_simulation(
                policy=policy,
                num_episodes=num_episodes,
                seed_start=seed_start,
                oracle_frequency=oracle_frequency,
                return_action_trace=False,
                return_monthly_trace=False,
                return_retrieval_trace=True,
            )
            retrieval_frames.append(_normalize_retrieval_df(retrieval_rows, policy=policy, label=label))
        else:
            episode_df = run_simulation(
                policy=policy,
                num_episodes=num_episodes,
                seed_start=seed_start,
                oracle_frequency=oracle_frequency,
                return_action_trace=False,
                return_monthly_trace=False,
                return_retrieval_trace=False,
            )

        episode_frames.append(_normalize_episode_df(episode_df, policy=policy, label=label))

    primary_episode_df = pd.concat(episode_frames, ignore_index=True)
    primary_retrieval_df = (
        pd.concat(retrieval_frames, ignore_index=True)
        if retrieval_frames
        else pd.DataFrame(columns=_retrieval_columns())
    )
    primary_summary_df = compute_confirmation_summary(primary_episode_df)
    significance_df = compute_pairwise_significance(
        primary_episode_df,
        baseline_scenario_id="boardroom",
        metrics=(
            "post_shock_avg_rule40_25_60",
            "mean_recovery_time_months",
            "final_mrr",
        ),
    )

    primary_summary_path = resolved_output_dir / "primary_summary.csv"
    primary_episode_path = resolved_output_dir / "primary_episode_metrics.csv"
    primary_retrieval_path = resolved_output_dir / "primary_retrieval_trace.csv"
    report_path = resolved_output_dir / "thesis_summary_report.md"

    primary_summary_df.to_csv(primary_summary_path, index=False)
    primary_episode_df.to_csv(primary_episode_path, index=False)
    primary_retrieval_df.to_csv(primary_retrieval_path, index=False)
    write_confirmation_report(
        output_path=report_path,
        output_dir=resolved_output_dir,
        summary_df=primary_summary_df,
        significance_df=significance_df,
        num_episodes=num_episodes,
        seed_start=seed_start,
        oracle_frequency=oracle_frequency,
    )

    print("\nSaved confirmation outputs:")
    print(f"- {primary_summary_path.name}")
    print(f"- {primary_episode_path.name}")
    print(f"- {primary_retrieval_path.name}")
    print(f"- {report_path.name}")
    print(f"\nConfirmation output folder: {resolved_output_dir}")

    return resolved_output_dir


if __name__ == "__main__":
    run_oracle_v4_confirmation()
