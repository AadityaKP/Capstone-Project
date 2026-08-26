from __future__ import annotations

from pathlib import Path
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.thesis_analysis import (
    ABLATION_SCENARIOS,
    MEMORY_ABLATION_SCENARIOS,
    OUTPUT_DIR,
    POST_SHOCK_END_MONTH,
    POST_SHOCK_START_MONTH,
    PRIMARY_SCENARIOS,
    compute_ablation_summary,
    compute_decision_divergence,
    compute_decision_map,
    compute_episode_level_primary_metrics,
    compute_pairwise_significance,
    compute_primary_summary,
    compute_recovery_events,
    compute_retrieval_quality,
    compute_reward_curve,
    ensure_output_dir,
    run_policy_suite,
    save_plot_pack,
    write_case_study_report,
    write_summary_report,
)


def run_thesis_experiment(
    mode: str = "eval",
    num_episodes: int | None = None,
    case_study_warmup: int | None = None,
    seed_start: int = 0,
    oracle_frequency: int = 10,
    output_dir: str | Path = OUTPUT_DIR,
    include_primary: bool = True,
    include_ablation: bool = True,
    include_case_study: bool = True,
    include_memory_ablation: bool = False,
    deterministic_rng: bool = True,
) -> Path:
    """
    deterministic_rng gives every scenario in a suite the same world for a given
    seed. It defaults to True here, unlike the StartupEnv flag it sets, because
    a comparison run is exactly the case where the old behaviour is wrong: the
    physics otherwise draws from a stream that policies perturb and that
    desynchronises on its own once macro conditions diverge. Pass False only to
    reproduce a pre-existing result bit-for-bit.

    include_memory_ablation runs the four-arm suite that isolates the memory
    architecture (none / episodic / semantic / full). Off by default because it
    is a separate question from the version comparison in ABLATION_SCENARIOS and
    costs another full pass.
    """
    print("==================================================")
    print("ORACLE THESIS ANALYSIS SUITE")
    print("==================================================")

    environment_config = {"deterministic_rng": True} if deterministic_rng else None
    if deterministic_rng:
        print("[CONFIG] deterministic_rng=True - policies share one world per seed.")
    else:
        print("[CONFIG] deterministic_rng=False - LEGACY. Policies do NOT share a world;\n"
              "         cross-policy differences are confounded. Reproduction only.")

    mode = mode.lower()
    if mode == "dev":
        resolved_num_episodes = num_episodes or 20
        resolved_case_study_warmup = case_study_warmup or 5
    else:
        resolved_num_episodes = num_episodes or 200
        resolved_case_study_warmup = case_study_warmup or 12

    output_dir = ensure_output_dir(output_dir)

    print(
        f"\n[CONFIG] Mode: {mode.upper()} | Episodes: {resolved_num_episodes} | "
        f"Oracle Freq: {oracle_frequency} months | Post-shock window: {POST_SHOCK_START_MONTH}-{POST_SHOCK_END_MONTH}"
    )

    primary_episode_df = primary_action_df = primary_monthly_df = primary_retrieval_df = None
    primary_recovery_df = primary_reward_curve_df = primary_summary_df = None
    decision_summary_df = decision_detail_df = retrieval_quality_df = decision_map_df = None
    primary_episode_metric_df = significance_df = None
    ablation_episode_df = ablation_action_df = ablation_monthly_df = ablation_recovery_df = ablation_summary_df = None
    memory_episode_df = memory_monthly_df = memory_recovery_df = None
    memory_summary_df = memory_metric_df = memory_significance_df = None
    case_episode_df = case_action_slice = case_monthly_slice = case_retrieval_slice = None

    if include_primary:
        print("\n=== PRIMARY EXPERIMENT ===")
        primary_episode_df, primary_action_df, primary_monthly_df, primary_retrieval_df = run_policy_suite(
            scenarios=PRIMARY_SCENARIOS,
            num_episodes=resolved_num_episodes,
            seed_start=seed_start,
            oracle_frequency=oracle_frequency,
            environment_config=environment_config,
        )

        primary_recovery_df = compute_recovery_events(primary_monthly_df)
        primary_reward_curve_df = compute_reward_curve(primary_monthly_df)
        decision_summary_df, decision_detail_df = compute_decision_divergence(primary_action_df, reference_scenario_id="boardroom")
        primary_episode_metric_df = compute_episode_level_primary_metrics(primary_episode_df, primary_monthly_df)
        significance_df = compute_pairwise_significance(primary_episode_metric_df, baseline_scenario_id="boardroom")
        primary_summary_df = compute_primary_summary(
            episode_df=primary_episode_df,
            monthly_df=primary_monthly_df,
            recovery_df=primary_recovery_df,
            decision_summary_df=decision_summary_df,
        )
        retrieval_quality_df = compute_retrieval_quality(primary_retrieval_df)
        decision_map_df = compute_decision_map(primary_action_df)

    if include_ablation:
        print("\n=== ABLATION STUDY ===")
        ablation_episode_df, ablation_action_df, ablation_monthly_df, _ = run_policy_suite(
            scenarios=ABLATION_SCENARIOS,
            num_episodes=resolved_num_episodes,
            seed_start=seed_start,
            oracle_frequency=oracle_frequency,
            environment_config=environment_config,
        )
        ablation_recovery_df = compute_recovery_events(ablation_monthly_df)
        ablation_summary_df = compute_ablation_summary(
            episode_df=ablation_episode_df,
            monthly_df=ablation_monthly_df,
            recovery_df=ablation_recovery_df,
        )

    if include_memory_ablation:
        print("\n=== MEMORY ABLATION (V3 attribution) ===")
        memory_episode_df, _, memory_monthly_df, _ = run_policy_suite(
            scenarios=MEMORY_ABLATION_SCENARIOS,
            num_episodes=resolved_num_episodes,
            seed_start=seed_start,
            oracle_frequency=oracle_frequency,
            environment_config=environment_config,
        )
        memory_recovery_df = compute_recovery_events(memory_monthly_df)
        memory_summary_df = compute_ablation_summary(
            episode_df=memory_episode_df,
            monthly_df=memory_monthly_df,
            recovery_df=memory_recovery_df,
        )
        # Effect sizes, confidence intervals and a multiple-comparison
        # correction, against the no-memory arm. The architecture claim rests on
        # this table, so it reports how big the difference is and not only
        # whether it cleared a threshold.
        memory_metric_df = compute_episode_level_primary_metrics(
            memory_episode_df, memory_monthly_df
        )
        # Primary metric first: months to recover to the pre-shock level is the
        # quantity the memory claim is actually about. The rest are secondary
        # and are reported alongside rather than instead.
        memory_significance_df = compute_pairwise_significance(
            memory_metric_df,
            baseline_scenario_id="memory_none",
            metrics=(
                "mean_recovery_time_months",   # primary
                "total_reward",
                "final_cash",
                "steps",                       # survival proxy
                "post_shock_avg_rule40",
            ),
        )

    if include_case_study:
        print("\n=== CASE STUDY ===")
        case_episode_df, case_action_df, case_monthly_df, case_retrieval_df = run_policy_suite(
            scenarios=[
                {
                    "scenario_id": "oracle_v3_case_study",
                    "policy": "oracle_v3",
                    "label": "Oracle v3 Case Study",
                    "oracle_overrides": {},
                }
            ],
            num_episodes=resolved_case_study_warmup + 1,
            seed_start=seed_start,
            oracle_frequency=oracle_frequency,
        )
        case_episode = case_action_df["episode"].max() if not case_action_df.empty else 0
        case_action_slice = case_action_df[case_action_df["episode"] == case_episode].copy()
        case_monthly_slice = case_monthly_df[case_monthly_df["episode"] == case_episode].copy()
        case_retrieval_slice = case_retrieval_df[case_retrieval_df["episode"] == case_episode].copy()

    print("\n=== SAVING OUTPUTS ===")
    if include_primary and primary_episode_df is not None:
        primary_episode_df.to_csv(output_dir / "primary_episode_metrics.csv", index=False)
        primary_action_df.to_csv(output_dir / "primary_action_trace.csv", index=False)
        primary_monthly_df.to_csv(output_dir / "primary_monthly_trace.csv", index=False)
        primary_retrieval_df.to_csv(output_dir / "primary_retrieval_trace.csv", index=False)
        primary_recovery_df.to_csv(output_dir / "primary_recovery_events.csv", index=False)
        primary_reward_curve_df.to_csv(output_dir / "primary_reward_curve.csv", index=False)
        primary_summary_df.to_csv(output_dir / "primary_summary.csv", index=False)
        primary_episode_metric_df.to_csv(output_dir / "primary_episode_metric_summary.csv", index=False)
        significance_df.to_csv(output_dir / "primary_significance_tests.csv", index=False)
        decision_summary_df.to_csv(output_dir / "decision_difference_summary.csv", index=False)
        decision_detail_df.to_csv(output_dir / "decision_difference_detail.csv", index=False)
        retrieval_quality_df.to_csv(output_dir / "retrieval_quality.csv", index=False)
        decision_map_df.to_csv(output_dir / "oracle_decision_map.csv", index=False)

    if include_ablation and ablation_episode_df is not None:
        ablation_episode_df.to_csv(output_dir / "ablation_episode_metrics.csv", index=False)
        ablation_action_df.to_csv(output_dir / "ablation_action_trace.csv", index=False)
        ablation_monthly_df.to_csv(output_dir / "ablation_monthly_trace.csv", index=False)
        ablation_recovery_df.to_csv(output_dir / "ablation_recovery_events.csv", index=False)
        ablation_summary_df.to_csv(output_dir / "ablation_summary.csv", index=False)

    if include_memory_ablation and memory_episode_df is not None:
        memory_episode_df.to_csv(output_dir / "memory_ablation_episode_metrics.csv", index=False)
        memory_monthly_df.to_csv(output_dir / "memory_ablation_monthly_trace.csv", index=False)
        memory_recovery_df.to_csv(output_dir / "memory_ablation_recovery_events.csv", index=False)
        memory_summary_df.to_csv(output_dir / "memory_ablation_summary.csv", index=False)
        memory_metric_df.to_csv(output_dir / "memory_ablation_episode_metric_summary.csv", index=False)
        memory_significance_df.to_csv(output_dir / "memory_ablation_significance.csv", index=False)

    if include_case_study and case_episode_df is not None:
        case_episode_df.to_csv(output_dir / "case_study_episode_metrics.csv", index=False)
        case_action_slice.to_csv(output_dir / "case_study_action_trace.csv", index=False)
        case_monthly_slice.to_csv(output_dir / "case_study_monthly_trace.csv", index=False)
        case_retrieval_slice.to_csv(output_dir / "case_study_retrieval_trace.csv", index=False)
        write_case_study_report(
            action_df=case_action_slice,
            monthly_df=case_monthly_slice,
            output_path=output_dir / "case_study_report.md",
        )

    write_summary_report(
        output_path=output_dir / "thesis_summary_report.md",
        primary_summary_df=primary_summary_df,
        ablation_summary_df=ablation_summary_df,
        significance_df=significance_df,
    )

    save_plot_pack(
        output_dir=output_dir,
        reward_curve_df=primary_reward_curve_df if primary_reward_curve_df is not None else None,
        recovery_df=primary_recovery_df if primary_recovery_df is not None else None,
        ablation_summary_df=ablation_summary_df if ablation_summary_df is not None else None,
        retrieval_quality_df=retrieval_quality_df if retrieval_quality_df is not None else None,
        decision_map_df=decision_map_df if decision_map_df is not None else None,
    )

    if include_primary and primary_summary_df is not None:
        print("\n=== PRIMARY SUMMARY ===")
        print(primary_summary_df.to_markdown(index=False))
    if include_ablation and ablation_summary_df is not None:
        print("\n=== ABLATION SUMMARY ===")
        print(ablation_summary_df.to_markdown(index=False))
    print(f"\nSaved thesis-analysis outputs to {output_dir}")

    return output_dir


if __name__ == "__main__":
    run_thesis_experiment()
