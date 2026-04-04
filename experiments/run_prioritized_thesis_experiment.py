from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_thesis_experiment import run_thesis_experiment

DEFAULT_CASE_STUDY_WARMUP = 12
DEFAULT_PRIMARY_EPISODES = 75
DEFAULT_ORACLE_FREQUENCY = 10
DEFAULT_ROOT_DIR = Path("results") / "future_experiments" / "prioritized_thesis_run"


def make_run_root(root_dir: str | Path = DEFAULT_ROOT_DIR) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(root_dir) / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    return run_root


def build_primary_background_command(
    output_dir: str | Path,
    num_episodes: int = DEFAULT_PRIMARY_EPISODES,
    oracle_frequency: int = DEFAULT_ORACLE_FREQUENCY,
    seed_start: int = 0,
) -> list[str]:
    output_path = Path(output_dir)
    code = (
        "from experiments.run_thesis_experiment import run_thesis_experiment; "
        f"run_thesis_experiment(mode='eval', num_episodes={num_episodes}, seed_start={seed_start}, "
        f"oracle_frequency={oracle_frequency}, output_dir=r'{output_path}', "
        "include_primary=True, include_ablation=False, include_case_study=False)"
    )
    return [sys.executable, "-c", code]


def launch_primary_background(
    output_dir: str | Path,
    num_episodes: int = DEFAULT_PRIMARY_EPISODES,
    oracle_frequency: int = DEFAULT_ORACLE_FREQUENCY,
    seed_start: int = 0,
    log_path: str | Path | None = None,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    resolved_log_path = Path(log_path) if log_path is not None else output_path / "primary_background.log"
    command = build_primary_background_command(
        output_dir=output_path,
        num_episodes=num_episodes,
        oracle_frequency=oracle_frequency,
        seed_start=seed_start,
    )

    with open(resolved_log_path, "w", encoding="utf-8") as log_file:
        creationflags = 0
        if sys.platform == "win32":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS

        process = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[1],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
            close_fds=True,
        )

    metadata = {
        "kind": "primary_background_run",
        "pid": process.pid,
        "command": command,
        "output_dir": str(output_path),
        "log_path": str(resolved_log_path),
        "num_episodes": num_episodes,
        "oracle_frequency": oracle_frequency,
        "seed_start": seed_start,
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path = output_path / "primary_background_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def run_primary_in_terminal(
    output_dir: str | Path,
    num_episodes: int = DEFAULT_PRIMARY_EPISODES,
    oracle_frequency: int = DEFAULT_ORACLE_FREQUENCY,
    seed_start: int = 0,
) -> dict:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")

    run_thesis_experiment(
        mode="eval",
        num_episodes=num_episodes,
        seed_start=seed_start,
        oracle_frequency=oracle_frequency,
        output_dir=output_path,
        include_primary=True,
        include_ablation=False,
        include_case_study=False,
    )

    metadata = {
        "kind": "primary_terminal_run",
        "output_dir": str(output_path),
        "num_episodes": num_episodes,
        "oracle_frequency": oracle_frequency,
        "seed_start": seed_start,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path = output_path / "primary_terminal_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def run_prioritized_thesis_experiment(
    root_dir: str | Path = DEFAULT_ROOT_DIR,
    case_study_warmup: int = DEFAULT_CASE_STUDY_WARMUP,
    primary_num_episodes: int = DEFAULT_PRIMARY_EPISODES,
    oracle_frequency: int = DEFAULT_ORACLE_FREQUENCY,
    seed_start: int = 0,
    include_case_study: bool = True,
    run_in_background: bool = True,
) -> dict:
    run_root = make_run_root(root_dir)
    case_study_dir = run_root / "case_study"
    primary_dir_name = "primary_background" if run_in_background else "primary_terminal"
    primary_dir = run_root / primary_dir_name

    print("==================================================")
    print("PRIORITIZED THESIS RUN")
    print("==================================================")
    if include_case_study:
        print(f"\n[PHASE 1] Running case study first -> {case_study_dir}")
        run_thesis_experiment(
            mode="eval",
            case_study_warmup=case_study_warmup,
            seed_start=seed_start,
            oracle_frequency=oracle_frequency,
            output_dir=case_study_dir,
            include_primary=False,
            include_ablation=False,
            include_case_study=True,
        )
    else:
        print("\n[PHASE 1] Skipping case study for this run.")

    primary_phase_label = "[PHASE 2]" if include_case_study else "[PHASE 1]"
    if run_in_background:
        print(f"\n{primary_phase_label} Launching primary experiment in background -> {primary_dir}")
        primary_metadata = launch_primary_background(
            output_dir=primary_dir,
            num_episodes=primary_num_episodes,
            oracle_frequency=oracle_frequency,
            seed_start=seed_start,
        )
    else:
        print(f"\n{primary_phase_label} Running primary experiment in this terminal -> {primary_dir}")
        primary_metadata = run_primary_in_terminal(
            output_dir=primary_dir,
            num_episodes=primary_num_episodes,
            oracle_frequency=oracle_frequency,
            seed_start=seed_start,
        )

    summary = {
        "run_root": str(run_root),
        "primary_dir": str(primary_dir),
    }
    if include_case_study:
        summary["case_study_dir"] = str(case_study_dir)
    if run_in_background:
        summary["primary_background"] = primary_metadata
    else:
        summary["primary_terminal"] = primary_metadata

    summary_path = run_root / "prioritized_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if include_case_study:
        print("\nCase study outputs are ready for analysis.")
    if run_in_background:
        print(f"Background primary PID: {primary_metadata['pid']}")
        print(f"Background primary log: {primary_metadata['log_path']}")
    else:
        print(f"Primary terminal metadata: {primary_metadata['metadata_path']}")
    print(f"Run summary: {summary_path}")
    return summary


if __name__ == "__main__":
    run_prioritized_thesis_experiment(
        primary_num_episodes=75,
        include_case_study=False,
        run_in_background=False,
    )
