import json
import os
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import experiments.run_prioritized_thesis_experiment as prioritized


def test_build_primary_background_command_targets_primary_only_output_dir(tmp_path):
    output_dir = tmp_path / "primary_background"
    command = prioritized.build_primary_background_command(
        output_dir=output_dir,
        num_episodes=50,
        oracle_frequency=10,
        seed_start=3,
    )

    assert command[0] == sys.executable
    assert command[1] == "-c"
    assert "num_episodes=50" in command[2]
    assert "include_primary=True" in command[2]
    assert "include_ablation=False" in command[2]
    assert "include_case_study=False" in command[2]
    assert f"output_dir=r'{output_dir}'" in command[2]


def test_run_prioritized_experiment_runs_case_study_then_background_primary(tmp_path, monkeypatch):
    calls = []

    def fake_run_thesis_experiment(**kwargs):
        calls.append(("case_study", kwargs))
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        return Path(kwargs["output_dir"])

    def fake_launch_primary_background(**kwargs):
        calls.append(("background", kwargs))
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "pid": 4321,
            "log_path": str(output_dir / "primary_background.log"),
            "metadata_path": str(output_dir / "primary_background_metadata.json"),
        }
        Path(metadata["metadata_path"]).write_text(json.dumps(metadata), encoding="utf-8")
        return metadata

    monkeypatch.setattr(prioritized, "run_thesis_experiment", fake_run_thesis_experiment)
    monkeypatch.setattr(prioritized, "launch_primary_background", fake_launch_primary_background)

    summary = prioritized.run_prioritized_thesis_experiment(
        root_dir=tmp_path,
        case_study_warmup=7,
        primary_num_episodes=50,
        oracle_frequency=10,
        seed_start=2,
    )

    assert [label for label, _ in calls] == ["case_study", "background"]

    case_kwargs = calls[0][1]
    assert case_kwargs["include_primary"] is False
    assert case_kwargs["include_ablation"] is False
    assert case_kwargs["include_case_study"] is True
    assert case_kwargs["case_study_warmup"] == 7

    background_kwargs = calls[1][1]
    assert background_kwargs["num_episodes"] == 50
    assert background_kwargs["oracle_frequency"] == 10
    assert background_kwargs["seed_start"] == 2

    summary_path = Path(summary["run_root"]) / "prioritized_run_summary.json"
    assert summary_path.exists()
