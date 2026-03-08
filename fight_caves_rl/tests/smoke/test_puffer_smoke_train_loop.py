from __future__ import annotations

from pathlib import Path

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script


def test_puffer_smoke_train_loop(tmp_path: Path):
    require_live_runtime()

    summary_path = tmp_path / "train_summary.json"
    data_dir = tmp_path / "train_artifacts"
    result = run_script(
        "train.py",
        "--config",
        "configs/train/smoke_ppo_v0.yaml",
        "--total-timesteps",
        "4",
        "--data-dir",
        str(data_dir),
        "--output",
        str(summary_path),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Train smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    summary = load_json(summary_path)
    checkpoint_path = Path(str(summary["checkpoint_path"]))
    metadata_path = Path(str(summary["checkpoint_metadata_path"]))

    assert summary["config_id"] == "smoke_ppo_v0"
    assert int(summary["global_step"]) >= 4
    assert checkpoint_path.exists()
    assert metadata_path.exists()
    assert int(summary["log_records"]) >= 1
