from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_wandb_offline_smoke_writes_local_run_and_artifact_dirs(tmp_path: Path):
    require_live_runtime()

    summary_path = tmp_path / "train_summary.json"
    wandb_dir = tmp_path / "wandb"
    wandb_data_dir = tmp_path / "wandb-data"
    wandb_cache_dir = tmp_path / "wandb-cache"
    result = run_script(
        "train.py",
        "--config",
        "configs/train/smoke_ppo_v0.yaml",
        "--total-timesteps",
        "4",
        "--data-dir",
        str(tmp_path / "train_artifacts"),
        "--output",
        str(summary_path),
        env={
            "WANDB_PROJECT": "fight-caves-rl-tests",
            "WANDB_MODE": "offline",
            "WANDB_GROUP": "pr6",
            "WANDB_TAGS": "integration,offline",
            "WANDB_DIR": str(wandb_dir),
            "WANDB_DATA_DIR": str(wandb_data_dir),
            "WANDB_CACHE_DIR": str(wandb_cache_dir),
        },
    )
    if result.returncode != 0:
        raise AssertionError(
            f"W&B offline smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    summary = load_json(summary_path)
    assert summary["wandb_run_id"]
    assert Path(str(summary["run_manifest_path"])).exists()
    wandb_run_root = wandb_dir / "wandb"
    assert any(wandb_run_root.glob("offline-run-*"))
    assert (wandb_run_root / "latest-run").exists()
    assert (wandb_data_dir / "artifacts" / "staging").is_dir()
    assert (wandb_cache_dir / "wandb" / "logs").is_dir()
