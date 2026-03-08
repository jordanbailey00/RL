from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import (
    load_json,
    offline_wandb_env,
    require_live_runtime,
    run_script,
)

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_multi_worker_smoke(tmp_path: Path):
    require_live_runtime()

    summary_path = tmp_path / "train_baseline_summary.json"
    data_dir = tmp_path / "train_baseline_artifacts"
    result = run_script(
        "train.py",
        "--config",
        "configs/train/train_baseline_v0.yaml",
        "--total-timesteps",
        "16",
        "--data-dir",
        str(data_dir),
        "--output",
        str(summary_path),
        env=offline_wandb_env(tmp_path, tags="smoke,train,vecenv"),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Vectorized train smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    summary = load_json(summary_path)
    assert summary["config_id"] == "train_baseline_v0"
    assert int(summary["global_step"]) >= 16
    assert int(summary["log_records"]) >= 1

