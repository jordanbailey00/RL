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


def test_fast_train_smoke(tmp_path: Path):
    require_live_runtime()

    summary_path = tmp_path / "train_fast_v2_summary.json"
    data_dir = tmp_path / "train_fast_v2_artifacts"
    result = run_script(
        "train.py",
        "--config",
        "configs/train/smoke_fast_v2.yaml",
        "--total-timesteps",
        "16",
        "--data-dir",
        str(data_dir),
        "--output",
        str(summary_path),
        env=offline_wandb_env(tmp_path, tags="smoke,train,v2_fast"),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Fast train smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    summary = load_json(summary_path)
    assert summary["transport_mode"] == "v2_fast_subprocess_shared_memory_v1"
    assert int(summary["global_step"]) >= 16
    assert summary["vecenv_topology"]["env_backend"] == "v2_fast"
    assert summary["vecenv_topology"]["transport_mode"] == "shared_memory_v1"
    assert summary["vecenv_topology"]["worker_count"] == 2
