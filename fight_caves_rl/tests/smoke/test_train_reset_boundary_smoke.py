from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fight_caves_rl.tests.smoke._helpers import (
    load_json,
    offline_wandb_env,
    require_live_runtime,
    run_script,
)

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_train_smoke_crosses_episode_reset_boundary(tmp_path: Path):
    require_live_runtime()

    config = yaml.safe_load(
        Path("configs/train/train_baseline_v0.yaml").read_text(encoding="utf-8")
    )
    config["env"]["tick_cap"] = 16
    config["train"]["checkpoint_interval"] = 1_000_000
    config_path = tmp_path / "train_reset_boundary.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    summary_path = tmp_path / "train_reset_boundary_summary.json"
    data_dir = tmp_path / "train_reset_boundary_artifacts"
    result = run_script(
        "train.py",
        "--config",
        str(config_path),
        "--total-timesteps",
        "64",
        "--data-dir",
        str(data_dir),
        "--output",
        str(summary_path),
        env=offline_wandb_env(tmp_path, tags="smoke,train,reset-boundary"),
        timeout_seconds=180.0,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Reset-boundary train smoke failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    summary = load_json(summary_path)
    assert summary["config_id"] == "train_baseline_v0"
    assert int(summary["global_step"]) >= 64

    manifest = load_json(Path(str(summary["run_manifest_path"])))
    assert manifest["bridge_mode"] == "subprocess_isolated_jvm"
