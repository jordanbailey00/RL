from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fight_caves_rl.envs.shared_memory_transport import SHARED_MEMORY_TRANSPORT_MODE
from fight_caves_rl.tests.smoke._helpers import (
    load_json,
    offline_wandb_env,
    require_live_runtime,
    run_script,
)

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_shared_memory_transport_train_smoke(tmp_path: Path):
    require_live_runtime()

    config = yaml.safe_load(
        Path("configs/train/train_baseline_v0.yaml").read_text(encoding="utf-8")
    )
    config.setdefault("env", {})["subprocess_transport_mode"] = SHARED_MEMORY_TRANSPORT_MODE
    config["train"]["checkpoint_interval"] = 1_000_000
    config_path = tmp_path / "train_shared_memory.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    summary_path = tmp_path / "train_shared_memory_summary.json"
    data_dir = tmp_path / "train_shared_memory_artifacts"
    result = run_script(
        "train.py",
        "--config",
        str(config_path),
        "--total-timesteps",
        "16",
        "--data-dir",
        str(data_dir),
        "--output",
        str(summary_path),
        env=offline_wandb_env(tmp_path, tags="smoke,train,shared-memory"),
        timeout_seconds=180.0,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Shared-memory train smoke failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    summary = load_json(summary_path)
    assert summary["config_id"] == "train_baseline_v0"
    assert summary["transport_mode"] == SHARED_MEMORY_TRANSPORT_MODE
    assert int(summary["global_step"]) >= 16
