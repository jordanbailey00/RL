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


def test_fast_eval_loop_smoke(tmp_path: Path):
    require_live_runtime()

    train_summary_path = tmp_path / "fast_train_summary.json"
    eval_path = tmp_path / "fast_eval_summary.json"
    train_result = run_script(
        "train.py",
        "--config",
        "configs/train/smoke_fast_v2.yaml",
        "--total-timesteps",
        "16",
        "--data-dir",
        str(tmp_path / "fast_train_artifacts"),
        "--output",
        str(train_summary_path),
        env=offline_wandb_env(tmp_path, tags="smoke,eval-train,v2-fast"),
        timeout_seconds=240.0,
    )
    if train_result.returncode != 0:
        raise AssertionError(
            f"Fast train smoke failed.\nstdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}"
        )

    checkpoint_path = str(load_json(train_summary_path)["checkpoint_path"])
    eval_result = run_script(
        "eval.py",
        "--checkpoint",
        checkpoint_path,
        "--config",
        "configs/eval/parity_fast_v2.yaml",
        "--max-steps",
        "16",
        "--output",
        str(eval_path),
        env=offline_wandb_env(tmp_path, tags="smoke,eval,v2-fast"),
        timeout_seconds=240.0,
    )
    if eval_result.returncode != 0:
        raise AssertionError(
            f"Fast eval smoke failed.\nstdout:\n{eval_result.stdout}\nstderr:\n{eval_result.stderr}"
        )

    summary = load_json(eval_path)
    assert summary["config_id"] == "parity_fast_v2"
    assert summary["reward_config_id"] == "reward_sparse_v2"
    assert summary["runtime_reward_config_id"] == "reward_sparse_v0"
    assert len(summary["per_seed"]) == 2
    assert summary["summary_digest"]
