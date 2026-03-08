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


def test_replay_generation_smoke(tmp_path: Path):
    require_live_runtime()

    train_summary_path = tmp_path / "train_summary.json"
    replay_summary_path = tmp_path / "replay_eval_summary.json"
    train_result = run_script(
        "train.py",
        "--config",
        "configs/train/smoke_ppo_v0.yaml",
        "--total-timesteps",
        "4",
        "--data-dir",
        str(tmp_path / "train_artifacts"),
        "--output",
        str(train_summary_path),
        env=offline_wandb_env(tmp_path, tags="integration,replay-generation"),
    )
    if train_result.returncode != 0:
        raise AssertionError(
            f"Train replay-generation smoke failed.\nstdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}"
        )

    checkpoint_path = str(load_json(train_summary_path)["checkpoint_path"])
    replay_result = run_script(
        "replay_eval.py",
        "--checkpoint",
        checkpoint_path,
        "--config",
        "configs/eval/replay_eval_v0.yaml",
        "--max-steps",
        "16",
        "--output",
        str(replay_summary_path),
        env=offline_wandb_env(tmp_path, tags="integration,replay-eval"),
    )
    if replay_result.returncode != 0:
        raise AssertionError(
            f"Replay eval smoke failed.\nstdout:\n{replay_result.stdout}\nstderr:\n{replay_result.stderr}"
        )

    summary = load_json(replay_summary_path)
    replay_pack = load_json(Path(str(summary["replay_pack_path"])))
    replay_index = load_json(Path(str(summary["replay_index_path"])))

    assert summary["config_id"] == "replay_eval_v0"
    assert summary["seed_pack"] == "bootstrap_smoke"
    assert summary["replay_step_cadence"] == 1
    assert replay_pack["schema_id"] == "replay_pack_v0"
    assert replay_index["schema_id"] == "replay_index_v0"
    assert replay_pack["summary_digest"] == summary["summary_digest"]
    assert replay_index["summary_digest"] == summary["summary_digest"]
    assert len(replay_pack["episodes"]) == 2
    assert len(replay_index["entries"]) == 2
    assert replay_index["replay_pack_filename"] == "replay_pack.json"
