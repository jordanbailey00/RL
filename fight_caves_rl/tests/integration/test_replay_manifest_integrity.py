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


def test_replay_manifest_integrity(tmp_path: Path):
    require_live_runtime()

    train_summary_path = tmp_path / "train_summary.json"
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
        env=offline_wandb_env(tmp_path, tags="integration,replay-manifest-train"),
    )
    if train_result.returncode != 0:
        raise AssertionError(
            f"Replay manifest train setup failed.\nstdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}"
        )

    checkpoint_path = str(load_json(train_summary_path)["checkpoint_path"])
    replay_summary_path = tmp_path / "replay_eval_summary.json"
    result = run_script(
        "replay_eval.py",
        "--checkpoint",
        checkpoint_path,
        "--config",
        "configs/eval/replay_eval_v0.yaml",
        "--max-steps",
        "16",
        "--output",
        str(replay_summary_path),
        env=offline_wandb_env(tmp_path, tags="integration,replay-manifest"),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Replay manifest integrity run failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    summary = load_json(replay_summary_path)
    manifest = load_json(Path(str(summary["run_manifest_path"])))
    replay_pack = load_json(Path(str(summary["replay_pack_path"])))
    replay_index = load_json(Path(str(summary["replay_index_path"])))

    artifact_categories = {artifact["category"] for artifact in manifest["artifacts"]}
    assert manifest["run_kind"] == "eval"
    assert manifest["replay_mode"] == "seed_pack_eval"
    assert manifest["summary_digest"] == summary["summary_digest"]
    assert manifest["seed_pack"] == "bootstrap_smoke"
    assert artifact_categories == {
        "eval_summary",
        "replay_index",
        "replay_pack",
        "run_manifest",
    }
    assert replay_pack["checkpoint_path"] == checkpoint_path
    assert replay_pack["seed_pack"] == manifest["seed_pack"]
    assert replay_index["summary_digest"] == manifest["summary_digest"]
    assert replay_index["checkpoint_format_id"] == manifest["checkpoint_format_id"]
