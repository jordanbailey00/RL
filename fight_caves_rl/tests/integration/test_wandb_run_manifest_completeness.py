from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def _wandb_env(tmp_path: Path) -> dict[str, str]:
    return {
        "WANDB_PROJECT": "fight-caves-rl-tests",
        "WANDB_MODE": "offline",
        "WANDB_GROUP": "pr6",
        "WANDB_TAGS": "integration,pr6",
        "WANDB_DIR": str(tmp_path / "wandb"),
        "WANDB_DATA_DIR": str(tmp_path / "wandb-data"),
        "WANDB_CACHE_DIR": str(tmp_path / "wandb-cache"),
    }


def test_train_and_eval_manifests_record_required_contracts(tmp_path: Path):
    require_live_runtime()

    env = _wandb_env(tmp_path)
    train_summary_path = tmp_path / "train_summary.json"
    train_data_dir = tmp_path / "train_artifacts"
    train_result = run_script(
        "train.py",
        "--config",
        "configs/train/smoke_ppo_v0.yaml",
        "--total-timesteps",
        "4",
        "--data-dir",
        str(train_data_dir),
        "--output",
        str(train_summary_path),
        env=env,
    )
    if train_result.returncode != 0:
        raise AssertionError(
            f"Train PR6 manifest test failed.\nstdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}"
        )

    train_summary = load_json(train_summary_path)
    train_manifest = load_json(Path(str(train_summary["run_manifest_path"])))
    assert train_manifest["run_kind"] == "train"
    assert train_manifest["config_id"] == "smoke_ppo_v0"
    assert train_manifest["benchmark_profile_id"] == "official_profile_v0"
    assert train_manifest["bridge_protocol_id"] == "fight_caves_bridge_v0"
    assert train_manifest["episode_start_contract_id"] == "fight_cave_episode_start_v1"
    assert train_manifest["observation_schema_id"] == "headless_observation_v1"
    assert train_manifest["action_schema_id"] == "headless_action_v1"
    assert train_manifest["policy_observation_schema_id"] == "puffer_policy_observation_v0"
    assert train_manifest["policy_action_schema_id"] == "puffer_policy_action_v0"
    assert train_manifest["pufferlib_distribution"] == "pufferlib-core"
    assert train_manifest["checkpoint_format_id"] == "rl_checkpoint_v0"
    assert train_manifest["rl_commit_sha"]
    assert train_manifest["sim_commit_sha"]
    assert train_manifest["rsps_commit_sha"]
    assert train_manifest["artifacts"]

    eval_summary_path = tmp_path / "eval_summary.json"
    eval_result = run_script(
        "eval.py",
        "--checkpoint",
        str(train_summary["checkpoint_path"]),
        "--config",
        "configs/eval/replay_eval_v0.yaml",
        "--max-steps",
        "16",
        "--output",
        str(eval_summary_path),
        env=env,
    )
    if eval_result.returncode != 0:
        raise AssertionError(
            f"Eval PR6 manifest test failed.\nstdout:\n{eval_result.stdout}\nstderr:\n{eval_result.stderr}"
        )

    eval_summary = load_json(eval_summary_path)
    eval_manifest = load_json(Path(str(eval_summary["run_manifest_path"])))
    assert eval_manifest["run_kind"] == "eval"
    assert eval_manifest["seed_pack"] == "bootstrap_smoke"
    assert eval_manifest["seed_pack_version"] == 0
    assert eval_manifest["summary_digest"] == eval_summary["summary_digest"]
    assert eval_manifest["checkpoint_path"] == train_summary["checkpoint_path"]
    assert eval_manifest["artifacts"]
