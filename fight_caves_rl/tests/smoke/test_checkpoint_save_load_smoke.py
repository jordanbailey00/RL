from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.envs.puffer_encoding import build_policy_action_space, build_policy_observation_space
from fight_caves_rl.policies.checkpointing import load_policy_checkpoint
from fight_caves_rl.policies.mlp import MultiDiscreteMLPPolicy
from fight_caves_rl.tests.smoke._helpers import (
    load_json,
    offline_wandb_env,
    require_live_runtime,
    run_script,
)

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_checkpoint_save_load_smoke(tmp_path: Path):
    require_live_runtime()

    summary_path = tmp_path / "train_summary.json"
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
        env=offline_wandb_env(tmp_path, tags="smoke,checkpoint"),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Train smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    summary = load_json(summary_path)
    checkpoint_path = Path(str(summary["checkpoint_path"]))
    policy = MultiDiscreteMLPPolicy.from_spaces(
        build_policy_observation_space(),
        build_policy_action_space(),
        hidden_size=128,
    )
    metadata = load_policy_checkpoint(checkpoint_path, policy)

    assert metadata.checkpoint_format_id == "rl_checkpoint_v0"
    assert metadata.train_config_id == "smoke_ppo_v0"
    assert metadata.policy_observation_schema_id == "puffer_policy_observation_v0"
    assert metadata.policy_action_schema_id == "puffer_policy_action_v0"
