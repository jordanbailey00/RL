from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from fight_caves_rl.envs.puffer_encoding import (
    PUFFER_POLICY_ACTION_SCHEMA,
    PUFFER_POLICY_OBSERVATION_SCHEMA,
)
from fight_caves_rl.envs.schema import (
    FIGHT_CAVE_EPISODE_START_CONTRACT,
    HEADLESS_ACTION_SCHEMA,
    HEADLESS_OBSERVATION_SCHEMA,
)
from fight_caves_rl.manifests.versions import resolve_pufferlib_runtime_version


@dataclass(frozen=True)
class CheckpointMetadata:
    checkpoint_format_id: str
    checkpoint_format_version: int
    train_config_id: str
    policy_id: str
    reward_config_id: str
    curriculum_config_id: str
    sim_observation_schema_id: str
    sim_observation_schema_version: int
    sim_action_schema_id: str
    sim_action_schema_version: int
    episode_start_contract_id: str
    episode_start_contract_version: int
    policy_observation_schema_id: str
    policy_observation_schema_version: int
    policy_action_schema_id: str
    policy_action_schema_version: int
    pufferlib_distribution: str
    pufferlib_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_checkpoint_metadata(
    *,
    train_config_id: str,
    policy_id: str,
    reward_config_id: str,
    curriculum_config_id: str,
) -> CheckpointMetadata:
    runtime = resolve_pufferlib_runtime_version()
    return CheckpointMetadata(
        checkpoint_format_id="rl_checkpoint_v0",
        checkpoint_format_version=0,
        train_config_id=train_config_id,
        policy_id=policy_id,
        reward_config_id=reward_config_id,
        curriculum_config_id=curriculum_config_id,
        sim_observation_schema_id=HEADLESS_OBSERVATION_SCHEMA.contract_id,
        sim_observation_schema_version=HEADLESS_OBSERVATION_SCHEMA.version,
        sim_action_schema_id=HEADLESS_ACTION_SCHEMA.contract_id,
        sim_action_schema_version=HEADLESS_ACTION_SCHEMA.version,
        episode_start_contract_id=FIGHT_CAVE_EPISODE_START_CONTRACT.identity.contract_id,
        episode_start_contract_version=FIGHT_CAVE_EPISODE_START_CONTRACT.identity.version,
        policy_observation_schema_id=PUFFER_POLICY_OBSERVATION_SCHEMA.contract_id,
        policy_observation_schema_version=PUFFER_POLICY_OBSERVATION_SCHEMA.version,
        policy_action_schema_id=PUFFER_POLICY_ACTION_SCHEMA.contract_id,
        policy_action_schema_version=PUFFER_POLICY_ACTION_SCHEMA.version,
        pufferlib_distribution=runtime.distribution_name,
        pufferlib_version=runtime.distribution_version,
    )


def metadata_path_for_checkpoint(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.metadata.json")


def write_checkpoint_metadata(checkpoint_path: Path, metadata: CheckpointMetadata) -> Path:
    output = metadata_path_for_checkpoint(checkpoint_path)
    output.write_text(
        json.dumps(metadata.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def load_checkpoint_metadata(checkpoint_path: Path) -> CheckpointMetadata:
    payload = json.loads(metadata_path_for_checkpoint(checkpoint_path).read_text(encoding="utf-8"))
    return CheckpointMetadata(**payload)


def load_policy_checkpoint(checkpoint_path: Path, policy: torch.nn.Module) -> CheckpointMetadata:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    return load_checkpoint_metadata(checkpoint_path)

