from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from platform import machine, platform, processor, python_implementation, python_version
import json
import os
import subprocess
from typing import TYPE_CHECKING, Any, Sequence

from fight_caves_rl.bridge.launcher import build_bridge_handshake, discover_headless_runtime_paths
from fight_caves_rl.envs.schema import (
    OFFICIAL_BENCHMARK_PROFILE,
    PUFFER_POLICY_ACTION_SCHEMA,
    PUFFER_POLICY_OBSERVATION_SCHEMA,
)
from fight_caves_rl.logging.artifact_naming import ArtifactRecord
from fight_caves_rl.manifests.versions import resolve_pufferlib_runtime_version
from fight_caves_rl.utils.config import BootstrapConfig

if TYPE_CHECKING:
    from fight_caves_rl.policies.checkpointing import CheckpointMetadata


@dataclass(frozen=True)
class BootstrapRunManifest:
    created_at: str
    rl_repo: str
    sim_repo: str
    rsps_repo: str
    python_version: str
    python_baseline: str
    pufferlib_distribution: str
    pufferlib_version: str
    wandb_project: str
    wandb_mode: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_bootstrap_manifest(config: BootstrapConfig) -> BootstrapRunManifest:
    return BootstrapRunManifest(
        created_at=datetime.now(UTC).isoformat(),
        rl_repo=str(Path(config.rl_repo)),
        sim_repo=str(Path(config.sim_repo)),
        rsps_repo=str(Path(config.rsps_repo)),
        python_version=python_version(),
        python_baseline=config.python_baseline,
        pufferlib_distribution=config.pufferlib_distribution,
        pufferlib_version=config.pufferlib_version,
        wandb_project=config.wandb_project,
        wandb_mode=config.wandb_mode,
    )


@dataclass(frozen=True)
class HardwareProfile:
    platform: str
    machine: str
    processor: str
    cpu_count: int
    python_implementation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunManifest:
    created_at: str
    run_kind: str
    run_id: str
    config_id: str
    rl_repo: str
    rl_commit_sha: str
    sim_repo: str
    sim_commit_sha: str
    rsps_repo: str
    rsps_commit_sha: str
    run_output_dir: str
    python_version: str
    python_baseline: str
    pufferlib_distribution: str
    pufferlib_version: str
    pufferlib_import_name: str
    pufferlib_import_version: str | None
    wandb_project: str
    wandb_entity: str | None
    wandb_group: str | None
    wandb_mode: str
    wandb_resume: str
    wandb_tags: tuple[str, ...]
    wandb_dir: str
    wandb_data_dir: str
    wandb_cache_dir: str
    benchmark_profile_id: str
    benchmark_profile_version: int
    benchmark_mode: str
    bridge_mode: str
    replay_mode: str
    logging_mode: str
    dashboard_mode: str
    env_count: int
    sim_artifact_task: str
    sim_artifact_path: str
    bridge_protocol_id: str
    bridge_protocol_version: int
    observation_schema_id: str
    observation_schema_version: int
    action_schema_id: str
    action_schema_version: int
    episode_start_contract_id: str
    episode_start_contract_version: int
    policy_observation_schema_id: str
    policy_observation_schema_version: int
    policy_action_schema_id: str
    policy_action_schema_version: int
    reward_config_id: str
    curriculum_config_id: str
    policy_id: str
    checkpoint_format_id: str | None
    checkpoint_format_version: int | None
    checkpoint_path: str | None
    checkpoint_metadata_path: str | None
    seed_pack: str | None
    seed_pack_version: int | None
    summary_digest: str | None
    hardware_profile: HardwareProfile
    artifacts: tuple[ArtifactRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hardware_profile"] = self.hardware_profile.to_dict()
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return payload


def write_run_manifest(path: str | Path, manifest: RunManifest) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def build_train_run_manifest(
    *,
    bootstrap_config: BootstrapConfig,
    config_id: str,
    run_id: str,
    run_output_dir: str | Path,
    reward_config_id: str,
    curriculum_config_id: str,
    policy_id: str,
    env_count: int,
    dashboard_enabled: bool,
    wandb_tags: Sequence[str],
    checkpoint_metadata: CheckpointMetadata,
    checkpoint_path: str | Path,
    checkpoint_metadata_path: str | Path,
    artifacts: Sequence[ArtifactRecord],
) -> RunManifest:
    return _build_run_manifest(
        bootstrap_config=bootstrap_config,
        run_kind="train",
        config_id=config_id,
        run_id=run_id,
        run_output_dir=run_output_dir,
        reward_config_id=reward_config_id,
        curriculum_config_id=curriculum_config_id,
        policy_id=policy_id,
        replay_mode="disabled",
        logging_mode=f"wandb_{bootstrap_config.wandb_mode}",
        dashboard_mode="enabled" if dashboard_enabled else "disabled",
        env_count=env_count,
        wandb_tags=wandb_tags,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        seed_pack=None,
        seed_pack_version=None,
        summary_digest=None,
        artifacts=artifacts,
    )


def build_eval_run_manifest(
    *,
    bootstrap_config: BootstrapConfig,
    config_id: str,
    run_id: str,
    run_output_dir: str | Path,
    reward_config_id: str,
    curriculum_config_id: str,
    policy_id: str,
    env_count: int,
    wandb_tags: Sequence[str],
    checkpoint_metadata: CheckpointMetadata,
    checkpoint_path: str | Path,
    checkpoint_metadata_path: str | Path,
    seed_pack: str,
    seed_pack_version: int,
    summary_digest: str,
    artifacts: Sequence[ArtifactRecord],
) -> RunManifest:
    return _build_run_manifest(
        bootstrap_config=bootstrap_config,
        run_kind="eval",
        config_id=config_id,
        run_id=run_id,
        run_output_dir=run_output_dir,
        reward_config_id=reward_config_id,
        curriculum_config_id=curriculum_config_id,
        policy_id=policy_id,
        replay_mode="seed_pack_eval",
        logging_mode=f"wandb_{bootstrap_config.wandb_mode}",
        dashboard_mode="disabled",
        env_count=env_count,
        wandb_tags=wandb_tags,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=checkpoint_metadata_path,
        seed_pack=seed_pack,
        seed_pack_version=seed_pack_version,
        summary_digest=summary_digest,
        artifacts=artifacts,
    )


def _build_run_manifest(
    *,
    bootstrap_config: BootstrapConfig,
    run_kind: str,
    config_id: str,
    run_id: str,
    run_output_dir: str | Path,
    reward_config_id: str,
    curriculum_config_id: str,
    policy_id: str,
    replay_mode: str,
    logging_mode: str,
    dashboard_mode: str,
    env_count: int,
    wandb_tags: Sequence[str],
    checkpoint_metadata: CheckpointMetadata,
    checkpoint_path: str | Path,
    checkpoint_metadata_path: str | Path,
    seed_pack: str | None,
    seed_pack_version: int | None,
    summary_digest: str | None,
    artifacts: Sequence[ArtifactRecord],
) -> RunManifest:
    runtime = resolve_pufferlib_runtime_version()
    runtime_paths = discover_headless_runtime_paths(bootstrap_config.sim_repo)
    handshake = build_bridge_handshake(runtime_paths)
    return RunManifest(
        created_at=datetime.now(UTC).isoformat(),
        run_kind=run_kind,
        run_id=run_id,
        config_id=config_id,
        rl_repo=str(bootstrap_config.rl_repo.resolve()),
        rl_commit_sha=_resolve_commit_sha(bootstrap_config.rl_repo),
        sim_repo=str(bootstrap_config.sim_repo.resolve()),
        sim_commit_sha=_resolve_commit_sha(bootstrap_config.sim_repo),
        rsps_repo=str(bootstrap_config.rsps_repo.resolve()),
        rsps_commit_sha=_resolve_commit_sha(bootstrap_config.rsps_repo),
        run_output_dir=str(Path(run_output_dir)),
        python_version=python_version(),
        python_baseline=bootstrap_config.python_baseline,
        pufferlib_distribution=runtime.distribution_name,
        pufferlib_version=runtime.distribution_version,
        pufferlib_import_name=runtime.import_name,
        pufferlib_import_version=runtime.import_version,
        wandb_project=bootstrap_config.wandb_project,
        wandb_entity=bootstrap_config.wandb_entity,
        wandb_group=bootstrap_config.wandb_group,
        wandb_mode=bootstrap_config.wandb_mode,
        wandb_resume=bootstrap_config.wandb_resume,
        wandb_tags=tuple(wandb_tags),
        wandb_dir=str(bootstrap_config.wandb_dir),
        wandb_data_dir=str(bootstrap_config.wandb_data_dir),
        wandb_cache_dir=str(bootstrap_config.wandb_cache_dir),
        benchmark_profile_id=OFFICIAL_BENCHMARK_PROFILE.identity.contract_id,
        benchmark_profile_version=OFFICIAL_BENCHMARK_PROFILE.identity.version,
        benchmark_mode=OFFICIAL_BENCHMARK_PROFILE.benchmark_mode,
        bridge_mode=OFFICIAL_BENCHMARK_PROFILE.bridge_mode,
        replay_mode=replay_mode,
        logging_mode=logging_mode,
        dashboard_mode=dashboard_mode,
        env_count=int(env_count),
        sim_artifact_task=str(handshake.values["sim_artifact_task"]),
        sim_artifact_path=str(handshake.values["sim_artifact_path"]),
        bridge_protocol_id=str(handshake.values["bridge_protocol_id"]),
        bridge_protocol_version=int(handshake.values["bridge_protocol_version"]),
        observation_schema_id=str(handshake.values["observation_schema_id"]),
        observation_schema_version=int(handshake.values["observation_schema_version"]),
        action_schema_id=str(handshake.values["action_schema_id"]),
        action_schema_version=int(handshake.values["action_schema_version"]),
        episode_start_contract_id=str(handshake.values["episode_start_contract_id"]),
        episode_start_contract_version=int(handshake.values["episode_start_contract_version"]),
        policy_observation_schema_id=PUFFER_POLICY_OBSERVATION_SCHEMA.contract_id,
        policy_observation_schema_version=PUFFER_POLICY_OBSERVATION_SCHEMA.version,
        policy_action_schema_id=PUFFER_POLICY_ACTION_SCHEMA.contract_id,
        policy_action_schema_version=PUFFER_POLICY_ACTION_SCHEMA.version,
        reward_config_id=reward_config_id,
        curriculum_config_id=curriculum_config_id,
        policy_id=policy_id,
        checkpoint_format_id=checkpoint_metadata.checkpoint_format_id,
        checkpoint_format_version=checkpoint_metadata.checkpoint_format_version,
        checkpoint_path=str(checkpoint_path),
        checkpoint_metadata_path=str(checkpoint_metadata_path),
        seed_pack=seed_pack,
        seed_pack_version=seed_pack_version,
        summary_digest=summary_digest,
        hardware_profile=_build_hardware_profile(),
        artifacts=tuple(artifacts),
    )


def _build_hardware_profile() -> HardwareProfile:
    return HardwareProfile(
        platform=platform(),
        machine=machine(),
        processor=processor(),
        cpu_count=int(os.cpu_count() or 1),
        python_implementation=python_implementation(),
    )


def _resolve_commit_sha(repo_path: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to resolve git commit SHA for {repo_path}: {result.stderr.strip()}"
        )
    return result.stdout.strip()
