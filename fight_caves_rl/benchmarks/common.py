from __future__ import annotations

from dataclasses import asdict, dataclass
from os import cpu_count
from pathlib import Path
from platform import machine, platform, processor, python_implementation, python_version
import subprocess
from typing import Any

from fight_caves_rl.bridge.launcher import build_bridge_handshake, discover_headless_runtime_paths
from fight_caves_rl.envs.schema import OFFICIAL_BENCHMARK_PROFILE
from fight_caves_rl.manifests.versions import resolve_pufferlib_runtime_version
from fight_caves_rl.utils.config import load_bootstrap_config


@dataclass(frozen=True)
class BenchmarkHardwareProfile:
    platform: str
    machine: str
    processor: str
    cpu_count: int
    python_implementation: str
    python_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkContext:
    benchmark_profile_id: str
    benchmark_profile_version: int
    benchmark_mode: str
    bridge_mode: str
    reward_config_id: str
    curriculum_config_id: str
    replay_mode: str
    logging_mode: str
    dashboard_mode: str
    env_count: int
    rl_repo: str
    rl_commit_sha: str
    sim_repo: str
    sim_commit_sha: str
    rsps_repo: str
    rsps_commit_sha: str
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
    pufferlib_distribution: str
    pufferlib_version: str
    hardware_profile: BenchmarkHardwareProfile

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hardware_profile"] = self.hardware_profile.to_dict()
        return payload


def build_benchmark_context(
    *,
    env_count: int,
    logging_mode: str,
    replay_mode: str,
    dashboard_mode: str,
    reward_config_id: str,
    curriculum_config_id: str,
) -> BenchmarkContext:
    bootstrap_config = load_bootstrap_config()
    runtime_paths = discover_headless_runtime_paths(bootstrap_config.sim_repo)
    handshake = build_bridge_handshake(runtime_paths)
    runtime = resolve_pufferlib_runtime_version()
    return BenchmarkContext(
        benchmark_profile_id=OFFICIAL_BENCHMARK_PROFILE.identity.contract_id,
        benchmark_profile_version=OFFICIAL_BENCHMARK_PROFILE.identity.version,
        benchmark_mode=OFFICIAL_BENCHMARK_PROFILE.benchmark_mode,
        bridge_mode=OFFICIAL_BENCHMARK_PROFILE.bridge_mode,
        reward_config_id=str(reward_config_id),
        curriculum_config_id=str(curriculum_config_id),
        replay_mode=str(replay_mode),
        logging_mode=str(logging_mode),
        dashboard_mode=str(dashboard_mode),
        env_count=int(env_count),
        rl_repo=str(bootstrap_config.rl_repo.resolve()),
        rl_commit_sha=_resolve_commit_sha(bootstrap_config.rl_repo),
        sim_repo=str(bootstrap_config.sim_repo.resolve()),
        sim_commit_sha=_resolve_commit_sha(bootstrap_config.sim_repo),
        rsps_repo=str(bootstrap_config.rsps_repo.resolve()),
        rsps_commit_sha=_resolve_commit_sha(bootstrap_config.rsps_repo),
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
        pufferlib_distribution=runtime.distribution_name,
        pufferlib_version=runtime.distribution_version,
        hardware_profile=_build_hardware_profile(),
    )


def _build_hardware_profile() -> BenchmarkHardwareProfile:
    return BenchmarkHardwareProfile(
        platform=platform(),
        machine=machine(),
        processor=processor(),
        cpu_count=int(cpu_count() or 1),
        python_implementation=python_implementation(),
        python_version=python_version(),
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
