from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from os import cpu_count, environ
from pathlib import Path
from platform import (
    machine,
    platform,
    processor,
    python_implementation,
    python_version,
    release,
    system,
    version,
)
import subprocess
from typing import Any

from fight_caves_rl.bridge.launcher import build_bridge_handshake, discover_headless_runtime_paths
from fight_caves_rl.envs.schema import OFFICIAL_BENCHMARK_PROFILE
from fight_caves_rl.manifests.versions import resolve_pufferlib_runtime_version
from fight_caves_rl.utils.java_runtime import resolve_java_executable
from fight_caves_rl.utils.config import load_bootstrap_config


@dataclass(frozen=True)
class BenchmarkHardwareProfile:
    platform: str
    system: str
    release: str
    version: str
    machine: str
    processor: str
    cpu_count: int
    python_implementation: str
    python_version: str
    host_class: str
    is_wsl: bool
    performance_source_of_truth: bool
    java_runtime_version: str | None
    java_vm_name: str | None

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
    system_name = system()
    release_name = release()
    version_name = version()
    host_class, is_wsl = detect_host_class(
        system_name=system_name,
        release_name=release_name,
        version_name=version_name,
        platform_string=platform(),
    )
    java_runtime_version, java_vm_name = _resolve_java_runtime_profile()
    performance_source_of_truth = _resolve_performance_source_of_truth(host_class)
    return BenchmarkHardwareProfile(
        platform=platform(),
        system=system_name,
        release=release_name,
        version=version_name,
        machine=machine(),
        processor=processor(),
        cpu_count=int(cpu_count() or 1),
        python_implementation=python_implementation(),
        python_version=python_version(),
        host_class=host_class,
        is_wsl=is_wsl,
        performance_source_of_truth=performance_source_of_truth,
        java_runtime_version=java_runtime_version,
        java_vm_name=java_vm_name,
    )


def detect_host_class(
    *,
    system_name: str,
    release_name: str,
    version_name: str,
    platform_string: str,
) -> tuple[str, bool]:
    joined = " ".join((system_name, release_name, version_name, platform_string)).lower()
    is_wsl = "microsoft" in joined or "wsl" in joined
    if system_name.lower() == "linux" and is_wsl:
        return "wsl2", True
    if system_name.lower() == "linux":
        return "linux_native", False
    if system_name.lower() == "windows":
        return "windows", False
    if system_name.lower() == "darwin":
        return "macos", False
    return "other", False


def _resolve_performance_source_of_truth(host_class: str) -> bool:
    override = environ.get("FC_RL_PERF_SOURCE_OF_TRUTH")
    if override is not None:
        return override.strip().lower() in {"1", "true", "yes", "on"}
    return host_class == "linux_native"


@lru_cache(maxsize=1)
def _resolve_java_runtime_profile() -> tuple[str | None, str | None]:
    java_executable = resolve_java_executable()
    if java_executable is None:
        return None, None
    try:
        result = subprocess.run(
            [str(java_executable), "-XshowSettings:properties", "-version"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None, None
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        return None, None
    java_runtime_version: str | None = None
    java_vm_name: str | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("java.runtime.version ="):
            java_runtime_version = stripped.split("=", 1)[1].strip()
        elif stripped.startswith("java.vm.name ="):
            java_vm_name = stripped.split("=", 1)[1].strip()
    return java_runtime_version, java_vm_name


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
