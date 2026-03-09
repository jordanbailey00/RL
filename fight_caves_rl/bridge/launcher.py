from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import zipfile

from fight_caves_rl.bridge.contracts import (
    BridgeHandshake,
    HeadlessBootstrapConfig,
    HeadlessRuntimePaths,
)
from fight_caves_rl.bridge.errors import (
    BridgeContractError,
    SimArtifactNotFoundError,
    SimPrerequisiteError,
)
from fight_caves_rl.envs.schema import (
    FIGHT_CAVE_EPISODE_START_CONTRACT,
    FIGHT_CAVES_BRIDGE_CONTRACT,
    HEADLESS_ACTION_SCHEMA,
    HEADLESS_OBSERVATION_SCHEMA,
    HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA,
    OFFICIAL_BENCHMARK_PROFILE,
)
from fight_caves_rl.manifests.versions import resolve_pufferlib_runtime_version
from fight_caves_rl.utils.config import load_bootstrap_config
from fight_caves_rl.utils.paths import repo_root

HEADLESS_EXTRACT_ROOT = repo_root() / "artifacts" / "headless_dist"


def resolve_sim_repo(sim_repo: Path | None = None) -> Path:
    if sim_repo is not None:
        return Path(sim_repo).expanduser().resolve()
    return load_bootstrap_config().sim_repo.resolve()


def discover_headless_runtime_paths(
    sim_repo: Path | None = None,
    extract_root: Path | None = None,
) -> HeadlessRuntimePaths:
    resolved_repo = resolve_sim_repo(sim_repo)
    distribution_zip = _find_distribution_zip(resolved_repo)
    extracted_distribution_dir = _extract_distribution(
        distribution_zip=distribution_zip,
        extract_root=(extract_root or HEADLESS_EXTRACT_ROOT),
    )
    headless_jar = extracted_distribution_dir / FIGHT_CAVES_BRIDGE_CONTRACT.sim_headless_jar_name
    if not headless_jar.is_file():
        raise SimArtifactNotFoundError(
            "Packaged headless jar is missing from the extracted distribution: "
            f"{headless_jar}"
        )

    return HeadlessRuntimePaths(
        sim_repo=resolved_repo,
        distribution_zip=distribution_zip,
        extracted_distribution_dir=extracted_distribution_dir,
        headless_jar=headless_jar,
        launch_cwd=resolved_repo / "game",
        cache_root=resolved_repo / "data" / "cache",
    )


def assert_sim_runtime_ready(paths: HeadlessRuntimePaths) -> None:
    missing = [
        relative
        for relative in FIGHT_CAVES_BRIDGE_CONTRACT.required_sim_workspace_paths
        if not (paths.sim_repo / relative).exists()
    ]
    if missing:
        missing_lines = "\n".join(f"- {item}" for item in missing)
        raise SimPrerequisiteError(
            "fight-caves-RL is not runtime-ready for PR3.\n"
            "Missing checked-out workspace prerequisites:\n"
            f"{missing_lines}\n"
            "The current headless bootstrap still requires the checked-out "
            "fight-caves-RL repo root plus a populated data/cache directory."
        )
    if not paths.launch_cwd.is_dir():
        raise SimPrerequisiteError(
            f"Expected sim launch cwd is missing: {paths.launch_cwd}"
        )


def build_headless_settings_overrides(
    paths: HeadlessRuntimePaths,
    bootstrap: HeadlessBootstrapConfig | None = None,
) -> dict[str, str]:
    overrides = {
        "storage.data": str((paths.sim_repo / "data").resolve()) + "/",
        "storage.data.modified": str((paths.sim_repo / "temp" / "data" / "headless-test-cache" / "modified.dat").resolve()),
        "storage.cache.path": str(paths.cache_root.resolve()) + "/",
        "storage.wildcards": str((paths.sim_repo / "temp" / "data" / "headless-test-cache" / "wildcards.txt").resolve()),
        "storage.caching.path": str((paths.sim_repo / "temp" / "data" / "headless-test-cache").resolve()) + "/",
        "storage.caching.active": "false",
        "storage.players.path": str((paths.sim_repo / "temp" / "data" / "test-saves").resolve()) + "/",
        "storage.players.logs": str((paths.sim_repo / "temp" / "data" / "test-logs").resolve()) + "/",
        "storage.players.errors": str((paths.sim_repo / "temp" / "data" / "test-errors").resolve()) + "/",
        "storage.autoSave.minutes": "0",
        "events.shootingStars.enabled": "false",
        "events.penguinHideAndSeek.enabled": "false",
        "bots.count": "0",
        "world.npcs.randomWalk": "false",
        "storage.disabled": "true",
        "headless.data.allowlist.path": str((paths.sim_repo / "config" / "headless_data_allowlist.toml").resolve()),
        "headless.scripts.allowlist.path": str((paths.sim_repo / "config" / "headless_scripts.txt").resolve()),
        "headless.manifest.path": str((paths.sim_repo / "config" / "headless_manifest.toml").resolve()),
    }
    if bootstrap is not None:
        overrides.update(dict(bootstrap.settings_overrides))
    return overrides


def build_bridge_handshake(paths: HeadlessRuntimePaths) -> BridgeHandshake:
    runtime_version = resolve_pufferlib_runtime_version()
    values = {
        "observation_schema_id": HEADLESS_OBSERVATION_SCHEMA.contract_id,
        "observation_schema_version": HEADLESS_OBSERVATION_SCHEMA.version,
        "observation_path_mode": "flat",
        "flat_observation_schema_id": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.identity.contract_id,
        "flat_observation_schema_version": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.identity.version,
        "flat_observation_dtype": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.dtype,
        "flat_observation_feature_count": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.feature_count,
        "flat_observation_max_visible_npcs": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.max_visible_npcs,
        "action_schema_id": HEADLESS_ACTION_SCHEMA.contract_id,
        "action_schema_version": HEADLESS_ACTION_SCHEMA.version,
        "episode_start_contract_id": FIGHT_CAVE_EPISODE_START_CONTRACT.identity.contract_id,
        "episode_start_contract_version": FIGHT_CAVE_EPISODE_START_CONTRACT.identity.version,
        "bridge_protocol_id": FIGHT_CAVES_BRIDGE_CONTRACT.identity.contract_id,
        "bridge_protocol_version": FIGHT_CAVES_BRIDGE_CONTRACT.identity.version,
        "benchmark_profile_id": OFFICIAL_BENCHMARK_PROFILE.identity.contract_id,
        "benchmark_profile_version": OFFICIAL_BENCHMARK_PROFILE.identity.version,
        "sim_artifact_task": FIGHT_CAVES_BRIDGE_CONTRACT.sim_artifact_task,
        "sim_artifact_path": str(paths.distribution_zip),
        "pufferlib_distribution": runtime_version.distribution_name,
        "pufferlib_version": runtime_version.distribution_version,
    }
    missing = [
        field
        for field in FIGHT_CAVES_BRIDGE_CONTRACT.required_handshake_fields
        if field not in values
    ]
    if missing:
        raise BridgeContractError(
            "Bridge handshake is missing required fields: "
            + ", ".join(sorted(missing))
        )
    return BridgeHandshake(values=values)


def _find_distribution_zip(sim_repo: Path) -> Path:
    matches = sorted(
        sim_repo.glob(FIGHT_CAVES_BRIDGE_CONTRACT.sim_distribution_glob),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise SimArtifactNotFoundError(
            "Could not find a packaged headless distribution matching "
            f"{FIGHT_CAVES_BRIDGE_CONTRACT.sim_distribution_glob!r} under "
            f"{sim_repo}.\nBuild it with "
            f"`./gradlew {FIGHT_CAVES_BRIDGE_CONTRACT.sim_artifact_task} --no-daemon`."
        )
    return matches[0].resolve()


def _extract_distribution(distribution_zip: Path, extract_root: Path) -> Path:
    target_root = Path(extract_root).resolve() / distribution_zip.stem
    marker = target_root / ".extract-complete"
    if marker.is_file() and marker.stat().st_mtime >= distribution_zip.stat().st_mtime:
        extracted = _resolve_distribution_root(target_root)
        if extracted is not None:
            return extracted

    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(distribution_zip) as archive:
        archive.extractall(target_root)

    extracted = _resolve_distribution_root(target_root)
    if extracted is None:
        raise SimArtifactNotFoundError(
            f"Unable to determine the extracted distribution root for {distribution_zip}."
        )

    marker.write_text(distribution_zip.name, encoding="utf-8")
    return extracted


def _resolve_distribution_root(target_root: Path) -> Path | None:
    candidates = [candidate for candidate in target_root.iterdir() if candidate.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    if (target_root / FIGHT_CAVES_BRIDGE_CONTRACT.sim_headless_jar_name).is_file():
        return target_root
    return None
