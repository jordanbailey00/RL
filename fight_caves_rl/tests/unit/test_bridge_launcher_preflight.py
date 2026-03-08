from pathlib import Path
import zipfile

import pytest

from fight_caves_rl.bridge.errors import SimPrerequisiteError
from fight_caves_rl.bridge.launcher import (
    assert_sim_runtime_ready,
    build_bridge_handshake,
    build_headless_settings_overrides,
    discover_headless_runtime_paths,
)


def test_launcher_discovers_globbed_distribution_and_extracts_jar(tmp_path: Path):
    sim_repo = _make_sim_repo(tmp_path, with_cache=True)
    extract_root = tmp_path / "extract"

    paths = discover_headless_runtime_paths(sim_repo=sim_repo, extract_root=extract_root)

    assert paths.distribution_zip.name == "fight-caves-headless-dev.zip"
    assert paths.headless_jar.is_file()
    assert paths.launch_cwd == sim_repo / "game"
    assert paths.cache_root == sim_repo / "data" / "cache"

    overrides = build_headless_settings_overrides(paths)
    assert overrides["storage.cache.path"].endswith("/data/cache/")
    assert overrides["headless.data.allowlist.path"].endswith("/config/headless_data_allowlist.toml")

    handshake = build_bridge_handshake(paths)
    assert handshake.values["sim_artifact_path"].endswith("fight-caves-headless-dev.zip")


def test_launcher_preflight_fails_fast_when_cache_is_missing(tmp_path: Path):
    sim_repo = _make_sim_repo(tmp_path, with_cache=False)
    paths = discover_headless_runtime_paths(sim_repo=sim_repo, extract_root=tmp_path / "extract")

    with pytest.raises(SimPrerequisiteError):
        assert_sim_runtime_ready(paths)


def _make_sim_repo(root: Path, with_cache: bool) -> Path:
    sim_repo = root / "fight-caves-RL"
    (sim_repo / "game" / "build" / "distributions").mkdir(parents=True)
    (sim_repo / "game").mkdir(exist_ok=True)
    (sim_repo / "config").mkdir()
    (sim_repo / "data").mkdir()
    (sim_repo / "temp" / "data" / "headless-test-cache").mkdir(parents=True)
    (sim_repo / "temp" / "data" / "test-saves").mkdir(parents=True)
    (sim_repo / "temp" / "data" / "test-logs").mkdir(parents=True)
    (sim_repo / "temp" / "data" / "test-errors").mkdir(parents=True)
    (sim_repo / "FCspec.md").write_text("# FCspec", encoding="utf-8")
    for name in (
        "headless_data_allowlist.toml",
        "headless_manifest.toml",
        "headless_scripts.txt",
    ):
        (sim_repo / "config" / name).write_text("", encoding="utf-8")
    if with_cache:
        (sim_repo / "data" / "cache").mkdir(parents=True)
        (sim_repo / "data" / "cache" / "main_file_cache.dat2").write_text(
            "cache",
            encoding="utf-8",
        )

    archive = sim_repo / "game" / "build" / "distributions" / "fight-caves-headless-dev.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("fight-caves-headless-dev/fight-caves-headless.jar", "jar")
        handle.writestr("fight-caves-headless-dev/game.properties", "server.name=test")
    return sim_repo
