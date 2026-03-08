from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class HeadlessRuntimePaths:
    sim_repo: Path
    distribution_zip: Path
    extracted_distribution_dir: Path
    headless_jar: Path
    launch_cwd: Path
    cache_root: Path


@dataclass(frozen=True)
class HeadlessBootstrapConfig:
    load_content_scripts: bool = True
    start_world: bool = True
    install_shutdown_hook: bool = False
    settings_overrides: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HeadlessPlayerConfig:
    account_name: str
    tile_x: int = 2438
    tile_y: int = 5168
    tile_level: int = 0


@dataclass(frozen=True)
class HeadlessEpisodeConfig:
    seed: int
    start_wave: int = 1
    ammo: int = 1000
    prayer_potions: int = 8
    sharks: int = 20


@dataclass(frozen=True)
class BridgeHandshake:
    values: Mapping[str, Any]


@dataclass(frozen=True)
class StepSnapshot:
    observation: dict[str, Any]
    action_result: dict[str, Any]
    visible_targets: list[dict[str, Any]]

