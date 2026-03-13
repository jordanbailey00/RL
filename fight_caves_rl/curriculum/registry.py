from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from fight_caves_rl.utils.paths import repo_root


@dataclass(frozen=True)
class CurriculumStage:
    until_episode: int | None
    start_wave: int | None = None
    start_waves: tuple[int, ...] = ()


@dataclass(frozen=True)
class CurriculumConfig:
    config_id: str
    enabled: bool
    mode: str
    schedule: tuple[CurriculumStage, ...] = ()


class CurriculumPolicy(Protocol):
    config: CurriculumConfig

    def reset_overrides(self, *, slot_index: int, episode_index: int) -> dict[str, object]:
        ...


def load_curriculum_config(config_id: str) -> CurriculumConfig:
    payload = yaml.safe_load(_config_path(config_id).read_text(encoding="utf-8")) or {}
    schedule = tuple(
        CurriculumStage(
            until_episode=None if stage.get("until_episode") is None else int(stage["until_episode"]),
            start_wave=(
                None if stage.get("start_wave") is None else int(stage["start_wave"])
            ),
            start_waves=tuple(int(value) for value in stage.get("start_waves", ())),
        )
        for stage in payload.get("schedule", ())
    )
    return CurriculumConfig(
        config_id=str(payload["config_id"]),
        enabled=bool(payload["enabled"]),
        mode=str(payload.get("mode", "disabled")),
        schedule=schedule,
    )


def build_curriculum(config_id: str) -> CurriculumPolicy:
    config = load_curriculum_config(config_id)
    if config.config_id == "curriculum_disabled_v0":
        from fight_caves_rl.curriculum.curriculum_disabled_v0 import build_curriculum

        return build_curriculum(config)
    if config.config_id == "curriculum_wave_progression_v0":
        from fight_caves_rl.curriculum.curriculum_wave_progression_v0 import build_curriculum

        return build_curriculum(config)
    if config.config_id == "curriculum_wave_progression_v2":
        from fight_caves_rl.curriculum.curriculum_wave_progression_v0 import build_curriculum

        return build_curriculum(config)
    raise ValueError(f"Unsupported curriculum config: {config_id!r}")


def _config_path(config_id: str) -> Path:
    return repo_root() / "configs" / "curriculum" / f"{config_id}.yaml"
