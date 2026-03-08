from __future__ import annotations

from dataclasses import dataclass

from fight_caves_rl.curriculum.registry import CurriculumConfig


@dataclass(frozen=True)
class DisabledCurriculum:
    config: CurriculumConfig

    def reset_overrides(self, *, slot_index: int, episode_index: int) -> dict[str, object]:
        del slot_index, episode_index
        return {}


def build_curriculum(config: CurriculumConfig) -> DisabledCurriculum:
    return DisabledCurriculum(config=config)
