from __future__ import annotations

from dataclasses import dataclass

from fight_caves_rl.curriculum.registry import CurriculumConfig, CurriculumStage


@dataclass(frozen=True)
class WaveProgressionCurriculum:
    config: CurriculumConfig

    def reset_overrides(self, *, slot_index: int, episode_index: int) -> dict[str, object]:
        del slot_index
        return {"start_wave": _select_stage(self.config.schedule, int(episode_index)).start_wave}


def build_curriculum(config: CurriculumConfig) -> WaveProgressionCurriculum:
    if not config.schedule:
        raise ValueError("curriculum_wave_progression_v0 requires a non-empty schedule.")
    return WaveProgressionCurriculum(config=config)


def _select_stage(schedule: tuple[CurriculumStage, ...], episode_index: int) -> CurriculumStage:
    for stage in schedule:
        if stage.until_episode is None or episode_index < int(stage.until_episode):
            return stage
    return schedule[-1]
