from __future__ import annotations

from dataclasses import dataclass

from fight_caves_rl.curriculum.registry import CurriculumConfig, CurriculumStage


@dataclass(frozen=True)
class WaveProgressionCurriculum:
    config: CurriculumConfig

    def reset_overrides(self, *, slot_index: int, episode_index: int) -> dict[str, object]:
        del slot_index
        stage = _select_stage(self.config.schedule, int(episode_index))
        return {"start_wave": _resolve_start_wave(stage, int(episode_index))}


def build_curriculum(config: CurriculumConfig) -> WaveProgressionCurriculum:
    if not config.schedule:
        raise ValueError("curriculum_wave_progression_v0 requires a non-empty schedule.")
    return WaveProgressionCurriculum(config=config)


def _select_stage(schedule: tuple[CurriculumStage, ...], episode_index: int) -> CurriculumStage:
    for stage in schedule:
        if stage.until_episode is None or episode_index < int(stage.until_episode):
            return stage
    return schedule[-1]


def _resolve_start_wave(stage: CurriculumStage, episode_index: int) -> int:
    if stage.start_waves:
        return int(stage.start_waves[int(episode_index) % len(stage.start_waves)])
    if stage.start_wave is None:
        raise ValueError("Wave progression stage must define start_wave or start_waves.")
    return int(stage.start_wave)
