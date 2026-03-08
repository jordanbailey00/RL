from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from fight_caves_rl.envs.schema import VersionedContract
from fight_caves_rl.replay.trace_packs import (
    project_episode_state_for_determinism,
    semantic_digest,
)

REPLAY_PACK_SCHEMA = VersionedContract(
    contract_id="replay_pack_v0",
    version=0,
    compatibility_policy="replace_on_schema_change",
)


@dataclass(frozen=True)
class ReplayEpisode:
    seed: int
    episode_reset_summary: dict[str, Any]
    semantic_episode_state: dict[str, Any]
    steps_taken: int
    captured_steps: int
    replay_step_cadence: int
    terminated: bool
    truncated: bool
    terminal_reason: str | None
    trajectory_digest: str
    replay_digest: str
    final_semantic_observation: dict[str, Any]
    steps: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayPack:
    schema_id: str
    schema_version: int
    config_id: str
    checkpoint_path: str
    checkpoint_metadata_path: str
    checkpoint_metadata: dict[str, Any]
    seed_pack: str
    seed_pack_version: int
    policy_mode: str
    reward_config_id: str
    curriculum_config_id: str
    replay_step_cadence: int
    summary_digest: str
    episodes: tuple[ReplayEpisode, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["episodes"] = [episode.to_dict() for episode in self.episodes]
        return payload


def build_replay_episode(
    *,
    seed: int,
    episode_reset_summary: Mapping[str, Any],
    episode_state: Mapping[str, Any],
    steps_taken: int,
    terminated: bool,
    truncated: bool,
    terminal_reason: str | None,
    trajectory_digest: str,
    final_semantic_observation: Mapping[str, Any],
    full_steps: Sequence[Mapping[str, Any]],
    replay_step_cadence: int,
) -> ReplayEpisode:
    captured_steps = sample_replay_steps(full_steps, replay_step_cadence)
    return ReplayEpisode(
        seed=int(seed),
        episode_reset_summary=dict(episode_reset_summary),
        semantic_episode_state=project_episode_state_for_determinism(episode_state),
        steps_taken=int(steps_taken),
        captured_steps=len(captured_steps),
        replay_step_cadence=int(replay_step_cadence),
        terminated=bool(terminated),
        truncated=bool(truncated),
        terminal_reason=None if terminal_reason is None else str(terminal_reason),
        trajectory_digest=str(trajectory_digest),
        replay_digest=semantic_digest(captured_steps),
        final_semantic_observation=dict(final_semantic_observation),
        steps=tuple(dict(step) for step in captured_steps),
    )


def build_replay_pack(
    *,
    config_id: str,
    checkpoint_path: str | Path,
    checkpoint_metadata_path: str | Path,
    checkpoint_metadata: Mapping[str, Any],
    seed_pack: str,
    seed_pack_version: int,
    policy_mode: str,
    reward_config_id: str,
    curriculum_config_id: str,
    replay_step_cadence: int,
    summary_digest: str,
    episodes: Sequence[ReplayEpisode],
) -> ReplayPack:
    return ReplayPack(
        schema_id=REPLAY_PACK_SCHEMA.contract_id,
        schema_version=REPLAY_PACK_SCHEMA.version,
        config_id=str(config_id),
        checkpoint_path=str(Path(checkpoint_path)),
        checkpoint_metadata_path=str(Path(checkpoint_metadata_path)),
        checkpoint_metadata=dict(checkpoint_metadata),
        seed_pack=str(seed_pack),
        seed_pack_version=int(seed_pack_version),
        policy_mode=str(policy_mode),
        reward_config_id=str(reward_config_id),
        curriculum_config_id=str(curriculum_config_id),
        replay_step_cadence=int(replay_step_cadence),
        summary_digest=str(summary_digest),
        episodes=tuple(episodes),
    )


def write_replay_pack(path: str | Path, replay_pack: ReplayPack) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(replay_pack.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def sample_replay_steps(
    steps: Sequence[Mapping[str, Any]],
    replay_step_cadence: int,
) -> list[dict[str, Any]]:
    cadence = int(replay_step_cadence)
    if cadence <= 0:
        raise ValueError(f"replay_step_cadence must be >= 1, got {replay_step_cadence}.")
    if not steps:
        return []

    sampled = [
        dict(step)
        for step in steps
        if int(step["step_index"]) % cadence == 0
    ]
    final_step = dict(steps[-1])
    if int(sampled[-1]["step_index"]) != int(final_step["step_index"]):
        sampled.append(final_step)
    return sampled
