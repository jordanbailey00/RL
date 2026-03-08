from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Sequence

from fight_caves_rl.envs.schema import VersionedContract
from fight_caves_rl.replay.replay_export import ReplayPack

REPLAY_INDEX_SCHEMA = VersionedContract(
    contract_id="replay_index_v0",
    version=0,
    compatibility_policy="replace_on_schema_change",
)


@dataclass(frozen=True)
class ReplayIndexEntry:
    episode_index: int
    seed: int
    steps_taken: int
    captured_steps: int
    replay_step_cadence: int
    trajectory_digest: str
    replay_digest: str
    terminated: bool
    truncated: bool
    terminal_reason: str | None
    replay_pointer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayIndex:
    schema_id: str
    schema_version: int
    config_id: str
    seed_pack: str
    seed_pack_version: int
    summary_digest: str
    replay_pack_filename: str
    eval_summary_filename: str
    checkpoint_format_id: str
    checkpoint_format_version: int
    policy_id: str
    reward_config_id: str
    curriculum_config_id: str
    entries: tuple[ReplayIndexEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["entries"] = [entry.to_dict() for entry in self.entries]
        return payload


def build_replay_index(
    *,
    replay_pack: ReplayPack,
    replay_pack_filename: str,
    eval_summary_filename: str,
    checkpoint_format_id: str,
    checkpoint_format_version: int,
    policy_id: str,
) -> ReplayIndex:
    entries = tuple(
        ReplayIndexEntry(
            episode_index=index,
            seed=int(episode.seed),
            steps_taken=int(episode.steps_taken),
            captured_steps=int(episode.captured_steps),
            replay_step_cadence=int(episode.replay_step_cadence),
            trajectory_digest=str(episode.trajectory_digest),
            replay_digest=str(episode.replay_digest),
            terminated=bool(episode.terminated),
            truncated=bool(episode.truncated),
            terminal_reason=None if episode.terminal_reason is None else str(episode.terminal_reason),
            replay_pointer=f"episodes[{index}]",
        )
        for index, episode in enumerate(replay_pack.episodes)
    )
    return ReplayIndex(
        schema_id=REPLAY_INDEX_SCHEMA.contract_id,
        schema_version=REPLAY_INDEX_SCHEMA.version,
        config_id=str(replay_pack.config_id),
        seed_pack=str(replay_pack.seed_pack),
        seed_pack_version=int(replay_pack.seed_pack_version),
        summary_digest=str(replay_pack.summary_digest),
        replay_pack_filename=str(replay_pack_filename),
        eval_summary_filename=str(eval_summary_filename),
        checkpoint_format_id=str(checkpoint_format_id),
        checkpoint_format_version=int(checkpoint_format_version),
        policy_id=str(policy_id),
        reward_config_id=str(replay_pack.reward_config_id),
        curriculum_config_id=str(replay_pack.curriculum_config_id),
        entries=entries,
    )


def write_replay_index(path: str | Path, replay_index: ReplayIndex) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(replay_index.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
