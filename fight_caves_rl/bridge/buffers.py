from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fight_caves_rl.bridge.protocol import BatchSlotResetResult, BatchSlotStepResult
from fight_caves_rl.envs.observation_views import (
    coerce_flat_observation_row,
    observation_episode_seed,
    observation_remaining,
    observation_visible_target_count,
    observation_wave,
)
from fight_caves_rl.envs.puffer_encoding import encode_observation_for_policy
from fight_caves_rl.envs.schema import HEADLESS_ACTION_REJECT_REASONS

TERMINAL_REASON_TO_CODE = {
    None: 0,
    "player_death": 1,
    "cave_complete": 2,
    "max_tick_cap": 3,
}
REJECTION_REASON_TO_CODE = {
    None: 0,
    **{reason: index + 1 for index, reason in enumerate(HEADLESS_ACTION_REJECT_REASONS)},
}


@dataclass(frozen=True)
class BatchResetBuffers:
    slot_indices: np.ndarray
    policy_observations: np.ndarray
    episode_seeds: np.ndarray
    waves: np.ndarray
    remaining: np.ndarray
    visible_target_counts: np.ndarray


@dataclass(frozen=True)
class BatchStepBuffers:
    slot_indices: np.ndarray
    policy_observations: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    action_ids: np.ndarray
    action_applied: np.ndarray
    rejection_reason_codes: np.ndarray
    terminal_reason_codes: np.ndarray
    episode_steps: np.ndarray
    episode_returns: np.ndarray
    waves: np.ndarray
    remaining: np.ndarray
    visible_target_counts: np.ndarray


def build_reset_buffers(results: Sequence[BatchSlotResetResult]) -> BatchResetBuffers:
    ordered = sorted(results, key=lambda result: result.slot_index)
    observations = [_policy_observation(result) for result in ordered]
    return BatchResetBuffers(
        slot_indices=np.asarray([result.slot_index for result in ordered], dtype=np.int32),
        policy_observations=np.stack(observations, axis=0).astype(np.float32, copy=False),
        episode_seeds=np.asarray(
            [observation_episode_seed(_semantic_observation(result)) for result in ordered],
            dtype=np.int64,
        ),
        waves=np.asarray(
            [observation_wave(_semantic_observation(result)) for result in ordered],
            dtype=np.int32,
        ),
        remaining=np.asarray(
            [observation_remaining(_semantic_observation(result)) for result in ordered],
            dtype=np.int32,
        ),
        visible_target_counts=np.asarray(
            [observation_visible_target_count(_semantic_observation(result)) for result in ordered],
            dtype=np.int32,
        ),
    )


def build_step_buffers(results: Sequence[BatchSlotStepResult]) -> BatchStepBuffers:
    ordered = sorted(results, key=lambda result: result.slot_index)
    observations = [_policy_observation(result) for result in ordered]
    return BatchStepBuffers(
        slot_indices=np.asarray([result.slot_index for result in ordered], dtype=np.int32),
        policy_observations=np.stack(observations, axis=0).astype(np.float32, copy=False),
        rewards=np.asarray([float(result.reward) for result in ordered], dtype=np.float32),
        terminated=np.asarray([bool(result.terminated) for result in ordered], dtype=np.bool_),
        truncated=np.asarray([bool(result.truncated) for result in ordered], dtype=np.bool_),
        action_ids=np.asarray([int(result.action.action_id) for result in ordered], dtype=np.int32),
        action_applied=np.asarray(
            [bool(result.info["action_result"]["action_applied"]) for result in ordered],
            dtype=np.bool_,
        ),
        rejection_reason_codes=np.asarray(
            [
                REJECTION_REASON_TO_CODE[result.info["action_result"]["rejection_reason"]]
                for result in ordered
            ],
            dtype=np.int32,
        ),
        terminal_reason_codes=np.asarray(
            [TERMINAL_REASON_TO_CODE[result.info["terminal_reason"]] for result in ordered],
            dtype=np.int32,
        ),
        episode_steps=np.asarray(
            [int(result.info["episode_steps"]) for result in ordered],
            dtype=np.int32,
        ),
        episode_returns=np.asarray(
            [float(result.info["episode_return"]) for result in ordered],
            dtype=np.float32,
        ),
        waves=np.asarray(
            [observation_wave(_semantic_observation(result)) for result in ordered],
            dtype=np.int32,
        ),
        remaining=np.asarray(
            [observation_remaining(_semantic_observation(result)) for result in ordered],
            dtype=np.int32,
        ),
        visible_target_counts=np.asarray(
            [int(result.info["visible_target_count"]) for result in ordered],
            dtype=np.int32,
        ),
    )


def _policy_observation(result: BatchSlotResetResult | BatchSlotStepResult) -> np.ndarray:
    if result.flat_observation is not None:
        return coerce_flat_observation_row(result.flat_observation)
    if result.observation is None:
        raise ValueError("Batch result is missing both raw and flat observations.")
    return encode_observation_for_policy(result.observation)


def _semantic_observation(result: BatchSlotResetResult | BatchSlotStepResult):
    if result.flat_observation is not None:
        return result.flat_observation
    if result.observation is None:
        raise ValueError("Batch result is missing both raw and flat observations.")
    return result.observation
