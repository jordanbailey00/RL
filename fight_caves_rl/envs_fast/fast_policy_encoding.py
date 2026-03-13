from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from fight_caves_rl.envs_fast.fast_spaces import FAST_ACTION_NVECS

FAST_PACKED_ACTION_WORD_COUNT = 4
FAST_POLICY_ACTION_SIZE = len(FAST_ACTION_NVECS)


def pack_action_from_policy(action: Sequence[int] | np.ndarray) -> np.ndarray:
    vector = np.asarray(action, dtype=np.int32).reshape(-1)
    _validate_policy_action_vector(vector)
    packed = np.zeros((FAST_PACKED_ACTION_WORD_COUNT,), dtype=np.int32)
    action_id = int(vector[0])
    packed[0] = action_id
    if action_id == 1:
        packed[1] = int(vector[1])
        packed[2] = int(vector[2])
        packed[3] = int(vector[3])
    elif action_id == 2:
        packed[1] = int(vector[4])
    elif action_id == 3:
        packed[1] = int(vector[5])
    return packed


def pack_joint_actions(actions: np.ndarray | Sequence[Sequence[int]]) -> np.ndarray:
    action_batch = np.asarray(actions, dtype=np.int32)
    if action_batch.ndim != 2 or action_batch.shape[1] != FAST_POLICY_ACTION_SIZE:
        raise ValueError(
            "Fast policy action batch layout drift: "
            f"expected (*, {FAST_POLICY_ACTION_SIZE}), got {action_batch.shape}."
        )
    packed = np.zeros((action_batch.shape[0], FAST_PACKED_ACTION_WORD_COUNT), dtype=np.int32)
    packed[:, 0] = action_batch[:, 0]

    walk_mask = action_batch[:, 0] == 1
    attack_mask = action_batch[:, 0] == 2
    prayer_mask = action_batch[:, 0] == 3

    packed[walk_mask, 1] = action_batch[walk_mask, 1]
    packed[walk_mask, 2] = action_batch[walk_mask, 2]
    packed[walk_mask, 3] = action_batch[walk_mask, 3]
    packed[attack_mask, 1] = action_batch[attack_mask, 4]
    packed[prayer_mask, 1] = action_batch[prayer_mask, 5]
    return packed.reshape(-1)


def _validate_policy_action_vector(vector: np.ndarray) -> None:
    if vector.shape != (FAST_POLICY_ACTION_SIZE,):
        raise ValueError(
            "Fast policy action layout drift: "
            f"expected {(FAST_POLICY_ACTION_SIZE,)}, got {vector.shape}."
        )
    if np.any(vector < 0):
        raise ValueError(f"Fast policy action values must be >= 0, got {vector.tolist()!r}.")
    if np.any(vector >= FAST_ACTION_NVECS):
        raise ValueError(
            "Fast policy action values exceed the configured MultiDiscrete bounds: "
            f"{vector.tolist()!r} vs {FAST_ACTION_NVECS.tolist()!r}."
        )
