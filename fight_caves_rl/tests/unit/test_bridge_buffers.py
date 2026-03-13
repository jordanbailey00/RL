import numpy as np

from fight_caves_rl.bridge.buffers import (
    build_reset_buffers,
    build_step_buffers,
    build_vecenv_reset_buffers,
    build_vecenv_step_buffers,
)
from fight_caves_rl.bridge.protocol import BatchSlotResetResult, BatchSlotStepResult
from fight_caves_rl.envs.action_mapping import normalize_action
from fight_caves_rl.envs.schema import HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA


def _flat_row(fill_value: float) -> np.ndarray:
    return np.full(
        (HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.feature_count,),
        fill_value,
        dtype=np.float32,
    )


def test_build_vecenv_reset_buffers_matches_full_builder_for_used_fields():
    results = (
        BatchSlotResetResult(
            slot_index=0,
            observation=None,
            flat_observation=_flat_row(1.0),
            episode_state=None,
            info={},
        ),
        BatchSlotResetResult(
            slot_index=1,
            observation=None,
            flat_observation=_flat_row(2.0),
            episode_state=None,
            info={},
        ),
    )

    full = build_reset_buffers(results)
    fast = build_vecenv_reset_buffers(results)

    np.testing.assert_array_equal(fast.slot_indices, full.slot_indices)
    np.testing.assert_allclose(fast.policy_observations, full.policy_observations)


def test_build_vecenv_step_buffers_matches_full_builder_for_used_fields():
    idle = normalize_action(0)
    results = (
        BatchSlotStepResult(
            slot_index=0,
            action=idle,
            observation=None,
            flat_observation=_flat_row(3.0),
            reward=1.5,
            terminated=False,
            truncated=False,
            action_result={"action_applied": True, "rejection_reason": None},
            visible_target_count=0,
            episode_steps=1,
            episode_return=1.5,
            terminal_reason=None,
            visible_targets=None,
            info={},
        ),
        BatchSlotStepResult(
            slot_index=1,
            action=idle,
            observation=None,
            flat_observation=_flat_row(4.0),
            reward=-0.25,
            terminated=True,
            truncated=False,
            action_result={"action_applied": False, "rejection_reason": "PlayerBusy"},
            visible_target_count=2,
            episode_steps=5,
            episode_return=3.25,
            terminal_reason="player_death",
            visible_targets=None,
            info={},
        ),
    )

    full = build_step_buffers(results)
    fast = build_vecenv_step_buffers(results)

    np.testing.assert_array_equal(fast.slot_indices, full.slot_indices)
    np.testing.assert_allclose(fast.policy_observations, full.policy_observations)
    np.testing.assert_allclose(fast.rewards, full.rewards)
    np.testing.assert_array_equal(fast.terminated, full.terminated)
    np.testing.assert_array_equal(fast.truncated, full.truncated)
