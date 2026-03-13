from __future__ import annotations

import numpy as np
import pytest

from fight_caves_rl.envs_fast.fast_policy_encoding import pack_action_from_policy, pack_joint_actions


def test_pack_action_from_policy_maps_multidiscrete_heads_to_packed_words():
    assert pack_action_from_policy(np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int32)).tolist() == [0, 0, 0, 0]
    assert pack_action_from_policy(np.asarray([1, 12, 34, 2, 0, 0], dtype=np.int32)).tolist() == [1, 12, 34, 2]
    assert pack_action_from_policy(np.asarray([2, 0, 0, 0, 7, 0], dtype=np.int32)).tolist() == [2, 7, 0, 0]
    assert pack_action_from_policy(np.asarray([3, 0, 0, 0, 0, 2], dtype=np.int32)).tolist() == [3, 2, 0, 0]


def test_pack_joint_actions_flattens_env_major_batch():
    packed = pack_joint_actions(
        np.asarray(
            [
                [0, 0, 0, 0, 0, 0],
                [2, 0, 0, 0, 3, 0],
                [1, 55, 66, 1, 0, 0],
            ],
            dtype=np.int32,
        )
    )

    assert packed.tolist() == [0, 0, 0, 0, 2, 3, 0, 0, 1, 55, 66, 1]


def test_pack_action_from_policy_rejects_out_of_range_values():
    with pytest.raises(ValueError, match="exceed the configured MultiDiscrete bounds"):
        pack_action_from_policy(np.asarray([3, 0, 0, 0, 0, 9], dtype=np.int32))
