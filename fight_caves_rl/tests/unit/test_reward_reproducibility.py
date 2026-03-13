from __future__ import annotations

from math import isclose

import numpy as np

from fight_caves_rl.contracts.reward_feature_schema import REWARD_FEATURE_NAMES
from fight_caves_rl.envs_fast.fast_reward_adapter import FastRewardAdapter
from fight_caves_rl.rewards.registry import load_reward_config, resolve_reward_fn


def test_reward_configs_load_with_expected_modes():
    assert load_reward_config("reward_sparse_v0").mode == "sparse"
    assert load_reward_config("reward_shaped_v0").mode == "shaped"
    assert load_reward_config("reward_sparse_v2").mode == "sparse"
    assert load_reward_config("reward_shaped_v2").mode == "shaped"


def test_reward_functions_are_reproducible_for_identical_inputs():
    previous = _observation(
        wave=1,
        remaining=2,
        player_hp=70,
        sharks=20,
        prayer_doses=32,
        ammo=1000,
        npcs=((0, "tz_kih", 10, 10),),
    )
    current = _observation(
        wave=2,
        remaining=4,
        player_hp=68,
        sharks=19,
        prayer_doses=32,
        ammo=998,
        npcs=((0, "tz_kih", 6, 10),),
    )
    action_result = {"action_applied": True, "rejection_reason": None}

    sparse = resolve_reward_fn("reward_sparse_v0")
    shaped = resolve_reward_fn("reward_shaped_v0")

    sparse_a = sparse(previous, action_result, current, False, False)
    sparse_b = sparse(previous, action_result, current, False, False)
    shaped_a = shaped(previous, action_result, current, False, False)
    shaped_b = shaped(previous, action_result, current, False, False)

    assert isclose(sparse_a, sparse_b)
    assert isclose(shaped_a, shaped_b)
    assert shaped_a != sparse_a


def test_fast_reward_adapter_is_reproducible_for_identical_feature_batches():
    adapter = FastRewardAdapter.from_config_id("reward_shaped_v2")
    features = np.zeros((2, len(REWARD_FEATURE_NAMES)), dtype=np.float32)
    features[0, 0] = 5.0
    features[0, 3] = 1.0
    features[1, 6] = 1.0

    weighted_a = adapter.weight_batch(features)
    weighted_b = adapter.weight_batch(features.copy())

    assert np.allclose(weighted_a, weighted_b)


def _observation(
    *,
    wave: int,
    remaining: int,
    player_hp: int,
    sharks: int,
    prayer_doses: int,
    ammo: int,
    npcs: tuple[tuple[int, str, int, int], ...],
) -> dict[str, object]:
    return {
        "wave": {
            "wave": wave,
            "remaining": remaining,
        },
        "player": {
            "hitpoints_current": player_hp,
            "consumables": {
                "shark_count": sharks,
                "prayer_potion_dose_count": prayer_doses,
                "ammo_count": ammo,
            },
        },
        "npcs": [
            {
                "npc_index": npc_index,
                "id": npc_id,
                "hitpoints_current": current_hp,
                "hitpoints_max": max_hp,
            }
            for npc_index, npc_id, current_hp, max_hp in npcs
        ],
    }
