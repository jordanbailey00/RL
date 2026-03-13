from __future__ import annotations

from copy import deepcopy
from math import isclose

import pytest

from fight_caves_rl.rewards.registry import resolve_reward_fn


def test_reward_functions_ignore_extra_future_like_payloads():
    previous = _observation(wave=1, remaining=3, player_hp=70, sharks=20, prayer_doses=32, ammo=1000)
    current = _observation(wave=1, remaining=2, player_hp=69, sharks=19, prayer_doses=31, ammo=999)
    previous_with_future = deepcopy(previous)
    current_with_future = deepcopy(current)
    previous_with_future["future_leakage"] = {"oracle_wave_clear_tick": 999999}
    current_with_future["future_leakage"] = {"incoming_damage_next_tick": 999999}
    action_result = {"action_applied": True, "rejection_reason": None}

    for config_id in ("reward_sparse_v0", "reward_shaped_v0"):
        reward_fn = resolve_reward_fn(config_id)
        baseline = reward_fn(previous, action_result, current, False, False)
        with_future = reward_fn(previous_with_future, action_result, current_with_future, False, False)
        assert isclose(baseline, with_future), config_id


def test_v2_reward_configs_do_not_fall_back_to_observation_based_reward_functions():
    for config_id in ("reward_sparse_v2", "reward_shaped_v2"):
        with pytest.raises(ValueError, match="env_backend='v2_fast'"):
            resolve_reward_fn(config_id)


def _observation(
    *,
    wave: int,
    remaining: int,
    player_hp: int,
    sharks: int,
    prayer_doses: int,
    ammo: int,
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
                "npc_index": 0,
                "id": "tz_kih",
                "hitpoints_current": 8,
                "hitpoints_max": 10,
            }
        ],
    }
