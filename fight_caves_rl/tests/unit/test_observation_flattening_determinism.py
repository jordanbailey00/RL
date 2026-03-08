from copy import deepcopy

import pytest

from fight_caves_rl.envs.observation_mapping import flatten_observation


def test_observation_flattening_is_deterministic_for_same_payload():
    observation = _sample_observation()

    left = flatten_observation(observation)
    right = flatten_observation(deepcopy(observation))

    assert left == right
    assert left[0] == "headless_observation_v1"
    assert left[-12] == 1


def test_observation_flattening_fails_on_contract_order_drift():
    observation = {
        "schema_version": 1,
        "schema_id": "headless_observation_v1",
        "compatibility_policy": "v1_additive_only",
        "tick": 0,
        "episode_seed": 123,
        "player": _sample_observation()["player"],
        "wave": _sample_observation()["wave"],
        "npcs": [],
    }

    with pytest.raises(ValueError):
        flatten_observation(observation)


def _sample_observation() -> dict[str, object]:
    return {
        "schema_id": "headless_observation_v1",
        "schema_version": 1,
        "compatibility_policy": "v1_additive_only",
        "tick": 42,
        "episode_seed": 123,
        "player": {
            "tile": {"x": 1, "y": 2, "level": 0},
            "hitpoints_current": 700,
            "hitpoints_max": 700,
            "prayer_current": 43,
            "prayer_max": 43,
            "run_energy": 100,
            "run_energy_max": 100,
            "run_energy_percent": 100,
            "running": True,
            "protection_prayers": {
                "protect_from_magic": False,
                "protect_from_missiles": True,
                "protect_from_melee": False,
            },
            "lockouts": {
                "attack_locked": False,
                "food_locked": False,
                "drink_locked": False,
                "combo_locked": False,
                "busy_locked": False,
            },
            "consumables": {
                "shark_count": 20,
                "prayer_potion_dose_count": 32,
                "ammo_id": "adamant_bolts",
                "ammo_count": 1000,
            },
        },
        "wave": {
            "wave": 1,
            "rotation": 2,
            "remaining": 3,
        },
        "npcs": [
            {
                "visible_index": 0,
                "npc_index": 17,
                "id": "tz-kih",
                "tile": {"x": 4, "y": 5, "level": 0},
                "hitpoints_current": 10,
                "hitpoints_max": 10,
                "hidden": False,
                "dead": False,
                "under_attack": True,
            }
        ],
    }
