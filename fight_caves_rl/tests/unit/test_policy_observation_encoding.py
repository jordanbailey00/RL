import numpy as np

from fight_caves_rl.envs.puffer_encoding import (
    POLICY_OBSERVATION_SIZE,
    encode_observation_for_policy,
)


def test_policy_encoding_includes_jad_telegraph_field():
    observation = {
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
        "wave": {"wave": 63, "rotation": 1, "remaining": 1},
        "npcs": [
            {
                "visible_index": 0,
                "npc_index": 99,
                "id": "tztok_jad",
                "tile": {"x": 4, "y": 5, "level": 0},
                "hitpoints_current": 250,
                "hitpoints_max": 250,
                "hidden": False,
                "dead": False,
                "under_attack": True,
                "jad_telegraph_state": 1,
            }
        ],
    }

    encoded = encode_observation_for_policy(observation)

    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (POLICY_OBSERVATION_SIZE,)
    assert encoded[42] == 1.0
