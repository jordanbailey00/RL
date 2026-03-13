import numpy as np

from fight_caves_rl.envs.puffer_encoding import (
    POLICY_OBSERVATION_SIZE,
    decode_action_from_policy,
    encode_action_for_policy,
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


def test_policy_action_round_trip_preserves_semantics():
    walk = decode_action_from_policy(np.asarray((1, 3200, 3201, 0, 0, 0), dtype=np.int32))
    attack = decode_action_from_policy(np.asarray((2, 0, 0, 0, 3, 0), dtype=np.int32))
    prayer = decode_action_from_policy(np.asarray((3, 0, 0, 0, 0, 1), dtype=np.int32))
    idle = decode_action_from_policy(np.asarray((0, 0, 0, 0, 0, 0), dtype=np.int32))

    assert walk.tile is not None
    assert (walk.tile.x, walk.tile.y, walk.tile.level) == (3200, 3201, 0)
    assert walk.visible_npc_index is None
    assert walk.prayer is None

    assert attack.visible_npc_index == 3
    assert attack.tile is None
    assert attack.prayer is None

    assert prayer.prayer == "protect_from_missiles"
    assert prayer.tile is None
    assert prayer.visible_npc_index is None

    encoded_idle = encode_action_for_policy(idle)
    np.testing.assert_array_equal(encoded_idle, np.asarray((0, 0, 0, 0, 0, 0), dtype=np.int64))
