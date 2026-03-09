import numpy as np

from fight_caves_rl.envs.observation_views import (
    coerce_flat_observation_batch,
    observation_consumable_value,
    observation_episode_seed,
    observation_npc_health_projection,
    observation_player_hitpoints_current,
    observation_remaining,
    observation_tick,
    observation_visible_target_count,
    observation_visible_targets,
    observation_wave,
    reconstruct_raw_observation_from_flat,
)


def _sample_flat_observation() -> np.ndarray:
    row = np.zeros((134,), dtype=np.float32)
    row[0] = 1
    row[1] = 42
    row[2] = 123
    row[3] = 1
    row[4] = 2
    row[5] = 0
    row[6] = 700
    row[7] = 700
    row[8] = 43
    row[9] = 43
    row[10] = 100
    row[11] = 100
    row[12] = 100
    row[13] = 1
    row[15] = 1
    row[22] = 20
    row[23] = 32
    row[24] = 1
    row[25] = 1000
    row[26] = 63
    row[27] = 1
    row[28] = 1
    row[29] = 1
    row[30] = 1
    row[31] = 0
    row[32] = 99
    row[33] = 11
    row[34] = 4
    row[35] = 5
    row[36] = 0
    row[37] = 250
    row[38] = 250
    row[41] = 1
    row[42] = 2
    return row


def test_flat_observation_views_decode_expected_values():
    row = _sample_flat_observation()

    assert observation_tick(row) == 42
    assert observation_episode_seed(row) == 123
    assert observation_player_hitpoints_current(row) == 700
    assert observation_wave(row) == 63
    assert observation_remaining(row) == 1
    assert observation_visible_target_count(row) == 1
    assert observation_consumable_value(row, "ammo_count") == 1000
    assert observation_consumable_value(row, "shark_count") == 20
    assert observation_npc_health_projection(row) == {(99, "tztok_jad"): 250}
    assert observation_visible_targets(row) == [
        {
            "visible_index": 0,
            "npc_index": 99,
            "id": "tztok_jad",
            "tile": {"x": 4, "y": 5, "level": 0},
        }
    ]


def test_reconstruct_raw_observation_from_flat_restores_semantic_shape():
    raw = reconstruct_raw_observation_from_flat(_sample_flat_observation())

    assert raw["schema_id"] == "headless_observation_v1"
    assert raw["tick"] == 42
    assert raw["episode_seed"] == 123
    assert raw["player"]["consumables"]["ammo_id"] == "adamant_bolts"
    assert raw["wave"] == {"wave": 63, "rotation": 1, "remaining": 1}
    assert raw["npcs"][0]["id"] == "tztok_jad"
    assert raw["npcs"][0]["jad_telegraph_state"] == 2


def test_coerce_flat_observation_batch_validates_shape():
    batch = np.stack([_sample_flat_observation(), _sample_flat_observation()], axis=0)
    coerced = coerce_flat_observation_batch(batch)

    assert coerced.shape == (2, 134)
    assert coerced.dtype == np.float32
