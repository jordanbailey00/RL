from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from fight_caves_rl.envs.observation_mapping import validate_observation_contract
from fight_caves_rl.envs.puffer_encoding import (
    POLICY_AMMO_IDS,
    POLICY_MAX_VISIBLE_NPCS,
    POLICY_NPC_IDS,
    POLICY_NPC_SLOT_FIELD_ORDER,
    POLICY_OBSERVATION_BASE_FIELD_ORDER,
    POLICY_OBSERVATION_SIZE,
)
from fight_caves_rl.envs.schema import (
    HEADLESS_OBSERVATION_COMPATIBILITY_POLICY,
    HEADLESS_OBSERVATION_SCHEMA,
)

POLICY_BASE_FIELD_COUNT = len(POLICY_OBSERVATION_BASE_FIELD_ORDER)
POLICY_NPC_FIELD_COUNT = len(POLICY_NPC_SLOT_FIELD_ORDER)
POLICY_BASE_FIELD_INDEX = {
    field_name: index for index, field_name in enumerate(POLICY_OBSERVATION_BASE_FIELD_ORDER)
}
POLICY_NPC_FIELD_INDEX = {
    field_name: index for index, field_name in enumerate(POLICY_NPC_SLOT_FIELD_ORDER)
}
POLICY_INDEX_TO_AMMO_ID = {index: ammo_id for index, ammo_id in enumerate(POLICY_AMMO_IDS)}
POLICY_INDEX_TO_NPC_ID = {index: npc_id for index, npc_id in enumerate(POLICY_NPC_IDS)}


def coerce_flat_observation_row(observation: Any) -> np.ndarray:
    row = np.asarray(observation, dtype=np.float32)
    if row.shape != (POLICY_OBSERVATION_SIZE,):
        raise ValueError(
            "Flat observation row layout drift: "
            f"expected {(POLICY_OBSERVATION_SIZE,)}, got {row.shape}."
        )
    return row


def coerce_flat_observation_batch(observation: Any) -> np.ndarray:
    batch = np.asarray(observation, dtype=np.float32)
    if batch.ndim != 2 or batch.shape[1] != POLICY_OBSERVATION_SIZE:
        raise ValueError(
            "Flat observation batch layout drift: "
            f"expected (*, {POLICY_OBSERVATION_SIZE}), got {batch.shape}."
        )
    return batch


def observation_tick(observation: Mapping[str, Any] | np.ndarray) -> int:
    if _is_raw_mapping(observation):
        return int(observation["tick"])
    return int(coerce_flat_observation_row(observation)[POLICY_BASE_FIELD_INDEX["tick"]])


def observation_episode_seed(observation: Mapping[str, Any] | np.ndarray) -> int:
    if _is_raw_mapping(observation):
        return int(observation["episode_seed"])
    return int(
        coerce_flat_observation_row(observation)[POLICY_BASE_FIELD_INDEX["episode_seed"]]
    )


def observation_player_hitpoints_current(observation: Mapping[str, Any] | np.ndarray) -> int:
    if _is_raw_mapping(observation):
        return int(observation["player"]["hitpoints_current"])
    return int(
        coerce_flat_observation_row(observation)[
            POLICY_BASE_FIELD_INDEX["player.hitpoints_current"]
        ]
    )


def observation_wave(observation: Mapping[str, Any] | np.ndarray) -> int:
    if _is_raw_mapping(observation):
        return int(observation["wave"]["wave"])
    return int(coerce_flat_observation_row(observation)[POLICY_BASE_FIELD_INDEX["wave.wave"]])


def observation_remaining(observation: Mapping[str, Any] | np.ndarray) -> int:
    if _is_raw_mapping(observation):
        return int(observation["wave"]["remaining"])
    return int(
        coerce_flat_observation_row(observation)[POLICY_BASE_FIELD_INDEX["wave.remaining"]]
    )


def observation_visible_target_count(observation: Mapping[str, Any] | np.ndarray) -> int:
    if _is_raw_mapping(observation):
        return int(len(observation["npcs"]))
    return int(
        coerce_flat_observation_row(observation)[
            POLICY_BASE_FIELD_INDEX["npcs.visible_count"]
        ]
    )


def observation_consumable_value(
    observation: Mapping[str, Any] | np.ndarray,
    key: str,
) -> int:
    if _is_raw_mapping(observation):
        return int(observation["player"]["consumables"][key])
    field_name = f"player.consumables.{key}"
    return int(coerce_flat_observation_row(observation)[POLICY_BASE_FIELD_INDEX[field_name]])


def observation_visible_targets(
    observation: Mapping[str, Any] | np.ndarray,
) -> list[dict[str, Any]]:
    if _is_raw_mapping(observation):
        validate_observation_contract(observation)
        return [
            {
                "visible_index": int(npc["visible_index"]),
                "npc_index": int(npc["npc_index"]),
                "id": str(npc["id"]),
                "tile": {
                    "x": int(npc["tile"]["x"]),
                    "y": int(npc["tile"]["y"]),
                    "level": int(npc["tile"]["level"]),
                },
            }
            for npc in observation["npcs"]
        ]

    row = coerce_flat_observation_row(observation)
    targets: list[dict[str, Any]] = []
    for slot_index in range(POLICY_MAX_VISIBLE_NPCS):
        offset = _npc_offset(slot_index)
        if int(row[offset + POLICY_NPC_FIELD_INDEX["present"]]) == 0:
            continue
        targets.append(
            {
                "visible_index": int(row[offset + POLICY_NPC_FIELD_INDEX["visible_index"]]),
                "npc_index": int(row[offset + POLICY_NPC_FIELD_INDEX["npc_index"]]),
                "id": _decode_npc_id(
                    int(row[offset + POLICY_NPC_FIELD_INDEX["id_code"]])
                ),
                "tile": {
                    "x": int(row[offset + POLICY_NPC_FIELD_INDEX["tile.x"]]),
                    "y": int(row[offset + POLICY_NPC_FIELD_INDEX["tile.y"]]),
                    "level": int(row[offset + POLICY_NPC_FIELD_INDEX["tile.level"]]),
                },
            }
        )
    return targets


def observation_npc_health_projection(
    observation: Mapping[str, Any] | np.ndarray,
) -> dict[tuple[int, str], int]:
    if _is_raw_mapping(observation):
        return {
            (int(npc["npc_index"]), str(npc["id"])): int(npc["hitpoints_current"])
            for npc in observation.get("npcs", [])
        }

    row = coerce_flat_observation_row(observation)
    projection: dict[tuple[int, str], int] = {}
    for slot_index in range(POLICY_MAX_VISIBLE_NPCS):
        offset = _npc_offset(slot_index)
        if int(row[offset + POLICY_NPC_FIELD_INDEX["present"]]) == 0:
            continue
        projection[
            (
                int(row[offset + POLICY_NPC_FIELD_INDEX["npc_index"]]),
                _decode_npc_id(int(row[offset + POLICY_NPC_FIELD_INDEX["id_code"]])),
            )
        ] = int(row[offset + POLICY_NPC_FIELD_INDEX["hitpoints_current"]])
    return projection


def reconstruct_raw_observation_from_flat(observation: np.ndarray) -> dict[str, Any]:
    row = coerce_flat_observation_row(observation)
    raw_npcs: list[dict[str, Any]] = []
    for slot_index in range(POLICY_MAX_VISIBLE_NPCS):
        offset = _npc_offset(slot_index)
        if int(row[offset + POLICY_NPC_FIELD_INDEX["present"]]) == 0:
            continue
        raw_npcs.append(
            {
                "visible_index": int(row[offset + POLICY_NPC_FIELD_INDEX["visible_index"]]),
                "npc_index": int(row[offset + POLICY_NPC_FIELD_INDEX["npc_index"]]),
                "id": _decode_npc_id(
                    int(row[offset + POLICY_NPC_FIELD_INDEX["id_code"]])
                ),
                "tile": {
                    "x": int(row[offset + POLICY_NPC_FIELD_INDEX["tile.x"]]),
                    "y": int(row[offset + POLICY_NPC_FIELD_INDEX["tile.y"]]),
                    "level": int(row[offset + POLICY_NPC_FIELD_INDEX["tile.level"]]),
                },
                "hitpoints_current": int(
                    row[offset + POLICY_NPC_FIELD_INDEX["hitpoints_current"]]
                ),
                "hitpoints_max": int(row[offset + POLICY_NPC_FIELD_INDEX["hitpoints_max"]]),
                "hidden": bool(int(row[offset + POLICY_NPC_FIELD_INDEX["hidden"]])),
                "dead": bool(int(row[offset + POLICY_NPC_FIELD_INDEX["dead"]])),
                "under_attack": bool(
                    int(row[offset + POLICY_NPC_FIELD_INDEX["under_attack"]])
                ),
                "jad_telegraph_state": int(
                    row[offset + POLICY_NPC_FIELD_INDEX["jad_telegraph_state"]]
                ),
            }
        )

    return {
        "schema_id": HEADLESS_OBSERVATION_SCHEMA.contract_id,
        "schema_version": int(row[POLICY_BASE_FIELD_INDEX["schema_version"]]),
        "compatibility_policy": HEADLESS_OBSERVATION_COMPATIBILITY_POLICY,
        "tick": int(row[POLICY_BASE_FIELD_INDEX["tick"]]),
        "episode_seed": int(row[POLICY_BASE_FIELD_INDEX["episode_seed"]]),
        "player": {
            "tile": {
                "x": int(row[POLICY_BASE_FIELD_INDEX["player.tile.x"]]),
                "y": int(row[POLICY_BASE_FIELD_INDEX["player.tile.y"]]),
                "level": int(row[POLICY_BASE_FIELD_INDEX["player.tile.level"]]),
            },
            "hitpoints_current": int(
                row[POLICY_BASE_FIELD_INDEX["player.hitpoints_current"]]
            ),
            "hitpoints_max": int(row[POLICY_BASE_FIELD_INDEX["player.hitpoints_max"]]),
            "prayer_current": int(row[POLICY_BASE_FIELD_INDEX["player.prayer_current"]]),
            "prayer_max": int(row[POLICY_BASE_FIELD_INDEX["player.prayer_max"]]),
            "run_energy": int(row[POLICY_BASE_FIELD_INDEX["player.run_energy"]]),
            "run_energy_max": int(row[POLICY_BASE_FIELD_INDEX["player.run_energy_max"]]),
            "run_energy_percent": int(
                row[POLICY_BASE_FIELD_INDEX["player.run_energy_percent"]]
            ),
            "running": bool(int(row[POLICY_BASE_FIELD_INDEX["player.running"]])),
            "protection_prayers": {
                "protect_from_magic": bool(
                    int(
                        row[
                            POLICY_BASE_FIELD_INDEX[
                                "player.protection_prayers.protect_from_magic"
                            ]
                        ]
                    )
                ),
                "protect_from_missiles": bool(
                    int(
                        row[
                            POLICY_BASE_FIELD_INDEX[
                                "player.protection_prayers.protect_from_missiles"
                            ]
                        ]
                    )
                ),
                "protect_from_melee": bool(
                    int(
                        row[
                            POLICY_BASE_FIELD_INDEX[
                                "player.protection_prayers.protect_from_melee"
                            ]
                        ]
                    )
                ),
            },
            "lockouts": {
                "attack_locked": bool(
                    int(row[POLICY_BASE_FIELD_INDEX["player.lockouts.attack_locked"]])
                ),
                "food_locked": bool(
                    int(row[POLICY_BASE_FIELD_INDEX["player.lockouts.food_locked"]])
                ),
                "drink_locked": bool(
                    int(row[POLICY_BASE_FIELD_INDEX["player.lockouts.drink_locked"]])
                ),
                "combo_locked": bool(
                    int(row[POLICY_BASE_FIELD_INDEX["player.lockouts.combo_locked"]])
                ),
                "busy_locked": bool(
                    int(row[POLICY_BASE_FIELD_INDEX["player.lockouts.busy_locked"]])
                ),
            },
            "consumables": {
                "shark_count": int(
                    row[POLICY_BASE_FIELD_INDEX["player.consumables.shark_count"]]
                ),
                "prayer_potion_dose_count": int(
                    row[
                        POLICY_BASE_FIELD_INDEX[
                            "player.consumables.prayer_potion_dose_count"
                        ]
                    ]
                ),
                "ammo_id": _decode_ammo_id(
                    int(row[POLICY_BASE_FIELD_INDEX["player.consumables.ammo_id_code"]])
                ),
                "ammo_count": int(
                    row[POLICY_BASE_FIELD_INDEX["player.consumables.ammo_count"]]
                ),
            },
        },
        "wave": {
            "wave": int(row[POLICY_BASE_FIELD_INDEX["wave.wave"]]),
            "rotation": int(row[POLICY_BASE_FIELD_INDEX["wave.rotation"]]),
            "remaining": int(row[POLICY_BASE_FIELD_INDEX["wave.remaining"]]),
        },
        "npcs": raw_npcs,
    }


def _decode_ammo_id(code: int) -> str:
    try:
        return POLICY_INDEX_TO_AMMO_ID[int(code)]
    except KeyError as exc:
        raise ValueError(f"Unknown flat observation ammo code: {code!r}") from exc


def _decode_npc_id(code: int) -> str:
    try:
        return POLICY_INDEX_TO_NPC_ID[int(code)]
    except KeyError as exc:
        raise ValueError(f"Unknown flat observation npc code: {code!r}") from exc


def _npc_offset(slot_index: int) -> int:
    return POLICY_BASE_FIELD_COUNT + (int(slot_index) * POLICY_NPC_FIELD_COUNT)


def _is_raw_mapping(observation: Any) -> bool:
    return isinstance(observation, Mapping)
