from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from gymnasium import spaces

from fight_caves_rl.envs.action_mapping import NormalizedAction, normalize_action
from fight_caves_rl.envs.observation_mapping import validate_observation_contract
from fight_caves_rl.envs.schema import (
    HEADLESS_PROTECTION_PRAYER_IDS,
    PUFFER_POLICY_ACTION_SCHEMA,
    PUFFER_POLICY_OBSERVATION_SCHEMA,
)

POLICY_MAX_VISIBLE_NPCS = 8
POLICY_WORLD_X_CATEGORIES = 16_384
POLICY_WORLD_Y_CATEGORIES = 16_384
POLICY_WORLD_LEVEL_CATEGORIES = 4

POLICY_AMMO_IDS = ("", "adamant_bolts")
POLICY_NPC_IDS = (
    "tz_kih",
    "tz_kih_spawn_point",
    "tz_kek",
    "tz_kek_spawn_point",
    "tz_kek_spawn",
    "tok_xil",
    "tok_xil_spawn_point",
    "yt_mej_kot",
    "yt_mej_kot_spawn_point",
    "ket_zek",
    "ket_zek_spawn_point",
    "tztok_jad",
    "yt_hur_kot",
)

POLICY_ACTION_HEAD_ORDER = (
    "action_id",
    "tile_x",
    "tile_y",
    "tile_level",
    "visible_npc_index",
    "prayer_index",
)

POLICY_OBSERVATION_BASE_FIELD_ORDER = (
    "schema_version",
    "tick",
    "episode_seed",
    "player.tile.x",
    "player.tile.y",
    "player.tile.level",
    "player.hitpoints_current",
    "player.hitpoints_max",
    "player.prayer_current",
    "player.prayer_max",
    "player.run_energy",
    "player.run_energy_max",
    "player.run_energy_percent",
    "player.running",
    "player.protection_prayers.protect_from_magic",
    "player.protection_prayers.protect_from_missiles",
    "player.protection_prayers.protect_from_melee",
    "player.lockouts.attack_locked",
    "player.lockouts.food_locked",
    "player.lockouts.drink_locked",
    "player.lockouts.combo_locked",
    "player.lockouts.busy_locked",
    "player.consumables.shark_count",
    "player.consumables.prayer_potion_dose_count",
    "player.consumables.ammo_id_code",
    "player.consumables.ammo_count",
    "wave.wave",
    "wave.rotation",
    "wave.remaining",
    "npcs.visible_count",
)

POLICY_NPC_SLOT_FIELD_ORDER = (
    "present",
    "visible_index",
    "npc_index",
    "id_code",
    "tile.x",
    "tile.y",
    "tile.level",
    "hitpoints_current",
    "hitpoints_max",
    "hidden",
    "dead",
    "under_attack",
    "jad_telegraph_state",
)

POLICY_ACTION_NVECS = np.asarray(
    (
        7,
        POLICY_WORLD_X_CATEGORIES,
        POLICY_WORLD_Y_CATEGORIES,
        POLICY_WORLD_LEVEL_CATEGORIES,
        POLICY_MAX_VISIBLE_NPCS,
        len(HEADLESS_PROTECTION_PRAYER_IDS),
    ),
    dtype=np.int64,
)

POLICY_AMMO_ID_TO_INDEX = {ammo_id: index for index, ammo_id in enumerate(POLICY_AMMO_IDS)}
POLICY_NPC_ID_TO_INDEX = {npc_id: index for index, npc_id in enumerate(POLICY_NPC_IDS)}
POLICY_PRAYER_TO_INDEX = {
    prayer: index for index, prayer in enumerate(HEADLESS_PROTECTION_PRAYER_IDS)
}
POLICY_INDEX_TO_PRAYER = {
    index: prayer for prayer, index in POLICY_PRAYER_TO_INDEX.items()
}

POLICY_OBSERVATION_SIZE = len(POLICY_OBSERVATION_BASE_FIELD_ORDER) + (
    POLICY_MAX_VISIBLE_NPCS * len(POLICY_NPC_SLOT_FIELD_ORDER)
)


def build_policy_observation_space() -> spaces.Box:
    return spaces.Box(
        low=np.float32(-1_000_000.0),
        high=np.float32(1_000_000.0),
        shape=(POLICY_OBSERVATION_SIZE,),
        dtype=np.float32,
    )


def build_policy_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete(POLICY_ACTION_NVECS)


def encode_observation_for_policy(observation: dict[str, object]) -> np.ndarray:
    validate_observation_contract(observation)

    player = _mapping(observation["player"])
    player_tile = _mapping(player["tile"])
    prayers = _mapping(player["protection_prayers"])
    lockouts = _mapping(player["lockouts"])
    consumables = _mapping(player["consumables"])
    wave = _mapping(observation["wave"])
    npcs = _sequence(observation["npcs"])

    encoded: list[float] = [
        float(int(observation["schema_version"])),
        float(int(observation["tick"])),
        float(int(observation["episode_seed"])),
        float(int(player_tile["x"])),
        float(int(player_tile["y"])),
        float(int(player_tile["level"])),
        float(int(player["hitpoints_current"])),
        float(int(player["hitpoints_max"])),
        float(int(player["prayer_current"])),
        float(int(player["prayer_max"])),
        float(int(player["run_energy"])),
        float(int(player["run_energy_max"])),
        float(int(player["run_energy_percent"])),
        float(int(bool(player["running"]))),
        float(int(bool(prayers["protect_from_magic"]))),
        float(int(bool(prayers["protect_from_missiles"]))),
        float(int(bool(prayers["protect_from_melee"]))),
        float(int(bool(lockouts["attack_locked"]))),
        float(int(bool(lockouts["food_locked"]))),
        float(int(bool(lockouts["drink_locked"]))),
        float(int(bool(lockouts["combo_locked"]))),
        float(int(bool(lockouts["busy_locked"]))),
        float(int(consumables["shark_count"])),
        float(int(consumables["prayer_potion_dose_count"])),
        float(_categorical_index(POLICY_AMMO_ID_TO_INDEX, str(consumables["ammo_id"]), "ammo_id")),
        float(int(consumables["ammo_count"])),
        float(int(wave["wave"])),
        float(int(wave["rotation"])),
        float(int(wave["remaining"])),
        float(len(npcs)),
    ]

    for slot_index in range(POLICY_MAX_VISIBLE_NPCS):
        if slot_index >= len(npcs):
            encoded.extend((0.0,) * len(POLICY_NPC_SLOT_FIELD_ORDER))
            continue

        npc = _mapping(npcs[slot_index])
        npc_tile = _mapping(npc["tile"])
        encoded.extend(
            (
                1.0,
                float(int(npc["visible_index"])),
                float(int(npc["npc_index"])),
                float(_categorical_index(POLICY_NPC_ID_TO_INDEX, str(npc["id"]), "npc_id")),
                float(int(npc_tile["x"])),
                float(int(npc_tile["y"])),
                float(int(npc_tile["level"])),
                float(int(npc["hitpoints_current"])),
                float(int(npc["hitpoints_max"])),
                float(int(bool(npc["hidden"]))),
                float(int(bool(npc["dead"]))),
                float(int(bool(npc["under_attack"]))),
                float(int(npc.get("jad_telegraph_state", 0))),
            )
        )

    array = np.asarray(encoded, dtype=np.float32)
    if array.shape != (POLICY_OBSERVATION_SIZE,):
        raise ValueError(
            f"Policy observation layout drift: expected {(POLICY_OBSERVATION_SIZE,)}, got {array.shape}."
        )
    return array


def encode_action_for_policy(action: int | str | dict[str, object] | NormalizedAction) -> np.ndarray:
    normalized = normalize_action(action)
    prayer_index = 0 if normalized.prayer is None else POLICY_PRAYER_TO_INDEX[normalized.prayer]
    tile_x = 0 if normalized.tile is None else normalized.tile.x
    tile_y = 0 if normalized.tile is None else normalized.tile.y
    tile_level = 0 if normalized.tile is None else normalized.tile.level
    visible_npc_index = 0 if normalized.visible_npc_index is None else normalized.visible_npc_index

    encoded = np.asarray(
        (
            normalized.action_id,
            tile_x,
            tile_y,
            tile_level,
            visible_npc_index,
            prayer_index,
        ),
        dtype=np.int64,
    )
    _validate_action_vector(encoded)
    return encoded


def decode_action_from_policy(action: Sequence[int] | np.ndarray) -> NormalizedAction:
    vector = np.asarray(action, dtype=np.int64).reshape(-1)
    _validate_action_vector(vector)
    prayer = POLICY_INDEX_TO_PRAYER[int(vector[5])]
    payload: dict[str, object] = {"action_id": int(vector[0])}
    if int(vector[0]) == 1:
        payload["tile"] = {
            "x": int(vector[1]),
            "y": int(vector[2]),
            "level": int(vector[3]),
        }
    elif int(vector[0]) == 2:
        payload["visible_npc_index"] = int(vector[4])
    elif int(vector[0]) == 3:
        payload["prayer"] = prayer
    return normalize_action(payload)


def _validate_action_vector(vector: np.ndarray) -> None:
    if vector.shape != (len(POLICY_ACTION_HEAD_ORDER),):
        raise ValueError(
            "Policy action layout drift: "
            f"expected {(len(POLICY_ACTION_HEAD_ORDER),)}, got {vector.shape}."
        )
    for index, limit in enumerate(POLICY_ACTION_NVECS):
        value = int(vector[index])
        if value < 0 or value >= int(limit):
            raise ValueError(
                f"Action head {POLICY_ACTION_HEAD_ORDER[index]!r} out of bounds: {value} not in [0, {int(limit)})."
            )


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"Expected dict, got {type(value)!r}")
    return value


def _sequence(value: object) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"Expected sequence, got {type(value)!r}")
    return value


def _categorical_index(mapping: dict[str, int], value: str, label: str) -> int:
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported {label} for {PUFFER_POLICY_OBSERVATION_SCHEMA.contract_id}: {value!r}"
        ) from exc
