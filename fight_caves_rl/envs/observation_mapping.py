from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from fight_caves_rl.envs.schema import (
    HEADLESS_OBSERVATION_COMPATIBILITY_POLICY,
    HEADLESS_OBSERVATION_SCHEMA,
    HEADLESS_OBSERVATION_TOP_LEVEL_FIELDS,
)


def validate_observation_contract(observation: Mapping[str, Any]) -> None:
    expected_keys = list(HEADLESS_OBSERVATION_TOP_LEVEL_FIELDS)
    if "debug_future_leakage" in observation:
        expected_keys.append("debug_future_leakage")
    actual_keys = list(observation.keys())
    if actual_keys != expected_keys:
        raise ValueError(
            "Observation field order drift detected. "
            f"Expected {expected_keys}, got {actual_keys}."
        )
    if observation.get("schema_id") != HEADLESS_OBSERVATION_SCHEMA.contract_id:
        raise ValueError(
            "Observation schema id mismatch: "
            f"{observation.get('schema_id')!r}"
        )
    if int(observation.get("schema_version", -1)) != HEADLESS_OBSERVATION_SCHEMA.version:
        raise ValueError(
            "Observation schema version mismatch: "
            f"{observation.get('schema_version')!r}"
        )
    if observation.get("compatibility_policy") != HEADLESS_OBSERVATION_COMPATIBILITY_POLICY:
        raise ValueError(
            "Observation compatibility policy mismatch: "
            f"{observation.get('compatibility_policy')!r}"
        )


def flatten_observation(observation: Mapping[str, Any]) -> tuple[Any, ...]:
    validate_observation_contract(observation)

    player = _mapping(observation["player"])
    wave = _mapping(observation["wave"])
    tile = _mapping(player["tile"])
    prayers = _mapping(player["protection_prayers"])
    lockouts = _mapping(player["lockouts"])
    consumables = _mapping(player["consumables"])

    flattened: list[Any] = [
        observation["schema_id"],
        int(observation["schema_version"]),
        observation["compatibility_policy"],
        int(observation["tick"]),
        int(observation["episode_seed"]),
        int(tile["x"]),
        int(tile["y"]),
        int(tile["level"]),
        int(player["hitpoints_current"]),
        int(player["hitpoints_max"]),
        int(player["prayer_current"]),
        int(player["prayer_max"]),
        int(player["run_energy"]),
        int(player["run_energy_max"]),
        int(player["run_energy_percent"]),
        int(bool(player["running"])),
        int(bool(prayers["protect_from_magic"])),
        int(bool(prayers["protect_from_missiles"])),
        int(bool(prayers["protect_from_melee"])),
        int(bool(lockouts["attack_locked"])),
        int(bool(lockouts["food_locked"])),
        int(bool(lockouts["drink_locked"])),
        int(bool(lockouts["combo_locked"])),
        int(bool(lockouts["busy_locked"])),
        int(consumables["shark_count"]),
        int(consumables["prayer_potion_dose_count"]),
        str(consumables["ammo_id"]),
        int(consumables["ammo_count"]),
        int(wave["wave"]),
        int(wave["rotation"]),
        int(wave["remaining"]),
        len(_sequence(observation["npcs"])),
    ]

    for npc in _sequence(observation["npcs"]):
        npc_mapping = _mapping(npc)
        npc_tile = _mapping(npc_mapping["tile"])
        flattened.extend(
            (
                int(npc_mapping["visible_index"]),
                int(npc_mapping["npc_index"]),
                str(npc_mapping["id"]),
                int(npc_tile["x"]),
                int(npc_tile["y"]),
                int(npc_tile["level"]),
                int(npc_mapping["hitpoints_current"]),
                int(npc_mapping["hitpoints_max"]),
                int(bool(npc_mapping["hidden"])),
                int(bool(npc_mapping["dead"])),
                int(bool(npc_mapping["under_attack"])),
                int(npc_mapping.get("jad_telegraph_state", 0)),
            )
        )

    if "debug_future_leakage" in observation:
        debug = _mapping(observation["debug_future_leakage"])
        flattened.append(int(bool(debug["enabled"])))
        fields = tuple(str(field) for field in _sequence(debug["fields"]))
        flattened.append(fields)

    return tuple(flattened)


def visible_targets_from_observation(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_observation_contract(observation)
    targets: list[dict[str, Any]] = []
    for npc in _sequence(observation["npcs"]):
        npc_mapping = _mapping(npc)
        targets.append(
            {
                "visible_index": int(npc_mapping["visible_index"]),
                "npc_index": int(npc_mapping["npc_index"]),
                "id": str(npc_mapping["id"]),
                "tile": dict(_mapping(npc_mapping["tile"])),
            }
        )
    return targets


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Expected mapping, got {type(value)!r}")
    return value


def _sequence(value: Any) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"Expected sequence, got {type(value)!r}")
    return value
