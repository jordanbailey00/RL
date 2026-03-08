from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fight_caves_rl.envs.schema import (
    HEADLESS_ACTION_DEFINITIONS,
    HEADLESS_ACTION_SCHEMA,
    HEADLESS_PROTECTION_PRAYER_IDS,
)

ACTION_DEFINITION_BY_ID = {
    action.action_id: action for action in HEADLESS_ACTION_DEFINITIONS
}
ACTION_ID_BY_NAME = {
    action.name: action.action_id for action in HEADLESS_ACTION_DEFINITIONS
}


@dataclass(frozen=True)
class TileCoordinates:
    x: int
    y: int
    level: int = 0


@dataclass(frozen=True)
class NormalizedAction:
    action_id: int
    name: str
    tile: TileCoordinates | None = None
    visible_npc_index: int | None = None
    prayer: str | None = None


def normalize_action(action: int | str | Mapping[str, object] | NormalizedAction) -> NormalizedAction:
    if isinstance(action, NormalizedAction):
        return _validate_action(action)

    if isinstance(action, int):
        return _validate_action(
            NormalizedAction(
                action_id=action,
                name=_action_name(action),
            )
        )

    if isinstance(action, str):
        action_id = ACTION_ID_BY_NAME[action]
        return _validate_action(NormalizedAction(action_id=action_id, name=action))

    if not isinstance(action, Mapping):
        raise TypeError(f"Unsupported action payload type: {type(action)!r}")

    action_id = _resolve_action_id(action)
    tile = _parse_tile(action.get("tile"), action)
    normalized = NormalizedAction(
        action_id=action_id,
        name=_action_name(action_id),
        tile=tile,
        visible_npc_index=_optional_int(action.get("visible_npc_index")),
        prayer=_optional_str(action.get("prayer")),
    )
    return _validate_action(normalized)


def _resolve_action_id(action: Mapping[str, object]) -> int:
    if "action_id" in action:
        return int(action["action_id"])
    if "name" in action:
        name = str(action["name"])
        if name not in ACTION_ID_BY_NAME:
            raise ValueError(f"Unknown action name: {name!r}")
        return ACTION_ID_BY_NAME[name]
    raise ValueError("Action mapping must contain `action_id` or `name`.")


def _parse_tile(tile: object, action: Mapping[str, object]) -> TileCoordinates | None:
    if tile is None and ("x" in action or "y" in action or "level" in action):
        tile = action
    if tile is None:
        return None
    if not isinstance(tile, Mapping):
        raise TypeError("Walk actions must provide `tile` as a mapping.")
    return TileCoordinates(
        x=int(tile["x"]),
        y=int(tile["y"]),
        level=int(tile.get("level", 0)),
    )


def _validate_action(action: NormalizedAction) -> NormalizedAction:
    definition = ACTION_DEFINITION_BY_ID.get(action.action_id)
    if definition is None:
        raise ValueError(
            f"Action id {action.action_id} is not part of "
            f"{HEADLESS_ACTION_SCHEMA.contract_id}."
        )
    if action.name != definition.name:
        raise ValueError(
            f"Action id/name mismatch: {action.action_id} != {action.name!r}."
        )

    if action.action_id == 1 and action.tile is None:
        raise ValueError("walk_to_tile requires tile coordinates.")
    if action.action_id == 2 and action.visible_npc_index is None:
        raise ValueError("attack_visible_npc requires visible_npc_index.")
    if action.action_id == 3:
        if action.prayer is None:
            raise ValueError("toggle_protection_prayer requires a prayer id.")
        if action.prayer not in HEADLESS_PROTECTION_PRAYER_IDS:
            raise ValueError(f"Unsupported protection prayer: {action.prayer!r}.")
    if action.action_id not in {1} and action.tile is not None:
        raise ValueError(f"{action.name} does not accept tile coordinates.")
    if action.action_id not in {2} and action.visible_npc_index is not None:
        raise ValueError(f"{action.name} does not accept visible_npc_index.")
    if action.action_id not in {3} and action.prayer is not None:
        raise ValueError(f"{action.name} does not accept prayer.")
    return action


def _action_name(action_id: int) -> str:
    definition = ACTION_DEFINITION_BY_ID.get(action_id)
    if definition is None:
        raise ValueError(f"Unknown action id: {action_id}")
    return definition.name


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)
