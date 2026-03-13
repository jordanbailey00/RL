from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from fight_caves_rl.envs.schema import VersionedContract


MECHANICS_PARITY_TRACE_SCHEMA = VersionedContract(
    contract_id="fight_caves_mechanics_parity_trace_v1",
    version=1,
    compatibility_policy="append_only_fields",
)


@dataclass(frozen=True)
class ParityTraceFieldDefinition:
    name: str
    dtype: str
    description: str


MECHANICS_PARITY_TRACE_FIELDS = (
    ParityTraceFieldDefinition("tick_index", "int32", "Episode-relative tick index."),
    ParityTraceFieldDefinition("action_name", "string", "Submitted action intent identifier."),
    ParityTraceFieldDefinition(
        "action_accepted",
        "bool",
        "Whether the runtime accepted the submitted action for this tick.",
    ),
    ParityTraceFieldDefinition(
        "rejection_code",
        "uint8",
        "Stable rejection code emitted when the action is not accepted.",
    ),
    ParityTraceFieldDefinition("player_hitpoints", "int32", "Player hitpoints after the step."),
    ParityTraceFieldDefinition("player_prayer_points", "int32", "Player prayer points after the step."),
    ParityTraceFieldDefinition("run_enabled", "bool", "Current player run toggle state."),
    ParityTraceFieldDefinition("inventory_ammo", "int32", "Remaining equipped ammo count."),
    ParityTraceFieldDefinition("inventory_sharks", "int32", "Remaining shark count."),
    ParityTraceFieldDefinition(
        "inventory_prayer_potions",
        "int32",
        "Remaining prayer potion dose count.",
    ),
    ParityTraceFieldDefinition("wave_id", "int32", "Current Fight Caves wave identifier."),
    ParityTraceFieldDefinition("remaining_npcs", "int32", "Remaining alive NPC count."),
    ParityTraceFieldDefinition(
        "visible_target_order",
        "int32[]",
        "Stable visible-target ordering emitted for target selection parity.",
    ),
    ParityTraceFieldDefinition(
        "visible_npc_type",
        "string[]",
        "Visible NPC type ids in the shared target ordering.",
    ),
    ParityTraceFieldDefinition(
        "visible_npc_hitpoints",
        "int32[]",
        "Visible NPC hitpoints in the shared target ordering.",
    ),
    ParityTraceFieldDefinition(
        "visible_npc_alive",
        "bool[]",
        "Visible NPC alive flags in the shared target ordering.",
    ),
    ParityTraceFieldDefinition(
        "jad_telegraph_state",
        "uint8",
        "Jad telegraph state emitted by the mechanics contract.",
    ),
    ParityTraceFieldDefinition(
        "jad_hit_resolve_outcome",
        "string",
        "Outcome recorded at Jad hit resolution.",
    ),
    ParityTraceFieldDefinition("damage_dealt", "float32", "Damage dealt on this step."),
    ParityTraceFieldDefinition("damage_taken", "float32", "Damage taken on this step."),
    ParityTraceFieldDefinition("terminal_code", "uint8", "Stable terminal code for the step."),
)

MECHANICS_PARITY_TRACE_FIELD_NAMES = tuple(
    field.name for field in MECHANICS_PARITY_TRACE_FIELDS
)

_PARITY_FIELD_BY_NAME = {
    field.name: field for field in MECHANICS_PARITY_TRACE_FIELDS
}


def coerce_mechanics_parity_trace_record(record: Mapping[str, Any]) -> dict[str, Any]:
    missing = [name for name in MECHANICS_PARITY_TRACE_FIELD_NAMES if name not in record]
    extra = [name for name in record if name not in _PARITY_FIELD_BY_NAME]
    if missing or extra:
        raise ValueError(
            "Mechanics parity trace shape drifted: "
            f"missing={missing!r} extra={extra!r}."
        )

    return {
        field.name: _coerce_trace_value(field.dtype, record[field.name])
        for field in MECHANICS_PARITY_TRACE_FIELDS
    }


def coerce_mechanics_parity_trace_records(
    records: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    return tuple(coerce_mechanics_parity_trace_record(record) for record in records)


def mechanics_parity_trace_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = coerce_mechanics_parity_trace_records(records)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _coerce_trace_value(dtype: str, value: Any) -> Any:
    if dtype == "bool":
        return bool(value)
    if dtype in {"int32", "uint8"}:
        return int(value)
    if dtype == "float32":
        return float(value)
    if dtype == "string":
        return str(value)
    if dtype == "int32[]":
        return [int(item) for item in value]
    if dtype == "string[]":
        return [str(item) for item in value]
    if dtype == "bool[]":
        return [bool(item) for item in value]
    raise ValueError(f"Unsupported mechanics parity dtype: {dtype!r}.")
