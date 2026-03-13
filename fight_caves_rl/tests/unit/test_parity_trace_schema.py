from __future__ import annotations

from fight_caves_rl.contracts.parity_trace_schema import (
    MECHANICS_PARITY_TRACE_FIELD_NAMES,
    coerce_mechanics_parity_trace_record,
    mechanics_parity_trace_digest,
)


def _sample_trace_record() -> dict[str, object]:
    return {
        "tick_index": "0",
        "action_name": "reset",
        "action_accepted": 1,
        "rejection_code": "0",
        "player_hitpoints": "700",
        "player_prayer_points": "43",
        "run_enabled": 1,
        "inventory_ammo": "1000",
        "inventory_sharks": "20",
        "inventory_prayer_potions": "8",
        "wave_id": "1",
        "remaining_npcs": "2",
        "visible_target_order": ("11", "12"),
        "visible_npc_type": ("tz_kih", "tz_kih"),
        "visible_npc_hitpoints": ("10", "10"),
        "visible_npc_alive": (1, 1),
        "jad_telegraph_state": "0",
        "jad_hit_resolve_outcome": "none",
        "damage_dealt": "0.0",
        "damage_taken": 0,
        "terminal_code": "0",
    }


def test_coerce_mechanics_parity_trace_record_enforces_field_order_and_types():
    record = coerce_mechanics_parity_trace_record(_sample_trace_record())

    assert tuple(record.keys()) == MECHANICS_PARITY_TRACE_FIELD_NAMES
    assert record["tick_index"] == 0
    assert record["action_accepted"] is True
    assert record["visible_target_order"] == [11, 12]
    assert record["visible_npc_alive"] == [True, True]
    assert record["damage_dealt"] == 0.0


def test_mechanics_parity_trace_digest_is_stable_across_input_key_order():
    record = _sample_trace_record()
    reordered = dict(reversed(tuple(record.items())))

    assert mechanics_parity_trace_digest((record,)) == mechanics_parity_trace_digest((reordered,))
