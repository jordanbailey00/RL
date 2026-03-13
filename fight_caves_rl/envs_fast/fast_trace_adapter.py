from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fight_caves_rl.contracts.parity_trace_schema import (
    coerce_mechanics_parity_trace_record,
)
from fight_caves_rl.envs_fast.fast_vector_env import _jget


def adapt_fast_parity_trace(trace: Any) -> dict[str, Any]:
    return coerce_mechanics_parity_trace_record(
        {
            "tick_index": int(_jget(trace, "tickIndex")),
            "action_name": str(_jget(trace, "actionName")),
            "action_accepted": bool(_jget(trace, "actionAccepted")),
            "rejection_code": int(_jget(trace, "rejectionCode")),
            "player_hitpoints": int(_jget(trace, "playerHitpoints")),
            "player_prayer_points": int(_jget(trace, "playerPrayerPoints")),
            "run_enabled": bool(_jget(trace, "runEnabled")),
            "inventory_ammo": int(_jget(trace, "inventoryAmmo")),
            "inventory_sharks": int(_jget(trace, "inventorySharks")),
            "inventory_prayer_potions": int(_jget(trace, "inventoryPrayerPotions")),
            "wave_id": int(_jget(trace, "waveId")),
            "remaining_npcs": int(_jget(trace, "remainingNpcs")),
            "visible_target_order": [int(value) for value in _jget(trace, "visibleTargetOrder")],
            "visible_npc_type": [str(value) for value in _jget(trace, "visibleNpcType")],
            "visible_npc_hitpoints": [int(value) for value in _jget(trace, "visibleNpcHitpoints")],
            "visible_npc_alive": [bool(value) for value in _jget(trace, "visibleNpcAlive")],
            "jad_telegraph_state": int(_jget(trace, "jadTelegraphState")),
            "jad_hit_resolve_outcome": str(_jget(trace, "jadHitResolveOutcome")),
            "damage_dealt": float(_jget(trace, "damageDealt")),
            "damage_taken": float(_jget(trace, "damageTaken")),
            "terminal_code": int(_jget(trace, "terminalCode")),
        }
    )


def adapt_fast_parity_traces(traces: Iterable[Any]) -> tuple[dict[str, Any], ...]:
    return tuple(adapt_fast_parity_trace(trace) for trace in traces)


def extract_fast_parity_traces(response: Any) -> tuple[dict[str, Any], ...]:
    return adapt_fast_parity_traces(_jget(response, "parityTraces"))
