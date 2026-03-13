from __future__ import annotations

from fight_caves_rl.contracts.parity_trace_schema import (
    MECHANICS_PARITY_TRACE_FIELD_NAMES,
)
from fight_caves_rl.envs_fast.fast_trace_adapter import adapt_fast_parity_trace


class _FakeFastParityTrace:
    tickIndex = 4
    actionName = "attack_visible_npc"
    actionAccepted = False
    rejectionCode = 3
    playerHitpoints = 55
    playerPrayerPoints = 12
    runEnabled = True
    inventoryAmmo = 321
    inventorySharks = 9
    inventoryPrayerPotions = 4
    waveId = 31
    remainingNpcs = 2
    visibleTargetOrder = (101, 102)
    visibleNpcType = ("ket_zek", "yt_mej_kot")
    visibleNpcHitpoints = (180, 75)
    visibleNpcAlive = (True, False)
    jadTelegraphState = 0
    jadHitResolveOutcome = "none"
    damageDealt = 6.5
    damageTaken = 0.0
    terminalCode = 0


def test_adapt_fast_parity_trace_matches_shared_schema():
    record = adapt_fast_parity_trace(_FakeFastParityTrace())

    assert tuple(record.keys()) == MECHANICS_PARITY_TRACE_FIELD_NAMES
    assert record["action_name"] == "attack_visible_npc"
    assert record["action_accepted"] is False
    assert record["rejection_code"] == 3
    assert record["visible_npc_type"] == ["ket_zek", "yt_mej_kot"]
    assert record["damage_dealt"] == 6.5
