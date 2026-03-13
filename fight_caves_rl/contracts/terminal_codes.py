from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from fight_caves_rl.envs.schema import VersionedContract


TERMINAL_CODE_SCHEMA = VersionedContract(
    contract_id="fight_caves_v2_terminal_codes_v1",
    version=1,
    compatibility_policy="append_only_codes",
)


class TerminalCode(IntEnum):
    NONE = 0
    PLAYER_DEATH = 1
    CAVE_COMPLETE = 2
    TICK_CAP = 3
    INVALID_STATE = 4
    ORACLE_ABORT = 5


@dataclass(frozen=True)
class TerminalCodeDefinition:
    code: TerminalCode
    description: str


TERMINAL_CODE_DEFINITIONS = (
    TerminalCodeDefinition(TerminalCode.NONE, "No terminal condition was emitted for the slot."),
    TerminalCodeDefinition(
        TerminalCode.PLAYER_DEATH,
        "The player died and the episode terminated.",
    ),
    TerminalCodeDefinition(
        TerminalCode.CAVE_COMPLETE,
        "The player completed Fight Caves and emitted the success terminal.",
    ),
    TerminalCodeDefinition(
        TerminalCode.TICK_CAP,
        "The episode reached its configured tick cap and truncated.",
    ),
    TerminalCodeDefinition(
        TerminalCode.INVALID_STATE,
        "The runtime detected a state invariant violation and aborted the slot.",
    ),
    TerminalCodeDefinition(
        TerminalCode.ORACLE_ABORT,
        "Oracle/reference validation aborted the slot before a gameplay terminal.",
    ),
)
