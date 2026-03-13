from __future__ import annotations

from dataclasses import dataclass

from fight_caves_rl.contracts.parity_trace_schema import MECHANICS_PARITY_TRACE_SCHEMA
from fight_caves_rl.contracts.reward_feature_schema import REWARD_FEATURE_SCHEMA
from fight_caves_rl.contracts.terminal_codes import TERMINAL_CODE_SCHEMA
from fight_caves_rl.envs.schema import (
    FIGHT_CAVE_EPISODE_START_CONTRACT,
    HEADLESS_ACTION_SCHEMA,
    VersionedContract,
)


MECHANICS_PARITY_INVARIANTS = (
    "tick_cadence",
    "action_meanings",
    "action_rejection_rules",
    "attack_timings",
    "prayer_timings",
    "jad_telegraph_semantics",
    "consumable_rules",
    "movement_and_run_rules",
    "wave_progression",
    "episode_reset_contract",
    "terminal_outcomes",
)


@dataclass(frozen=True)
class MechanicsContract:
    identity: VersionedContract
    oracle_role: str
    headed_demo_role: str
    trainer_role: str
    parity_definition: str
    action_schema_id: str
    action_schema_version: int
    reset_contract_id: str
    reset_contract_version: int
    terminal_code_schema_id: str
    terminal_code_schema_version: int
    reward_feature_schema_id: str
    reward_feature_schema_version: int
    parity_trace_schema_id: str
    parity_trace_schema_version: int
    v2_runtime_surface_contract_id: str
    v2_runtime_surface_contract_version: int
    v2_runtime_surface_policy: str
    portable_kernel_goal: str
    invariants: tuple[str, ...]


FIGHT_CAVES_V2_MECHANICS_CONTRACT = MechanicsContract(
    identity=VersionedContract(
        contract_id="fight_caves_v2_mechanics_v1",
        version=1,
        compatibility_policy="replace_on_semantic_change",
    ),
    oracle_role="Current simulator-backed V1 oracle/reference/debug path.",
    headed_demo_role=(
        "RSPS-backed headed demo/replay target sharing the same mechanics contract, "
        "with fight-caves-demo-lite frozen as fallback/reference only."
    ),
    trainer_role="V2 fast headless training path with a portable RL-facing interface.",
    parity_definition=(
        "mechanics parity between the V2 fast trainer and the RSPS-backed headed/oracle path, "
        "not full engine/runtime parity"
    ),
    action_schema_id=HEADLESS_ACTION_SCHEMA.contract_id,
    action_schema_version=HEADLESS_ACTION_SCHEMA.version,
    reset_contract_id=FIGHT_CAVE_EPISODE_START_CONTRACT.identity.contract_id,
    reset_contract_version=FIGHT_CAVE_EPISODE_START_CONTRACT.identity.version,
    terminal_code_schema_id=TERMINAL_CODE_SCHEMA.contract_id,
    terminal_code_schema_version=TERMINAL_CODE_SCHEMA.version,
    reward_feature_schema_id=REWARD_FEATURE_SCHEMA.contract_id,
    reward_feature_schema_version=REWARD_FEATURE_SCHEMA.version,
    parity_trace_schema_id=MECHANICS_PARITY_TRACE_SCHEMA.contract_id,
    parity_trace_schema_version=MECHANICS_PARITY_TRACE_SCHEMA.version,
    v2_runtime_surface_contract_id="fight_caves_fast_kernel_surface_v1",
    v2_runtime_surface_contract_version=1,
    v2_runtime_surface_policy=(
        "V2 runtime/kernel-specific surfaces version independently under the "
        "fight_caves_fast_kernel_surface_v1 family, while the RL-facing action, "
        "reset, terminal-code, reward-feature, and parity-trace contracts stay "
        "shared until their own semantics change."
    ),
    portable_kernel_goal=(
        "Keep the RL-facing action, reset, terminal, reward-feature, and parity-trace "
        "contracts stable so the first Kotlin/JVM kernel can later be replaced by a "
        "native or C-backed kernel without changing the RL interface."
    ),
    invariants=MECHANICS_PARITY_INVARIANTS,
)
