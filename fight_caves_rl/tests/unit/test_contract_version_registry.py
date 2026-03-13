from fight_caves_rl.contracts import (
    FIGHT_CAVES_V2_MECHANICS_CONTRACT,
    MECHANICS_PARITY_INVARIANTS,
    MECHANICS_PARITY_TRACE_FIELDS,
    MECHANICS_PARITY_TRACE_SCHEMA,
    REWARD_FEATURE_NAMES,
    REWARD_FEATURE_SCHEMA,
    TERMINAL_CODE_DEFINITIONS,
    TERMINAL_CODE_SCHEMA,
    TerminalCode,
)
from fight_caves_rl.envs.schema import (
    FIGHT_CAVE_EPISODE_START_CONTRACT,
    FIGHT_CAVES_BRIDGE_CONTRACT,
    HEADLESS_ACTION_DEFINITIONS,
    HEADLESS_ACTION_REJECT_REASONS,
    HEADLESS_ACTION_SCHEMA,
    HEADLESS_OBSERVATION_SCHEMA,
    HEADLESS_OBSERVATION_TOP_LEVEL_FIELDS,
    HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA,
    OFFICIAL_BENCHMARK_PROFILE,
    PUFFER_POLICY_OBSERVATION_SCHEMA,
)
from fight_caves_rl.manifests.versions import (
    PUFFERLIB_BASELINE_DISTRIBUTION,
    PUFFERLIB_BASELINE_VERSION,
)


def test_contract_registry_exports_expected_core_versions():
    assert HEADLESS_OBSERVATION_SCHEMA.contract_id == "headless_observation_v1"
    assert HEADLESS_OBSERVATION_SCHEMA.version == 1
    assert HEADLESS_ACTION_SCHEMA.contract_id == "headless_action_v1"
    assert HEADLESS_ACTION_SCHEMA.version == 1
    assert FIGHT_CAVE_EPISODE_START_CONTRACT.identity.contract_id == "fight_cave_episode_start_v1"
    assert FIGHT_CAVE_EPISODE_START_CONTRACT.identity.version == 1
    assert FIGHT_CAVES_BRIDGE_CONTRACT.identity.contract_id == "fight_caves_bridge_v2"
    assert FIGHT_CAVES_BRIDGE_CONTRACT.identity.version == 2
    assert OFFICIAL_BENCHMARK_PROFILE.identity.contract_id == "official_profile_v0"
    assert OFFICIAL_BENCHMARK_PROFILE.identity.version == 0
    assert HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.identity.contract_id == "headless_training_flat_observation_v1"
    assert HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.identity.version == 1
    assert PUFFER_POLICY_OBSERVATION_SCHEMA.contract_id == "puffer_policy_observation_v1"
    assert PUFFER_POLICY_OBSERVATION_SCHEMA.version == 1
    assert TERMINAL_CODE_SCHEMA.contract_id == "fight_caves_v2_terminal_codes_v1"
    assert TERMINAL_CODE_SCHEMA.version == 1
    assert REWARD_FEATURE_SCHEMA.contract_id == "fight_caves_v2_reward_features_v1"
    assert REWARD_FEATURE_SCHEMA.version == 1
    assert MECHANICS_PARITY_TRACE_SCHEMA.contract_id == "fight_caves_mechanics_parity_trace_v1"
    assert MECHANICS_PARITY_TRACE_SCHEMA.version == 1
    assert FIGHT_CAVES_V2_MECHANICS_CONTRACT.identity.contract_id == "fight_caves_v2_mechanics_v1"
    assert FIGHT_CAVES_V2_MECHANICS_CONTRACT.identity.version == 1


def test_contract_registry_keeps_expected_schema_shapes():
    assert HEADLESS_OBSERVATION_TOP_LEVEL_FIELDS == (
        "schema_id",
        "schema_version",
        "compatibility_policy",
        "tick",
        "episode_seed",
        "player",
        "wave",
        "npcs",
    )
    assert [action.action_id for action in HEADLESS_ACTION_DEFINITIONS] == list(range(7))
    assert HEADLESS_ACTION_DEFINITIONS[2].name == "attack_visible_npc"
    assert "AlreadyActedThisTick" in HEADLESS_ACTION_REJECT_REASONS
    assert HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.dtype == "float32"
    assert HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.feature_count == 134
    assert [definition.code for definition in TERMINAL_CODE_DEFINITIONS] == list(TerminalCode)
    assert REWARD_FEATURE_NAMES == (
        "damage_dealt",
        "damage_taken",
        "npc_kill",
        "wave_clear",
        "jad_damage_dealt",
        "jad_kill",
        "player_death",
        "cave_complete",
        "food_used",
        "prayer_potion_used",
        "correct_jad_prayer_on_resolve",
        "wrong_jad_prayer_on_resolve",
        "invalid_action",
        "movement_progress",
        "idle_penalty_flag",
        "tick_penalty_base",
    )
    assert [field.name for field in MECHANICS_PARITY_TRACE_FIELDS] == [
        "tick_index",
        "action_name",
        "action_accepted",
        "rejection_code",
        "player_hitpoints",
        "player_prayer_points",
        "run_enabled",
        "inventory_ammo",
        "inventory_sharks",
        "inventory_prayer_potions",
        "wave_id",
        "remaining_npcs",
        "visible_target_order",
        "visible_npc_type",
        "visible_npc_hitpoints",
        "visible_npc_alive",
        "jad_telegraph_state",
        "jad_hit_resolve_outcome",
        "damage_dealt",
        "damage_taken",
        "terminal_code",
    ]
    assert MECHANICS_PARITY_INVARIANTS == (
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


def test_v2_mechanics_contract_links_current_action_and_reset_contracts():
    assert FIGHT_CAVES_V2_MECHANICS_CONTRACT.action_schema_id == HEADLESS_ACTION_SCHEMA.contract_id
    assert FIGHT_CAVES_V2_MECHANICS_CONTRACT.action_schema_version == HEADLESS_ACTION_SCHEMA.version
    assert (
        FIGHT_CAVES_V2_MECHANICS_CONTRACT.reset_contract_id
        == FIGHT_CAVE_EPISODE_START_CONTRACT.identity.contract_id
    )
    assert (
        FIGHT_CAVES_V2_MECHANICS_CONTRACT.reset_contract_version
        == FIGHT_CAVE_EPISODE_START_CONTRACT.identity.version
    )
    assert "mechanics parity" in FIGHT_CAVES_V2_MECHANICS_CONTRACT.parity_definition
    assert (
        FIGHT_CAVES_V2_MECHANICS_CONTRACT.v2_runtime_surface_contract_id
        == "fight_caves_fast_kernel_surface_v1"
    )
    assert FIGHT_CAVES_V2_MECHANICS_CONTRACT.v2_runtime_surface_contract_version == 1
    assert "version independently" in FIGHT_CAVES_V2_MECHANICS_CONTRACT.v2_runtime_surface_policy
    assert "RL-facing action, reset, terminal-code, reward-feature, and parity-trace contracts stay shared" in (
        FIGHT_CAVES_V2_MECHANICS_CONTRACT.v2_runtime_surface_policy
    )
    assert "native or C-backed kernel" in FIGHT_CAVES_V2_MECHANICS_CONTRACT.portable_kernel_goal


def test_pufferlib_baseline_metadata_stays_aligned_with_contract_work():
    assert PUFFERLIB_BASELINE_DISTRIBUTION == "pufferlib-core"
    assert PUFFERLIB_BASELINE_VERSION == "3.0.17"
