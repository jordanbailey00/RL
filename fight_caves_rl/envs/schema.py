from dataclasses import dataclass


@dataclass(frozen=True)
class VersionedContract:
    contract_id: str
    version: int
    compatibility_policy: str


@dataclass(frozen=True)
class ActionDefinition:
    action_id: int
    name: str
    source_name: str
    parameters: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeStartContract:
    identity: VersionedContract
    seed_key: str
    start_wave_min: int
    start_wave_max: int
    default_start_wave: int
    default_ammo: int
    default_prayer_potions: int
    default_sharks: int
    fixed_levels: tuple[tuple[str, int], ...]
    reset_clocks: tuple[str, ...]
    reset_variables: tuple[str, ...]
    equipment: tuple[str, ...]
    inventory_item_ids: tuple[str, ...]
    run_energy_percent: int
    run_toggle_on: bool
    xp_gain_blocked: bool
    start_wave_invocation: str


@dataclass(frozen=True)
class BridgeContract:
    identity: VersionedContract
    mode_a_transport: str
    mode_a_runtime_entrypoint: str
    mode_a_player_provisioning: tuple[str, ...]
    mode_b_transport_target: str
    mode_c_transport_target: str
    sim_artifact_task: str
    sim_artifact_fallback_task: str
    sim_distribution_glob: str
    sim_headless_jar_name: str
    requires_sim_workspace_checkout: bool
    required_sim_workspace_paths: tuple[str, ...]
    required_handshake_fields: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkProfileContract:
    identity: VersionedContract
    benchmark_mode: str
    bridge_mode: str
    reward_config_id: str
    curriculum_config_id: str
    replay_mode: str
    logging_mode: str
    dashboard_mode: str
    env_count_ladder: tuple[int, ...]
    required_manifest_fields: tuple[str, ...]


HEADLESS_OBSERVATION_SCHEMA = VersionedContract(
    contract_id="headless_observation_v1",
    version=1,
    compatibility_policy="v1_additive_only",
)
HEADLESS_OBSERVATION_COMPATIBILITY_POLICY = HEADLESS_OBSERVATION_SCHEMA.compatibility_policy

HEADLESS_OBSERVATION_TOP_LEVEL_FIELDS = (
    "schema_id",
    "schema_version",
    "compatibility_policy",
    "tick",
    "episode_seed",
    "player",
    "wave",
    "npcs",
)

HEADLESS_ACTION_SCHEMA = VersionedContract(
    contract_id="headless_action_v1",
    version=1,
    compatibility_policy="append_only_ids",
)
HEADLESS_ACTION_COMPATIBILITY_POLICY = HEADLESS_ACTION_SCHEMA.compatibility_policy

HEADLESS_ACTION_DEFINITIONS = (
    ActionDefinition(0, "wait", "Wait", ()),
    ActionDefinition(1, "walk_to_tile", "WalkToTile", ("tile.x", "tile.y", "tile.level")),
    ActionDefinition(2, "attack_visible_npc", "AttackVisibleNpc", ("visible_npc_index",)),
    ActionDefinition(3, "toggle_protection_prayer", "ToggleProtectionPrayer", ("prayer",)),
    ActionDefinition(4, "eat_shark", "EatShark", ()),
    ActionDefinition(5, "drink_prayer_potion", "DrinkPrayerPotion", ()),
    ActionDefinition(6, "toggle_run", "ToggleRun", ()),
)

HEADLESS_PROTECTION_PRAYER_IDS = (
    "protect_from_magic",
    "protect_from_missiles",
    "protect_from_melee",
)

HEADLESS_ACTION_REJECT_REASONS = (
    "AlreadyActedThisTick",
    "InvalidTargetIndex",
    "TargetNotVisible",
    "PlayerBusy",
    "MissingConsumable",
    "ConsumptionLocked",
    "PrayerPointsDepleted",
    "InsufficientRunEnergy",
    "NoMovementRequired",
)

FIGHT_CAVE_EPISODE_START_CONTRACT = EpisodeStartContract(
    identity=VersionedContract(
        contract_id="fight_cave_episode_start_v1",
        version=1,
        compatibility_policy="sim_aligned_replace_on_change",
    ),
    seed_key="episode_seed",
    start_wave_min=1,
    start_wave_max=63,
    default_start_wave=1,
    default_ammo=1000,
    default_prayer_potions=8,
    default_sharks=20,
    fixed_levels=(
        ("Attack", 1),
        ("Strength", 1),
        ("Defence", 70),
        ("Constitution", 700),
        ("Ranged", 70),
        ("Prayer", 43),
        ("Magic", 1),
    ),
    reset_clocks=(
        "delay",
        "movement_delay",
        "food_delay",
        "drink_delay",
        "combo_delay",
        "fight_cave_cooldown",
    ),
    reset_variables=(
        "fight_cave_wave",
        "fight_cave_rotation",
        "fight_cave_remaining",
        "fight_cave_start_time",
        "healed",
    ),
    equipment=(
        "coif",
        "rune_crossbow",
        "black_dragonhide_body",
        "black_dragonhide_chaps",
        "black_dragonhide_vambraces",
        "snakeskin_boots",
        "adamant_bolts",
    ),
    inventory_item_ids=("prayer_potion_4", "shark"),
    run_energy_percent=100,
    run_toggle_on=True,
    xp_gain_blocked=True,
    start_wave_invocation="fightCave.startWave(player, startWave, start = false)",
)

FIGHT_CAVES_BRIDGE_CONTRACT = BridgeContract(
    identity=VersionedContract(
        contract_id="fight_caves_bridge_v1",
        version=1,
        compatibility_policy="bump_on_transport_or_handshake_change",
    ),
    mode_a_transport="embedded_jvm_direct_runtime",
    mode_a_runtime_entrypoint="HeadlessMain.bootstrap",
    mode_a_player_provisioning=(
        "AccountManager.setup",
        "AccountManager.spawn",
        'player["creation"] = -1',
        'player["skip_level_up"] = true',
        "player.viewport.loaded = true",
    ),
    mode_b_transport_target="subprocess_binary_batch_ipc",
    mode_c_transport_target="shared_buffer_vector_backend",
    sim_artifact_task=":game:headlessDistZip",
    sim_artifact_fallback_task=":game:packageHeadless",
    sim_distribution_glob="game/build/distributions/fight-caves-headless*.zip",
    sim_headless_jar_name="fight-caves-headless.jar",
    requires_sim_workspace_checkout=True,
    required_sim_workspace_paths=(
        "FCspec.md",
        "config/headless_data_allowlist.toml",
        "config/headless_manifest.toml",
        "config/headless_scripts.txt",
        "data/cache/main_file_cache.dat2",
    ),
    required_handshake_fields=(
        "observation_schema_id",
        "observation_schema_version",
        "action_schema_id",
        "action_schema_version",
        "episode_start_contract_id",
        "episode_start_contract_version",
        "bridge_protocol_id",
        "bridge_protocol_version",
        "benchmark_profile_id",
        "benchmark_profile_version",
        "sim_artifact_task",
        "sim_artifact_path",
        "pufferlib_distribution",
        "pufferlib_version",
    ),
)

OFFICIAL_BENCHMARK_PROFILE = BenchmarkProfileContract(
    identity=VersionedContract(
        contract_id="official_profile_v0",
        version=0,
        compatibility_policy="manual_increment_on_profile_change",
    ),
    benchmark_mode="staged_gate_matrix",
    bridge_mode="recorded_per_run",
    reward_config_id="reward_sparse_v0",
    curriculum_config_id="curriculum_disabled_v0",
    replay_mode="disabled",
    logging_mode="standard",
    dashboard_mode="disabled",
    env_count_ladder=(1, 16, 64, 256, 1024),
    required_manifest_fields=(
        "benchmark_profile_id",
        "benchmark_profile_version",
        "hardware_profile",
        "rl_commit_sha",
        "sim_commit_sha",
        "sim_artifact_task",
        "sim_artifact_path",
        "bridge_protocol_id",
        "bridge_protocol_version",
        "observation_schema_id",
        "observation_schema_version",
        "action_schema_id",
        "action_schema_version",
        "episode_start_contract_id",
        "episode_start_contract_version",
        "pufferlib_distribution",
        "pufferlib_version",
        "reward_config_id",
        "curriculum_config_id",
        "replay_mode",
        "logging_mode",
        "dashboard_mode",
        "env_count",
    ),
)

PUFFER_POLICY_OBSERVATION_SCHEMA = VersionedContract(
    contract_id="puffer_policy_observation_v1",
    version=1,
    compatibility_policy="replace_on_layout_change",
)

PUFFER_POLICY_ACTION_SCHEMA = VersionedContract(
    contract_id="puffer_policy_action_v0",
    version=0,
    compatibility_policy="replace_on_layout_change",
)
