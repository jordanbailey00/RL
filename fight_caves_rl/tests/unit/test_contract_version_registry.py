from fight_caves_rl.envs.schema import (
    FIGHT_CAVE_EPISODE_START_CONTRACT,
    FIGHT_CAVES_BRIDGE_CONTRACT,
    HEADLESS_ACTION_DEFINITIONS,
    HEADLESS_ACTION_REJECT_REASONS,
    HEADLESS_ACTION_SCHEMA,
    HEADLESS_OBSERVATION_SCHEMA,
    HEADLESS_OBSERVATION_TOP_LEVEL_FIELDS,
    OFFICIAL_BENCHMARK_PROFILE,
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
    assert FIGHT_CAVES_BRIDGE_CONTRACT.identity.contract_id == "fight_caves_bridge_v0"
    assert FIGHT_CAVES_BRIDGE_CONTRACT.identity.version == 0
    assert OFFICIAL_BENCHMARK_PROFILE.identity.contract_id == "official_profile_v0"
    assert OFFICIAL_BENCHMARK_PROFILE.identity.version == 0


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


def test_pufferlib_baseline_metadata_stays_aligned_with_contract_work():
    assert PUFFERLIB_BASELINE_DISTRIBUTION == "pufferlib-core"
    assert PUFFERLIB_BASELINE_VERSION == "3.0.17"
