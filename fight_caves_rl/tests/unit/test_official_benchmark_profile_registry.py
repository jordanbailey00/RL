import yaml

from fight_caves_rl.envs.schema import FIGHT_CAVES_BRIDGE_CONTRACT, OFFICIAL_BENCHMARK_PROFILE
from fight_caves_rl.utils.paths import repo_root


def test_official_benchmark_profile_config_matches_registry():
    path = repo_root() / "configs/benchmark/official_profile_v0.yaml"
    payload = yaml.safe_load(path.read_text())

    assert payload["profile_id"] == OFFICIAL_BENCHMARK_PROFILE.identity.contract_id
    assert payload["profile_version"] == OFFICIAL_BENCHMARK_PROFILE.identity.version
    assert payload["benchmark_mode"] == OFFICIAL_BENCHMARK_PROFILE.benchmark_mode
    assert payload["bridge_mode"] == OFFICIAL_BENCHMARK_PROFILE.bridge_mode
    assert payload["reward_config"] == OFFICIAL_BENCHMARK_PROFILE.reward_config_id
    assert payload["curriculum_config"] == OFFICIAL_BENCHMARK_PROFILE.curriculum_config_id
    assert payload["replay_mode"] == OFFICIAL_BENCHMARK_PROFILE.replay_mode
    assert payload["logging_mode"] == OFFICIAL_BENCHMARK_PROFILE.logging_mode
    assert payload["dashboard_mode"] == OFFICIAL_BENCHMARK_PROFILE.dashboard_mode
    assert tuple(payload["env_count_ladder"]) == OFFICIAL_BENCHMARK_PROFILE.env_count_ladder
    assert tuple(payload["required_manifest_fields"]) == OFFICIAL_BENCHMARK_PROFILE.required_manifest_fields
def test_bridge_contract_registry_keeps_expected_artifact_defaults():
    assert FIGHT_CAVES_BRIDGE_CONTRACT.sim_artifact_task == ":game:headlessDistZip"
    assert FIGHT_CAVES_BRIDGE_CONTRACT.sim_artifact_fallback_task == ":game:packageHeadless"
    assert FIGHT_CAVES_BRIDGE_CONTRACT.sim_distribution_relative_path == (
        "game/build/distributions/fight-caves-headless.zip"
    )
    assert FIGHT_CAVES_BRIDGE_CONTRACT.sim_headless_jar_name == "fight-caves-headless.jar"
