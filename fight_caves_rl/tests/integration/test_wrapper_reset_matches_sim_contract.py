import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths
from fight_caves_rl.envs.correctness_env import FightCavesCorrectnessEnv
from fight_caves_rl.envs.schema import FIGHT_CAVE_EPISODE_START_CONTRACT


def test_wrapper_reset_matches_sim_contract():
    _require_live_runtime()

    env = FightCavesCorrectnessEnv()
    try:
        observation, info = env.reset(seed=123456789)
    finally:
        env.close()

    assert observation["episode_seed"] == 123456789
    assert observation["player"]["run_energy_percent"] == 100
    assert observation["player"]["running"] is True
    assert observation["player"]["consumables"]["shark_count"] == FIGHT_CAVE_EPISODE_START_CONTRACT.default_sharks
    assert observation["player"]["consumables"]["prayer_potion_dose_count"] == (
        FIGHT_CAVE_EPISODE_START_CONTRACT.default_prayer_potions * 4
    )
    assert info["episode_state"]["seed"] == 123456789


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
