import pytest

from fight_caves_rl.bridge.contracts import HeadlessEpisodeConfig, HeadlessPlayerConfig
from fight_caves_rl.bridge.debug_client import HeadlessDebugClient
from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths
from fight_caves_rl.envs.correctness_env import FightCavesCorrectnessEnv


def test_wrapper_step_matches_raw_sim_trace():
    _require_live_runtime()

    wrapper = FightCavesCorrectnessEnv()
    raw = HeadlessDebugClient.create()
    raw_player = raw.create_player_slot(HeadlessPlayerConfig(account_name="rl_raw_trace"))
    try:
        wrapper_observation, _ = wrapper.reset(seed=987654321)
        raw.reset_episode(raw_player, seed_config(987654321))
        raw_observation = raw.observe(raw_player)
        assert wrapper_observation == raw_observation

        wrapper_next, _, _, _, wrapper_info = wrapper.step(0)
        raw_next = raw.step_once(raw_player, 0)
        assert wrapper_next == raw_next.observation
        assert wrapper_info["action_result"] == raw_next.action_result
    finally:
        wrapper.close()
        raw.close()


def seed_config(seed: int) -> HeadlessEpisodeConfig:
    return HeadlessEpisodeConfig(seed=seed)


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
