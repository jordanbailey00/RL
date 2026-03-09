from __future__ import annotations

import pytest

from fight_caves_rl.bridge.contracts import HeadlessEpisodeConfig, HeadlessPlayerConfig
from fight_caves_rl.bridge.debug_client import HeadlessDebugClient
from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths
from fight_caves_rl.envs.puffer_encoding import encode_observation_for_policy


def test_live_flat_observation_matches_current_raw_policy_projection():
    _require_live_runtime()

    client = HeadlessDebugClient.create()
    try:
        player = client.create_player_slot(
            HeadlessPlayerConfig(account_name="flat_projection_live")
        )
        client.reset_episode(player, HeadlessEpisodeConfig(seed=41_001, start_wave=1))

        raw = client.observe(player)
        flat = client.observe_flat(player)

        expected = encode_observation_for_policy(raw)
        assert flat.shape == expected.shape
        assert flat.dtype == expected.dtype
        assert (flat == expected).all()
    finally:
        client.close()


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
