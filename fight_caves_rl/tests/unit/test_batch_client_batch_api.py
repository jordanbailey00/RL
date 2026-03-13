import numpy as np

from fight_caves_rl.bridge.batch_client import BatchClientConfig, HeadlessBatchClient
from fight_caves_rl.bridge.contracts import BridgeHandshake, HeadlessBootstrapConfig
from fight_caves_rl.envs.schema import (
    FIGHT_CAVE_EPISODE_START_CONTRACT,
    FIGHT_CAVES_BRIDGE_CONTRACT,
    HEADLESS_ACTION_SCHEMA,
    HEADLESS_OBSERVATION_SCHEMA,
    HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA,
)
from fight_caves_rl.envs.shared_memory_transport import INFO_PAYLOAD_MODE_MINIMAL


class _FakeDebugClient:
    def __init__(self) -> None:
        self.handshake = BridgeHandshake(
            values={
                "bridge_protocol_id": FIGHT_CAVES_BRIDGE_CONTRACT.identity.contract_id,
                "bridge_protocol_version": FIGHT_CAVES_BRIDGE_CONTRACT.identity.version,
                "observation_schema_id": HEADLESS_OBSERVATION_SCHEMA.contract_id,
                "observation_schema_version": HEADLESS_OBSERVATION_SCHEMA.version,
                "observation_path_mode": "flat",
                "flat_observation_schema_id": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.identity.contract_id,
                "flat_observation_schema_version": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.identity.version,
                "flat_observation_dtype": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.dtype,
                "flat_observation_feature_count": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.feature_count,
                "flat_observation_max_visible_npcs": HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.max_visible_npcs,
                "action_schema_id": HEADLESS_ACTION_SCHEMA.contract_id,
                "action_schema_version": HEADLESS_ACTION_SCHEMA.version,
                "episode_start_contract_id": FIGHT_CAVE_EPISODE_START_CONTRACT.identity.contract_id,
                "episode_start_contract_version": FIGHT_CAVE_EPISODE_START_CONTRACT.identity.version,
                "batch_apply_actions_api": "applyActionsBatch",
                "batch_observe_flat_api": "observeFlatBatch",
                "batch_flat_observation_layout": "env_major_contiguous_float32",
            }
        )
        self._ticks_by_player: dict[str, int] = {}
        self.apply_action_jvm_calls = 0
        self.apply_actions_batch_calls = 0
        self.observe_flat_calls = 0
        self.observe_flat_batch_calls = 0

    def create_player_slot(self, config):
        player = str(config.account_name)
        self._ticks_by_player[player] = 100
        return player

    def reset_episode(self, player, config):
        self._ticks_by_player[player] = 100 + int(config.seed % 10)
        return {
            "seed": int(config.seed),
            "wave": int(config.start_wave),
            "rotation": 0,
            "remaining": 63,
            "instance_id": int(config.seed),
            "player_tile": {"x": 0, "y": 0, "level": 0},
        }

    def observe_flat(self, player):
        self.observe_flat_calls += 1
        return self._flat_row(player)

    def observe_flat_batch(self, players):
        self.observe_flat_batch_calls += 1
        return np.stack([self._flat_row(player) for player in players], axis=0)

    def observe(self, player, include_future_leakage=False):
        raise AssertionError("Minimal-info batch API test should not use raw observe.")

    def observe_jvm(self, player, include_future_leakage=False):
        raise AssertionError("Minimal-info batch API test should not use raw observe.")

    def apply_action_jvm(self, player, action):
        self.apply_action_jvm_calls += 1
        raise AssertionError("Batch apply capability should bypass per-slot apply_action_jvm.")

    def apply_actions_batch(self, players, actions):
        self.apply_actions_batch_calls += 1
        return [
            {
                "action_type": "IDLE",
                "action_id": int(getattr(action, "action_id", action.action_id)),
                "action_applied": True,
                "rejection_reason": None,
                "metadata": {},
            }
            for action in actions
        ]

    def tick(self, times=1):
        for player in self._ticks_by_player:
            self._ticks_by_player[player] += int(times)

    def close(self):
        return None

    def _flat_row(self, player: str) -> np.ndarray:
        row = np.zeros(
            (HEADLESS_TRAINING_FLAT_OBSERVATION_SCHEMA.feature_count,),
            dtype=np.float32,
        )
        row[0] = float(HEADLESS_OBSERVATION_SCHEMA.version)
        row[1] = float(self._ticks_by_player[player])
        row[26] = 1.0
        row[28] = 63.0
        row[29] = 0.0
        return row


def test_batch_client_uses_batch_apply_and_flat_observe_capabilities(monkeypatch):
    fake = _FakeDebugClient()
    monkeypatch.setattr(
        "fight_caves_rl.bridge.batch_client.HeadlessDebugClient.create",
        lambda bootstrap: fake,
    )

    client = HeadlessBatchClient.create(
        BatchClientConfig(
            env_count=2,
            bootstrap=HeadlessBootstrapConfig(),
            info_payload_mode=INFO_PAYLOAD_MODE_MINIMAL,
            instrumentation_enabled=True,
        )
    )
    try:
        reset = client.reset_batch(seeds=[101, 202])
        assert len(reset.results) == 2
        assert fake.observe_flat_batch_calls == 1
        assert fake.observe_flat_calls == 0

        response = client.step_batch(actions=[0, 0])
        assert len(response.results) == 2
        assert fake.apply_actions_batch_calls == 1
        assert fake.apply_action_jvm_calls == 0
        assert fake.observe_flat_batch_calls == 2

        snapshot = client.instrumentation_snapshot()
        assert snapshot["client_apply_actions_batch_jvm"]["calls"] == 1
        assert snapshot["client_apply_action_jvm"]["calls"] == 2
        assert snapshot["client_flat_observe_batch"]["calls"] == 1
        assert snapshot["client_flat_observe"]["calls"] == 2
        assert snapshot["client_reset_observe_flat_batch"]["calls"] == 1
        assert snapshot["client_reset_observe_flat"]["calls"] == 2
    finally:
        client.close()
