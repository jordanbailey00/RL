from fight_caves_rl.bridge.contracts import BridgeHandshake
from fight_caves_rl.bridge.errors import BridgeContractError
from fight_caves_rl.bridge.launcher import build_bridge_handshake, discover_headless_runtime_paths
from fight_caves_rl.bridge.protocol import build_batch_protocol


def test_batch_protocol_fails_fast_on_schema_or_version_drift():
    handshake = build_bridge_handshake(discover_headless_runtime_paths())
    mutated = BridgeHandshake(
        values={
            **dict(handshake.values),
            "flat_observation_schema_version": int(handshake.values["flat_observation_schema_version"]) + 1,
        }
    )

    try:
        build_batch_protocol(mutated)
    except BridgeContractError:
        return
    raise AssertionError("Expected batch protocol validation to fail on bridge version drift.")
