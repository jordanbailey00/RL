from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from fight_caves_rl.bridge.contracts import BridgeHandshake, HeadlessEpisodeConfig
from fight_caves_rl.bridge.errors import BridgeContractError
from fight_caves_rl.envs.action_mapping import NormalizedAction, normalize_action
from fight_caves_rl.envs.schema import (
    FIGHT_CAVE_EPISODE_START_CONTRACT,
    FIGHT_CAVES_BRIDGE_CONTRACT,
    HEADLESS_ACTION_SCHEMA,
    HEADLESS_OBSERVATION_SCHEMA,
)

BATCH_TRANSPORT_MODE = "embedded_jvm_lockstep_batch"
BATCH_TICK_POLICY = "apply_all_then_shared_tick"
BATCH_SLOT_ISOLATION = "fight_cave_dynamic_instance"
BATCH_OBSERVATION_SOURCE = "observeFightCave"
BATCH_VISIBLE_TARGET_SOURCE = "observation.npcs"
BATCH_TICK_STRIDE = 1


@dataclass(frozen=True)
class BatchBridgeProtocol:
    bridge_protocol_id: str
    bridge_protocol_version: int
    transport_mode: str
    tick_policy: str
    slot_isolation: str
    observation_source: str
    visible_target_source: str
    tick_stride: int
    observation_schema_id: str
    observation_schema_version: int
    action_schema_id: str
    action_schema_version: int
    episode_start_contract_id: str
    episode_start_contract_version: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BatchResetSpec:
    slot_index: int
    episode: HeadlessEpisodeConfig


@dataclass(frozen=True)
class BatchResetRequest:
    protocol: BatchBridgeProtocol
    resets: tuple[BatchResetSpec, ...]


@dataclass(frozen=True)
class BatchActionSpec:
    slot_index: int
    action: NormalizedAction


@dataclass(frozen=True)
class BatchStepRequest:
    protocol: BatchBridgeProtocol
    actions: tuple[BatchActionSpec, ...]
    ticks_after: int = BATCH_TICK_STRIDE


@dataclass(frozen=True)
class BatchSlotResetResult:
    slot_index: int
    observation: dict[str, Any]
    info: dict[str, Any]


@dataclass(frozen=True)
class BatchResetResponse:
    protocol: BatchBridgeProtocol
    results: tuple[BatchSlotResetResult, ...]
    elapsed_nanos: int

    @property
    def env_count(self) -> int:
        return len(self.results)


@dataclass(frozen=True)
class BatchSlotStepResult:
    slot_index: int
    action: NormalizedAction
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


@dataclass(frozen=True)
class BatchStepResponse:
    protocol: BatchBridgeProtocol
    results: tuple[BatchSlotStepResult, ...]
    elapsed_nanos: int

    @property
    def env_count(self) -> int:
        return len(self.results)

    @property
    def env_steps_per_second(self) -> float:
        if self.elapsed_nanos <= 0 or not self.results:
            return 0.0
        return len(self.results) * 1_000_000_000.0 / float(self.elapsed_nanos)


def build_batch_protocol(handshake: BridgeHandshake) -> BatchBridgeProtocol:
    values = handshake.values
    _expect_field(values, "bridge_protocol_id", FIGHT_CAVES_BRIDGE_CONTRACT.identity.contract_id)
    _expect_field(values, "bridge_protocol_version", FIGHT_CAVES_BRIDGE_CONTRACT.identity.version)
    _expect_field(values, "observation_schema_id", HEADLESS_OBSERVATION_SCHEMA.contract_id)
    _expect_field(values, "observation_schema_version", HEADLESS_OBSERVATION_SCHEMA.version)
    _expect_field(values, "action_schema_id", HEADLESS_ACTION_SCHEMA.contract_id)
    _expect_field(values, "action_schema_version", HEADLESS_ACTION_SCHEMA.version)
    _expect_field(
        values,
        "episode_start_contract_id",
        FIGHT_CAVE_EPISODE_START_CONTRACT.identity.contract_id,
    )
    _expect_field(
        values,
        "episode_start_contract_version",
        FIGHT_CAVE_EPISODE_START_CONTRACT.identity.version,
    )
    return BatchBridgeProtocol(
        bridge_protocol_id=str(values["bridge_protocol_id"]),
        bridge_protocol_version=int(values["bridge_protocol_version"]),
        transport_mode=BATCH_TRANSPORT_MODE,
        tick_policy=BATCH_TICK_POLICY,
        slot_isolation=BATCH_SLOT_ISOLATION,
        observation_source=BATCH_OBSERVATION_SOURCE,
        visible_target_source=BATCH_VISIBLE_TARGET_SOURCE,
        tick_stride=BATCH_TICK_STRIDE,
        observation_schema_id=str(values["observation_schema_id"]),
        observation_schema_version=int(values["observation_schema_version"]),
        action_schema_id=str(values["action_schema_id"]),
        action_schema_version=int(values["action_schema_version"]),
        episode_start_contract_id=str(values["episode_start_contract_id"]),
        episode_start_contract_version=int(values["episode_start_contract_version"]),
    )


def build_batch_reset_request(
    protocol: BatchBridgeProtocol,
    *,
    slot_indices: Sequence[int],
    episodes: Sequence[HeadlessEpisodeConfig],
) -> BatchResetRequest:
    if len(slot_indices) != len(episodes):
        raise BridgeContractError(
            "Batch reset request slot/episode length mismatch: "
            f"{len(slot_indices)} != {len(episodes)}"
        )
    resets = tuple(
        BatchResetSpec(slot_index=int(slot_index), episode=episode)
        for slot_index, episode in zip(slot_indices, episodes, strict=True)
    )
    _validate_unique_slots(spec.slot_index for spec in resets)
    return BatchResetRequest(protocol=protocol, resets=resets)


def build_batch_step_request(
    protocol: BatchBridgeProtocol,
    *,
    slot_indices: Sequence[int],
    actions: Sequence[int | str | Mapping[str, object] | NormalizedAction],
    ticks_after: int = BATCH_TICK_STRIDE,
) -> BatchStepRequest:
    if len(slot_indices) != len(actions):
        raise BridgeContractError(
            "Batch step request slot/action length mismatch: "
            f"{len(slot_indices)} != {len(actions)}"
        )
    if int(ticks_after) != BATCH_TICK_STRIDE:
        raise BridgeContractError(
            "PR7 batch bridge only supports a shared ticks_after stride of "
            f"{BATCH_TICK_STRIDE}, got {ticks_after}."
        )
    action_specs = tuple(
        BatchActionSpec(
            slot_index=int(slot_index),
            action=normalize_action(action),
        )
        for slot_index, action in zip(slot_indices, actions, strict=True)
    )
    _validate_unique_slots(spec.slot_index for spec in action_specs)
    return BatchStepRequest(
        protocol=protocol,
        actions=action_specs,
        ticks_after=int(ticks_after),
    )


def _expect_field(values: Mapping[str, Any], key: str, expected: object) -> None:
    actual = values.get(key)
    if actual != expected:
        raise BridgeContractError(
            f"Bridge handshake {key!r} mismatch: expected {expected!r}, got {actual!r}."
        )


def _validate_unique_slots(slot_indices: Sequence[int]) -> None:
    seen: set[int] = set()
    for slot_index in slot_indices:
        if int(slot_index) in seen:
            raise BridgeContractError(
                f"Duplicate slot index in batch request: {slot_index}."
            )
        seen.add(int(slot_index))
