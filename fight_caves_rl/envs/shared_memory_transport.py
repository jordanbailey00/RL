from __future__ import annotations

from dataclasses import dataclass
import mmap
import os
import tempfile
from typing import Any

import numpy as np

PIPE_PICKLE_TRANSPORT_MODE = "pipe_pickle_v1"
SHARED_MEMORY_TRANSPORT_MODE = "shared_memory_v1"
SUBPROCESS_TRANSPORT_MODES = (
    PIPE_PICKLE_TRANSPORT_MODE,
    SHARED_MEMORY_TRANSPORT_MODE,
)
DEFAULT_RESPONSE_SLOT_COUNT = 2


@dataclass(frozen=True)
class SharedNdArraySpec:
    path: str
    shape: tuple[int, ...]
    dtype: str

    def to_payload(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "shape": list(self.shape),
            "dtype": str(self.dtype),
        }

    @classmethod
    def from_payload(cls, payload: "SharedNdArraySpec | dict[str, Any]") -> "SharedNdArraySpec":
        if isinstance(payload, cls):
            return payload
        return cls(
            path=str(payload["path"]),
            shape=tuple(int(value) for value in payload["shape"]),
            dtype=str(payload["dtype"]),
        )


class SharedNdArray:
    def __init__(
        self,
        *,
        path: str,
        file_handle: Any,
        mapping: mmap.mmap,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
        owner: bool,
    ) -> None:
        self._path = str(path)
        self._file_handle = file_handle
        self._mapping = mapping
        self._shape = tuple(int(value) for value in shape)
        self._dtype = np.dtype(dtype)
        self._owner = bool(owner)
        self.array = np.ndarray(self._shape, dtype=self._dtype, buffer=self._mapping)

    @classmethod
    def create(
        cls,
        *,
        shape: tuple[int, ...],
        dtype: np.dtype[Any] | str,
    ) -> "SharedNdArray":
        resolved_dtype = np.dtype(dtype)
        size_bytes = int(np.prod(shape, dtype=np.int64)) * resolved_dtype.itemsize
        if size_bytes <= 0:
            raise ValueError(f"Shared array size must be > 0, got {size_bytes}.")
        root = os.environ.get("TMPDIR", tempfile.gettempdir())
        fd, path = tempfile.mkstemp(prefix="fc_rl_shm_", dir=root)
        try:
            os.ftruncate(fd, size_bytes)
            file_handle = os.fdopen(fd, "r+b", buffering=0)
        except Exception:
            os.close(fd)
            os.unlink(path)
            raise
        mapping = mmap.mmap(file_handle.fileno(), length=size_bytes, access=mmap.ACCESS_WRITE)
        return cls(
            path=path,
            file_handle=file_handle,
            mapping=mapping,
            shape=shape,
            dtype=resolved_dtype,
            owner=True,
        )

    @classmethod
    def attach(cls, spec: SharedNdArraySpec) -> "SharedNdArray":
        file_handle = open(spec.path, "r+b", buffering=0)
        size_bytes = int(np.prod(spec.shape, dtype=np.int64)) * np.dtype(spec.dtype).itemsize
        mapping = mmap.mmap(file_handle.fileno(), length=size_bytes, access=mmap.ACCESS_WRITE)
        return cls(
            path=spec.path,
            file_handle=file_handle,
            mapping=mapping,
            shape=spec.shape,
            dtype=np.dtype(spec.dtype),
            owner=False,
        )

    def spec(self) -> SharedNdArraySpec:
        return SharedNdArraySpec(
            path=self._path,
            shape=self._shape,
            dtype=self._dtype.str,
        )

    def close(self, *, unlink: bool = False) -> None:
        try:
            self._mapping.close()
        finally:
            self._file_handle.close()
        if unlink and self._owner:
            try:
                os.unlink(self._path)
            except FileNotFoundError:
                pass


@dataclass(frozen=True)
class SharedTransitionSlotSpec:
    observations: SharedNdArraySpec
    rewards: SharedNdArraySpec
    terminals: SharedNdArraySpec
    truncations: SharedNdArraySpec
    teacher_actions: SharedNdArraySpec
    agent_ids: SharedNdArraySpec
    masks: SharedNdArraySpec

    def to_payload(self) -> dict[str, Any]:
        return {
            "observations": self.observations.to_payload(),
            "rewards": self.rewards.to_payload(),
            "terminals": self.terminals.to_payload(),
            "truncations": self.truncations.to_payload(),
            "teacher_actions": self.teacher_actions.to_payload(),
            "agent_ids": self.agent_ids.to_payload(),
            "masks": self.masks.to_payload(),
        }

    @classmethod
    def from_payload(
        cls,
        payload: "SharedTransitionSlotSpec | dict[str, Any]",
    ) -> "SharedTransitionSlotSpec":
        if isinstance(payload, cls):
            return payload
        return cls(
            observations=SharedNdArraySpec.from_payload(payload["observations"]),
            rewards=SharedNdArraySpec.from_payload(payload["rewards"]),
            terminals=SharedNdArraySpec.from_payload(payload["terminals"]),
            truncations=SharedNdArraySpec.from_payload(payload["truncations"]),
            teacher_actions=SharedNdArraySpec.from_payload(payload["teacher_actions"]),
            agent_ids=SharedNdArraySpec.from_payload(payload["agent_ids"]),
            masks=SharedNdArraySpec.from_payload(payload["masks"]),
        )


class SharedTransitionSlot:
    def __init__(
        self,
        *,
        observations: SharedNdArray,
        rewards: SharedNdArray,
        terminals: SharedNdArray,
        truncations: SharedNdArray,
        teacher_actions: SharedNdArray,
        agent_ids: SharedNdArray,
        masks: SharedNdArray,
    ) -> None:
        self.observations = observations
        self.rewards = rewards
        self.terminals = terminals
        self.truncations = truncations
        self.teacher_actions = teacher_actions
        self.agent_ids = agent_ids
        self.masks = masks

    @classmethod
    def create(
        cls,
        *,
        env_count: int,
        observation_dim: int,
    ) -> "SharedTransitionSlot":
        return cls(
            observations=SharedNdArray.create(
                shape=(int(env_count), int(observation_dim)),
                dtype=np.float32,
            ),
            rewards=SharedNdArray.create(shape=(int(env_count),), dtype=np.float32),
            terminals=SharedNdArray.create(shape=(int(env_count),), dtype=np.bool_),
            truncations=SharedNdArray.create(shape=(int(env_count),), dtype=np.bool_),
            teacher_actions=SharedNdArray.create(shape=(int(env_count),), dtype=np.int32),
            agent_ids=SharedNdArray.create(shape=(int(env_count),), dtype=np.int64),
            masks=SharedNdArray.create(shape=(int(env_count),), dtype=np.bool_),
        )

    @classmethod
    def attach(cls, spec: SharedTransitionSlotSpec) -> "SharedTransitionSlot":
        return cls(
            observations=SharedNdArray.attach(spec.observations),
            rewards=SharedNdArray.attach(spec.rewards),
            terminals=SharedNdArray.attach(spec.terminals),
            truncations=SharedNdArray.attach(spec.truncations),
            teacher_actions=SharedNdArray.attach(spec.teacher_actions),
            agent_ids=SharedNdArray.attach(spec.agent_ids),
            masks=SharedNdArray.attach(spec.masks),
        )

    def spec(self) -> SharedTransitionSlotSpec:
        return SharedTransitionSlotSpec(
            observations=self.observations.spec(),
            rewards=self.rewards.spec(),
            terminals=self.terminals.spec(),
            truncations=self.truncations.spec(),
            teacher_actions=self.teacher_actions.spec(),
            agent_ids=self.agent_ids.spec(),
            masks=self.masks.spec(),
        )

    def write_transition(self, transition: tuple[Any, ...]) -> None:
        observations, rewards, terminals, truncations, teacher_actions, _infos, agent_ids, masks = transition
        np.copyto(self.observations.array, np.asarray(observations, dtype=np.float32), casting="no")
        np.copyto(self.rewards.array, np.asarray(rewards, dtype=np.float32), casting="no")
        np.copyto(self.terminals.array, np.asarray(terminals, dtype=np.bool_), casting="no")
        np.copyto(self.truncations.array, np.asarray(truncations, dtype=np.bool_), casting="no")
        np.copyto(
            self.teacher_actions.array,
            np.asarray(teacher_actions, dtype=np.int32),
            casting="unsafe",
        )
        np.copyto(self.agent_ids.array, np.asarray(agent_ids, dtype=np.int64), casting="unsafe")
        np.copyto(self.masks.array, np.asarray(masks, dtype=np.bool_), casting="no")

    def materialize_transition(self, *, infos: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "observations": self.observations.array,
            "rewards": self.rewards.array,
            "terminals": self.terminals.array,
            "truncations": self.truncations.array,
            "teacher_actions": self.teacher_actions.array,
            "infos": list(infos),
            "agent_ids": self.agent_ids.array,
            "masks": self.masks.array,
        }

    def close(self, *, unlink: bool = False) -> None:
        self.observations.close(unlink=unlink)
        self.rewards.close(unlink=unlink)
        self.terminals.close(unlink=unlink)
        self.truncations.close(unlink=unlink)
        self.teacher_actions.close(unlink=unlink)
        self.agent_ids.close(unlink=unlink)
        self.masks.close(unlink=unlink)


@dataclass(frozen=True)
class SharedMemoryTransportSpec:
    action: SharedNdArraySpec
    responses: tuple[SharedTransitionSlotSpec, ...]
    response_slot_count: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "action": self.action.to_payload(),
            "responses": [entry.to_payload() for entry in self.responses],
            "response_slot_count": int(self.response_slot_count),
        }

    @classmethod
    def from_payload(
        cls,
        payload: "SharedMemoryTransportSpec | dict[str, Any]",
    ) -> "SharedMemoryTransportSpec":
        if isinstance(payload, cls):
            return payload
        return cls(
            action=SharedNdArraySpec.from_payload(payload["action"]),
            responses=tuple(
                SharedTransitionSlotSpec.from_payload(entry) for entry in payload["responses"]
            ),
            response_slot_count=int(payload["response_slot_count"]),
        )


class SharedMemoryTransportParent:
    def __init__(
        self,
        *,
        env_count: int,
        action_dim: int,
        observation_dim: int,
        response_slot_count: int = DEFAULT_RESPONSE_SLOT_COUNT,
    ) -> None:
        self.action = SharedNdArray.create(
            shape=(int(env_count), int(action_dim)),
            dtype=np.int32,
        )
        self.responses = tuple(
            SharedTransitionSlot.create(
                env_count=int(env_count),
                observation_dim=int(observation_dim),
            )
            for _ in range(int(response_slot_count))
        )

    def spec(self) -> SharedMemoryTransportSpec:
        return SharedMemoryTransportSpec(
            action=self.action.spec(),
            responses=tuple(slot.spec() for slot in self.responses),
            response_slot_count=len(self.responses),
        )

    def write_actions(self, actions: np.ndarray) -> None:
        np.copyto(self.action.array, np.asarray(actions, dtype=np.int32), casting="unsafe")

    def materialize_transition(self, payload: dict[str, Any]) -> dict[str, Any]:
        slot = self.responses[int(payload["buffer_index"])]
        return slot.materialize_transition(infos=list(payload["infos"]))

    def close(self, *, unlink: bool = False) -> None:
        self.action.close(unlink=unlink)
        for slot in self.responses:
            slot.close(unlink=unlink)


class SharedMemoryTransportWorker:
    def __init__(self, *, action: SharedNdArray, responses: tuple[SharedTransitionSlot, ...]) -> None:
        self.action = action
        self.responses = tuple(responses)
        self._next_response_slot = 0

    @classmethod
    def attach(cls, spec: SharedMemoryTransportSpec | dict[str, Any]) -> "SharedMemoryTransportWorker":
        spec = SharedMemoryTransportSpec.from_payload(spec)
        return cls(
            action=SharedNdArray.attach(spec.action),
            responses=tuple(SharedTransitionSlot.attach(entry) for entry in spec.responses),
        )

    def read_actions(self) -> np.ndarray:
        return self.action.array

    def publish_transition(self, transition: tuple[Any, ...]) -> dict[str, Any]:
        slot_index = self._next_response_slot
        self._next_response_slot = (self._next_response_slot + 1) % len(self.responses)
        slot = self.responses[slot_index]
        slot.write_transition(transition)
        return {
            "buffer_index": slot_index,
            "infos": list(transition[5]),
            "transport_mode": SHARED_MEMORY_TRANSPORT_MODE,
        }

    def close(self, *, unlink: bool = False) -> None:
        self.action.close(unlink=unlink)
        for slot in self.responses:
            slot.close(unlink=unlink)
