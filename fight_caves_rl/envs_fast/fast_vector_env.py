from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np
import pufferlib
import pufferlib.vector

from fight_caves_rl.benchmarks.instrumentation import (
    BucketInstrumentation,
    InstrumentationSnapshot,
)
from fight_caves_rl.bridge.contracts import HeadlessBootstrapConfig
from fight_caves_rl.bridge.errors import BridgeError, BridgeJVMStateError
from fight_caves_rl.bridge.launcher import (
    assert_sim_runtime_ready,
    build_headless_settings_overrides,
    discover_headless_runtime_paths,
)
from fight_caves_rl.contracts.mechanics_contract import FIGHT_CAVES_V2_MECHANICS_CONTRACT
from fight_caves_rl.contracts.reward_feature_schema import REWARD_FEATURE_SCHEMA
from fight_caves_rl.contracts.terminal_codes import TERMINAL_CODE_SCHEMA
from fight_caves_rl.envs.shared_memory_transport import (
    INFO_PAYLOAD_MODE_MINIMAL,
    INFO_PAYLOAD_MODES,
)
from fight_caves_rl.envs_fast.fast_policy_encoding import (
    pack_joint_actions,
)
from fight_caves_rl.envs_fast.fast_reward_adapter import FastRewardAdapter
from fight_caves_rl.envs_fast.fast_spaces import (
    FAST_OBSERVATION_FEATURE_COUNT,
    build_fast_action_space,
    build_fast_observation_space,
)
from fight_caves_rl.utils.java_runtime import resolve_jvm_library_path

QUIET_LOGBACK_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "logging" / "headless_quiet_logback.xml"
)


@dataclass
class _FastJVMContext:
    jpype: Any
    classpath: Path
    user_dir: Path
    classes: dict[str, Any]


_FAST_JVM_CONTEXT: _FastJVMContext | None = None
ResetOptionsProvider = Callable[[int, int], Mapping[str, object] | None]


@dataclass(frozen=True)
class FastKernelVecEnvConfig:
    env_count: int
    account_name_prefix: str = "rl_fast_vecenv"
    start_wave: int = 1
    ammo: int = 1000
    prayer_potions: int = 8
    sharks: int = 20
    tick_cap: int = 20_000
    include_future_leakage: bool = False
    info_payload_mode: str = INFO_PAYLOAD_MODE_MINIMAL
    instrumentation_enabled: bool = False
    bootstrap: HeadlessBootstrapConfig = field(default_factory=HeadlessBootstrapConfig)
    reset_options_provider: ResetOptionsProvider | None = None


class FastKernelVecEnv:
    reset = pufferlib.vector.reset
    step = pufferlib.vector.step

    def __init__(
        self,
        config: FastKernelVecEnvConfig,
        *,
        reward_adapter: FastRewardAdapter,
    ) -> None:
        if int(config.env_count) <= 0:
            raise ValueError(f"env_count must be > 0, got {config.env_count}.")
        if str(config.info_payload_mode) not in INFO_PAYLOAD_MODES:
            raise ValueError(
                f"Unsupported info_payload_mode: {config.info_payload_mode!r}. "
                f"Expected one of {INFO_PAYLOAD_MODES!r}."
            )
        if str(config.info_payload_mode) != INFO_PAYLOAD_MODE_MINIMAL:
            raise ValueError(
                "The Phase 4.1 fast vecenv only supports info_payload_mode='minimal'. "
                "It must not reconstruct structured semantics in Python."
            )
        reward_adapter.validate_supported()

        self.config = config
        self._reward_adapter = reward_adapter
        self._instrumentation = (
            BucketInstrumentation() if bool(config.instrumentation_enabled) else None
        )
        self._runtime, self._classes = _create_fast_kernel_runtime(config)
        self._descriptor = self._runtime.describe()
        self._reward_feature_count = int(_jget(self._descriptor, "rewardFeatureCount"))
        self._observation_feature_count = int(_jget(self._descriptor, "flatObservationFeatureCount"))
        self._validate_descriptor()

        self.driver_env = self
        self.agents_per_batch = int(config.env_count)
        self.num_agents = self.agents_per_batch
        self.single_observation_space = build_fast_observation_space()
        self.single_action_space = build_fast_action_space()
        self.action_space = pufferlib.spaces.joint_space(
            self.single_action_space,
            self.agents_per_batch,
        )
        self.observation_space = pufferlib.spaces.joint_space(
            self.single_observation_space,
            self.agents_per_batch,
        )
        self.emulated = {
            "observation_dtype": self.single_observation_space.dtype,
            "emulated_observation_dtype": self.single_observation_space.dtype,
        }
        pufferlib.set_buffers(self)
        self.agent_ids = np.arange(self.num_agents)
        self.initialized = False
        self.flag = pufferlib.vector.RESET
        self.infos: list[dict[str, Any]] = []
        self._seed_base: int | None = None
        self._episodes_started = np.zeros(self.num_agents, dtype=np.int64)
        self._episode_returns = np.zeros(self.num_agents, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_agents, dtype=np.int32)
        self._minimal_infos = tuple({} for _ in range(self.num_agents))

    @property
    def num_envs(self) -> int:
        return self.agents_per_batch

    @property
    def episode_counts(self) -> np.ndarray:
        return self._episodes_started.copy()

    def topology_snapshot(self) -> dict[str, Any]:
        return {
            "backend": "embedded",
            "env_backend": "v2_fast",
            "transport_mode": "embedded_jvm",
            "worker_count": 1,
            "worker_env_counts": [int(self.num_agents)],
            "info_payload_mode": str(self.config.info_payload_mode),
        }

    def instrumentation_snapshot(self) -> InstrumentationSnapshot:
        if self._instrumentation is None:
            return {}
        return self._instrumentation.snapshot()

    def reset_instrumentation(self) -> None:
        if self._instrumentation is not None:
            self._instrumentation.clear()

    def async_reset(self, seed: int | None = None) -> None:
        stage_started = perf_counter()
        self.flag = pufferlib.vector.RECV
        self._seed_base = None if seed is None else int(seed)
        self._episodes_started.fill(0)
        self._episode_returns.fill(0.0)
        self._episode_lengths.fill(0)
        slot_indices = tuple(range(self.num_agents))
        response = self._reset_slots(slot_indices)
        self._apply_reset_response(response)
        self._record_episode_starts(slot_indices)
        self.infos = list(self._minimal_infos)
        self._record_instrumentation("fast_vecenv_async_reset_total", perf_counter() - stage_started)

    def send(self, actions: np.ndarray) -> None:
        stage_started = perf_counter()
        if not actions.flags.contiguous:
            actions = np.ascontiguousarray(actions)
        actions = pufferlib.vector.send_precheck(self, actions)
        ordered_infos: list[dict[str, Any]] = list(self._minimal_infos)

        done_indices = tuple(
            int(index) for index in np.flatnonzero(np.logical_or(self.terminals, self.truncations))
        )
        if done_indices:
            reset_response = self._reset_slots(done_indices)
            self._apply_reset_response(reset_response)
            self._record_episode_starts(done_indices)

        active_indices = tuple(index for index in range(self.num_agents) if index not in done_indices)
        if active_indices:
            step_response = self._step_slots(active_indices, actions[np.asarray(active_indices)])
            self._apply_step_response(step_response, joint_actions=actions)

        self.infos = ordered_infos
        self._record_instrumentation("fast_vecenv_send_total", perf_counter() - stage_started)

    def recv(self):
        stage_started = perf_counter()
        pufferlib.vector.recv_precheck(self)
        transition = (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            self.teacher_actions,
            self.infos,
            self.agent_ids,
            self.masks,
        )
        self._record_instrumentation("fast_vecenv_recv_total", perf_counter() - stage_started)
        return transition

    def notify(self) -> None:
        return None

    def close(self) -> None:
        self._runtime.close()

    def _validate_descriptor(self) -> None:
        contract = _jget(self._descriptor, "contract")
        if str(_jget(contract, "sharedActionSchemaId")) != FIGHT_CAVES_V2_MECHANICS_CONTRACT.action_schema_id:
            raise BridgeError("Fast kernel action schema id drifted from the frozen mechanics contract.")
        if int(_jget(contract, "sharedActionSchemaVersion")) != FIGHT_CAVES_V2_MECHANICS_CONTRACT.action_schema_version:
            raise BridgeError("Fast kernel action schema version drifted from the frozen mechanics contract.")
        if str(_jget(self._descriptor, "rewardFeatureSchemaId")) != REWARD_FEATURE_SCHEMA.contract_id:
            raise BridgeError("Fast kernel reward feature schema id drifted from the frozen mechanics contract.")
        if int(_jget(self._descriptor, "rewardFeatureSchemaVersion")) != REWARD_FEATURE_SCHEMA.version:
            raise BridgeError("Fast kernel reward feature schema version drifted from the frozen mechanics contract.")
        if self._observation_feature_count != FAST_OBSERVATION_FEATURE_COUNT:
            raise BridgeError(
                "Fast kernel flat observation feature count drifted from the RL fast observation space."
            )
        if str(_jget(contract, "sharedTerminalCodeSchemaId")) != TERMINAL_CODE_SCHEMA.contract_id:
            raise BridgeError("Fast kernel terminal code schema id drifted from the frozen mechanics contract.")
        if int(_jget(contract, "sharedTerminalCodeSchemaVersion")) != TERMINAL_CODE_SCHEMA.version:
            raise BridgeError("Fast kernel terminal code schema version drifted from the frozen mechanics contract.")

    def _reset_slots(self, slot_indices: Sequence[int]):
        stage_started = perf_counter()
        seeds = self._allocate_seeds(slot_indices)
        configs = _java_episode_configs(self._classes, self._episode_configs(slot_indices, seeds))
        response = self._runtime.resetBatch(_java_int_array(slot_indices), configs, False, None, None)
        self._record_instrumentation("fast_vecenv_reset_batch_call", perf_counter() - stage_started)
        return response

    def _step_slots(self, slot_indices: Sequence[int], actions: np.ndarray):
        pack_started = perf_counter()
        packed_actions = pack_joint_actions(actions)
        self._record_instrumentation("fast_vecenv_action_pack", perf_counter() - pack_started)

        stage_started = perf_counter()
        response = self._runtime.stepBatch(
            _java_int_array(slot_indices),
            _java_int_array(packed_actions),
            False,
            None,
            None,
        )
        self._record_instrumentation("fast_vecenv_step_batch_call", perf_counter() - stage_started)
        return response

    def _episode_configs(
        self,
        slot_indices: Sequence[int],
        seeds: Sequence[int] | None,
    ) -> list[tuple[int, int, int, int, int]]:
        configs: list[tuple[int, int, int, int, int]] = []
        provider = self.config.reset_options_provider
        for position, slot_index in enumerate(slot_indices):
            options = {}
            if provider is not None:
                provided = provider(
                    slot_index=int(slot_index),
                    episode_index=int(self._episodes_started[int(slot_index)]),
                )
                if provided is not None:
                    options = dict(provided)
            seed = (
                int(seeds[position])
                if seeds is not None
                else int(options.get("seed", int(slot_index) + int(self._episodes_started[int(slot_index)]) * self.num_agents))
            )
            configs.append(
                (
                    seed,
                    int(options.get("start_wave", self.config.start_wave)),
                    int(options.get("ammo", self.config.ammo)),
                    int(options.get("prayer_potions", self.config.prayer_potions)),
                    int(options.get("sharks", self.config.sharks)),
                )
            )
        return configs

    def _allocate_seeds(self, slot_indices: Sequence[int]) -> list[int] | None:
        if self._seed_base is None:
            return None
        stride = self.num_agents
        return [
            int(self._seed_base) + int(slot_index) + int(self._episodes_started[int(slot_index)]) * stride
            for slot_index in slot_indices
        ]

    def _record_episode_starts(self, slot_indices: Sequence[int]) -> None:
        for slot_index in slot_indices:
            slot = int(slot_index)
            self._episodes_started[slot] += 1
            self._episode_returns[slot] = 0.0
            self._episode_lengths[slot] = 0

    def _apply_reset_response(self, response: Any) -> None:
        stage_started = perf_counter()
        slot_indices = np.asarray(_jget(response, "slotIndices"), dtype=np.int32)
        observations = np.asarray(_jget(response, "flatObservations"), dtype=np.float32).reshape(
            int(_jget(response, "envCount")),
            self._observation_feature_count,
        )
        self.observations[slot_indices] = observations
        self.rewards[slot_indices] = 0.0
        self.terminals[slot_indices] = False
        self.truncations[slot_indices] = False
        self.teacher_actions[slot_indices] = 0
        self.masks[slot_indices] = True
        self._record_instrumentation("fast_vecenv_apply_reset_buffers", perf_counter() - stage_started)

    def _apply_step_response(
        self,
        response: Any,
        *,
        joint_actions: np.ndarray,
    ) -> None:
        stage_started = perf_counter()
        slot_indices = np.asarray(_jget(response, "slotIndices"), dtype=np.int32)
        observations = np.asarray(_jget(response, "flatObservations"), dtype=np.float32).reshape(
            int(_jget(response, "envCount")),
            self._observation_feature_count,
        )
        reward_features = np.asarray(_jget(response, "rewardFeatures"), dtype=np.float32).reshape(
            int(_jget(response, "envCount")),
            self._reward_feature_count,
        )
        rewards = self._reward_adapter.weight_batch(reward_features)
        terminals = np.asarray(_jget(response, "terminated"), dtype=np.bool_)
        truncations = np.asarray(_jget(response, "truncated"), dtype=np.bool_)

        self.observations[slot_indices] = observations
        self.rewards[slot_indices] = rewards
        self.terminals[slot_indices] = terminals
        self.truncations[slot_indices] = truncations
        self.masks[slot_indices] = True
        self.actions[slot_indices] = joint_actions[slot_indices]
        self._episode_returns[slot_indices] += rewards
        self._episode_lengths[slot_indices] += 1
        self._record_instrumentation("fast_vecenv_apply_step_buffers", perf_counter() - stage_started)

    def _record_instrumentation(self, bucket: str, seconds: float) -> None:
        if self._instrumentation is None:
            return
        self._instrumentation.record(bucket, seconds)


def _create_fast_kernel_runtime(config: FastKernelVecEnvConfig) -> tuple[Any, dict[str, Any]]:
    paths = discover_headless_runtime_paths()
    assert_sim_runtime_ready(paths)
    classes = _ensure_fast_jvm(paths.headless_jar, paths.launch_cwd)
    override_map = classes["HashMap"]()
    for key, value in build_headless_settings_overrides(paths, config.bootstrap).items():
        override_map.put(str(key), str(value))
    runtime = classes["FastFightCavesKernelRuntime"].createKernel(
        int(config.env_count),
        int(config.tick_cap),
        str(config.account_name_prefix),
        override_map,
    )
    return runtime, classes


def _ensure_fast_jvm(headless_jar: Path, launch_cwd: Path) -> dict[str, Any]:
    global _FAST_JVM_CONTEXT
    if _FAST_JVM_CONTEXT is not None:
        if _FAST_JVM_CONTEXT.classpath != headless_jar.resolve():
            raise BridgeJVMStateError(
                "Embedded JVM is already pinned to a different classpath: "
                f"{_FAST_JVM_CONTEXT.classpath} != {headless_jar.resolve()}"
            )
        if _FAST_JVM_CONTEXT.user_dir != launch_cwd.resolve():
            raise BridgeJVMStateError(
                "Embedded JVM is already pinned to a different user.dir: "
                f"{_FAST_JVM_CONTEXT.user_dir} != {launch_cwd.resolve()}"
            )
        return _FAST_JVM_CONTEXT.classes

    try:
        import jpype
        from jpype.types import JInt, JLong
    except ModuleNotFoundError as exc:
        raise BridgeJVMStateError("jpype1 is not installed in the RL environment.") from exc

    if not jpype.isJVMStarted():
        start_jvm_kwargs: dict[str, object] = {"classpath": [str(headless_jar.resolve())]}
        jvm_library_path = resolve_jvm_library_path()
        if jvm_library_path is not None:
            start_jvm_kwargs["jvmpath"] = str(jvm_library_path)
        try:
            jpype.startJVM(
                f"-Duser.dir={launch_cwd.resolve()}",
                f"-Dlogback.configurationFile={QUIET_LOGBACK_CONFIG.resolve()}",
                **start_jvm_kwargs,
            )
        except jpype.JVMNotFoundException as exc:
            raise BridgeJVMStateError(
                "Could not resolve a Linux JVM runtime for the embedded fast kernel. "
                "Set FC_RL_JAVA_HOME or JAVA_HOME to a JDK/JRE home containing "
                "bin/java and lib/server/libjvm.so."
            ) from exc

    classes = {
        "jpype": jpype,
        "JInt": JInt,
        "JLong": JLong,
        "HashMap": jpype.JClass("java.util.HashMap"),
        "ArrayList": jpype.JClass("java.util.ArrayList"),
        "FastFightCavesKernelRuntime": jpype.JClass("FastFightCavesKernelRuntime"),
        "FastEpisodeConfig": jpype.JClass("headless.fast.FastEpisodeConfig"),
    }
    _FAST_JVM_CONTEXT = _FastJVMContext(
        jpype=jpype,
        classpath=headless_jar.resolve(),
        user_dir=launch_cwd.resolve(),
        classes=classes,
    )
    return classes


def _java_episode_configs(classes: dict[str, Any], configs: Sequence[tuple[int, int, int, int, int]]) -> Any:
    items = classes["ArrayList"]()
    for seed, start_wave, ammo, prayer_potions, sharks in configs:
        items.add(
            classes["FastEpisodeConfig"](
                classes["JLong"](int(seed)),
                int(start_wave),
                int(ammo),
                int(prayer_potions),
                int(sharks),
            )
        )
    return items


def _java_int_array(values: Sequence[int] | np.ndarray) -> Any:
    context = _FAST_JVM_CONTEXT
    if context is None:
        raise BridgeJVMStateError("Fast JVM context is not initialized.")
    jpype = context.jpype
    JInt = context.classes["JInt"]
    return jpype.JArray(JInt)([int(value) for value in values])


def _jget(target: Any, name: str) -> Any:
    if hasattr(target, name):
        return getattr(target, name)
    getter_name = "get" + name[0].upper() + name[1:]
    getter = getattr(target, getter_name, None)
    if getter is None:
        raise AttributeError(f"{target!r} has neither attribute {name!r} nor getter {getter_name!r}.")
    return getter()
