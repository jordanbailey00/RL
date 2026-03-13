from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from time import perf_counter, perf_counter_ns
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from fight_caves_rl.benchmarks.instrumentation import (
    BucketInstrumentation,
    InstrumentationSnapshot,
)
from fight_caves_rl.bridge.contracts import (
    HeadlessBootstrapConfig,
    HeadlessEpisodeConfig,
    HeadlessPlayerConfig,
)
from fight_caves_rl.bridge.debug_client import (
    HeadlessDebugClient,
    pythonize_action_result,
    pythonize_observation,
)
from fight_caves_rl.bridge.errors import BatchSlotExecutionError, BridgeContractError
from fight_caves_rl.bridge.protocol import (
    BatchBridgeProtocol,
    BatchResetResponse,
    BatchSlotResetResult,
    BatchSlotStepResult,
    BatchStepResponse,
    build_batch_protocol,
    build_batch_reset_request,
    build_batch_step_request,
)
from fight_caves_rl.envs.action_mapping import NormalizedAction
from fight_caves_rl.envs.correctness_env import infer_terminal_state
from fight_caves_rl.envs.observation_views import (
    observation_tick,
    observation_visible_target_count,
    observation_visible_targets,
)
from fight_caves_rl.envs.shared_memory_transport import (
    INFO_PAYLOAD_MODE_FULL,
    INFO_PAYLOAD_MODE_MINIMAL,
)

ObservationLike = dict[str, Any] | np.ndarray
RewardFn = Callable[[ObservationLike | None, dict[str, Any], ObservationLike, bool, bool], float]


def zero_reward(
    previous_observation: dict[str, Any] | None,
    action_result: dict[str, Any],
    observation: dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> float:
    return 0.0


@dataclass(frozen=True)
class BatchClientConfig:
    env_count: int
    account_name_prefix: str = "rl_batch"
    start_wave: int = 1
    ammo: int = 1000
    prayer_potions: int = 8
    sharks: int = 20
    tick_cap: int = 20_000
    include_future_leakage: bool = False
    info_payload_mode: str = INFO_PAYLOAD_MODE_FULL
    instrumentation_enabled: bool = False
    bootstrap: HeadlessBootstrapConfig = field(default_factory=HeadlessBootstrapConfig)


@dataclass
class _SlotState:
    slot_index: int
    player: Any
    account_name: str
    episode_start_tick: int | None = None
    episode_steps: int = 0
    episode_return: float = 0.0
    last_observation: ObservationLike | None = None


class HeadlessBatchClient:
    def __init__(
        self,
        config: BatchClientConfig,
        *,
        reward_fn: RewardFn = zero_reward,
    ) -> None:
        if int(config.env_count) <= 0:
            raise ValueError(f"env_count must be > 0, got {config.env_count}.")
        self.config = config
        self.reward_fn = reward_fn
        self.client = HeadlessDebugClient.create(bootstrap=config.bootstrap)
        self.protocol: BatchBridgeProtocol = build_batch_protocol(self.client.handshake)
        self._rng = Random()
        self._closed = False
        self._slots = tuple(self._create_slots())
        self._instrumentation = (
            BucketInstrumentation() if bool(config.instrumentation_enabled) else None
        )

    @classmethod
    def create(
        cls,
        config: BatchClientConfig,
        *,
        reward_fn: RewardFn = zero_reward,
    ) -> "HeadlessBatchClient":
        return cls(config=config, reward_fn=reward_fn)

    @property
    def slot_count(self) -> int:
        return len(self._slots)

    def instrumentation_snapshot(self) -> InstrumentationSnapshot:
        if self._instrumentation is None:
            return {}
        return self._instrumentation.snapshot()

    def reset_instrumentation(self) -> None:
        if self._instrumentation is None:
            return
        self._instrumentation.clear()

    def reset_batch(
        self,
        *,
        seeds: Sequence[int] | None = None,
        options: Sequence[Mapping[str, object] | None] | None = None,
        slot_indices: Sequence[int] | None = None,
    ) -> BatchResetResponse:
        selected_indices = self._resolve_slot_indices(slot_indices)
        episodes = self._build_episode_configs(
            slot_indices=selected_indices,
            seeds=seeds,
            options=options,
        )
        request = build_batch_reset_request(
            self.protocol,
            slot_indices=selected_indices,
            episodes=episodes,
        )

        started = perf_counter_ns()
        stage_started = perf_counter()
        results: list[BatchSlotResetResult] = []
        pending_reset_state: list[tuple[int, dict[str, Any] | None, dict[str, Any]]] = []
        for spec in request.resets:
            slot = self._slot(spec.slot_index)
            try:
                bucket_started = perf_counter()
                episode_state = self.client.reset_episode(slot.player, spec.episode)
                self._record_instrumentation("client_reset_episode", perf_counter() - bucket_started)
                observation = None
                if self.config.include_future_leakage or not self._use_minimal_info_payload():
                    bucket_started = perf_counter()
                    observation = self.client.observe(
                        slot.player,
                        include_future_leakage=self.config.include_future_leakage,
                    )
                    self._record_instrumentation(
                        "client_reset_observe_raw",
                        perf_counter() - bucket_started,
                    )
            except Exception as exc:  # pragma: no cover - live bridge failures are integration-tested
                raise BatchSlotExecutionError(spec.slot_index, "reset", str(exc)) from exc

            reset_info = (
                {}
                if self._use_minimal_info_payload()
                else {
                    "episode_state": episode_state,
                    "bridge_handshake": dict(self.client.handshake.values),
                    "batch_protocol": self.protocol.to_dict(),
                }
            )
            pending_reset_state.append((spec.slot_index, observation, reset_info))
        if self._supports_batch_flat_observe():
            players = [self._slot(spec.slot_index).player for spec in request.resets]
            bucket_started = perf_counter()
            flat_observations = self.client.observe_flat_batch(players)
            elapsed = perf_counter() - bucket_started
            self._record_instrumentation("client_reset_observe_flat_batch", elapsed)
            self._record_instrumentation(
                "client_reset_observe_flat",
                elapsed,
                calls=len(request.resets),
            )
        else:
            flat_observations = None

        for result_index, spec in enumerate(request.resets):
            slot = self._slot(spec.slot_index)
            _, observation, reset_info = pending_reset_state[result_index]
            if flat_observations is not None:
                flat_observation = flat_observations[result_index]
            else:
                try:
                    bucket_started = perf_counter()
                    flat_observation = self.client.observe_flat(slot.player)
                    self._record_instrumentation(
                        "client_reset_observe_flat",
                        perf_counter() - bucket_started,
                    )
                except Exception as exc:  # pragma: no cover - live bridge failures are integration-tested
                    raise BatchSlotExecutionError(spec.slot_index, "reset_observe_flat", str(exc)) from exc

            slot.episode_start_tick = observation_tick(flat_observation)
            slot.episode_steps = 0
            slot.episode_return = 0.0
            slot.last_observation = (
                observation if self.config.include_future_leakage else flat_observation
            )
            results.append(
                BatchSlotResetResult(
                    slot_index=spec.slot_index,
                    observation=observation,
                    flat_observation=flat_observation,
                    episode_state=episode_state,
                    info=reset_info,
                )
            )
        self._record_instrumentation("client_reset_total", perf_counter() - stage_started)

        return BatchResetResponse(
            protocol=self.protocol,
            results=tuple(results),
            elapsed_nanos=perf_counter_ns() - started,
        )

    def step_reference(
        self,
        actions: Sequence[int | str | Mapping[str, object] | NormalizedAction],
        *,
        slot_indices: Sequence[int] | None = None,
        ticks_after: int = 1,
    ) -> BatchStepResponse:
        request = self._build_step_request(
            actions=actions,
            slot_indices=slot_indices,
            ticks_after=ticks_after,
        )
        started = perf_counter_ns()
        stage_started = perf_counter()
        action_results: dict[int, dict[str, Any]] = {}

        for spec in request.actions:
            slot = self._slot(spec.slot_index)
            self._ensure_slot_ready(slot)
            try:
                bucket_started = perf_counter()
                action_results[spec.slot_index] = self.client.apply_action(slot.player, spec.action)
                self._record_instrumentation("client_apply_action_raw", perf_counter() - bucket_started)
            except Exception as exc:  # pragma: no cover - live bridge failures are integration-tested
                raise BatchSlotExecutionError(spec.slot_index, "apply_action", str(exc)) from exc

        tick_started = perf_counter()
        self.client.tick(request.ticks_after)
        self._record_instrumentation("client_runtime_tick", perf_counter() - tick_started)
        collect_started = perf_counter()
        results = self._collect_step_results(
            request.actions,
            action_results=action_results,
            use_raw_observe=True,
        )
        self._record_instrumentation("client_collect_step_results", perf_counter() - collect_started)
        self._record_instrumentation("client_step_reference_total", perf_counter() - stage_started)
        return BatchStepResponse(
            protocol=self.protocol,
            results=tuple(results),
            elapsed_nanos=perf_counter_ns() - started,
        )

    def step_batch(
        self,
        actions: Sequence[int | str | Mapping[str, object] | NormalizedAction],
        *,
        slot_indices: Sequence[int] | None = None,
        ticks_after: int = 1,
    ) -> BatchStepResponse:
        stage_started = perf_counter()
        request = self._build_step_request(
            actions=actions,
            slot_indices=slot_indices,
            ticks_after=ticks_after,
        )
        self._record_instrumentation("client_build_step_request", perf_counter() - stage_started)
        started = perf_counter_ns()
        stage_started = perf_counter()
        action_results: dict[int, dict[str, Any]] = {}
        if self._supports_batch_apply_actions():
            players = []
            actions_for_batch = []
            for spec in request.actions:
                slot = self._slot(spec.slot_index)
                self._ensure_slot_ready(slot)
                players.append(slot.player)
                actions_for_batch.append(spec.action)
            try:
                bucket_started = perf_counter()
                batch_results = self.client.apply_actions_batch(players, actions_for_batch)
                elapsed = perf_counter() - bucket_started
                self._record_instrumentation("client_apply_actions_batch_jvm", elapsed)
                self._record_instrumentation(
                    "client_apply_action_jvm",
                    elapsed,
                    calls=len(request.actions),
                )
                for spec, result in zip(request.actions, batch_results, strict=True):
                    action_results[spec.slot_index] = result
            except Exception as exc:  # pragma: no cover - live bridge failures are integration-tested
                raise BatchSlotExecutionError(-1, "apply_actions_batch_jvm", str(exc)) from exc
        else:
            for spec in request.actions:
                slot = self._slot(spec.slot_index)
                self._ensure_slot_ready(slot)
                try:
                    bucket_started = perf_counter()
                    result = self.client.apply_action_jvm(slot.player, spec.action)
                    action_results[spec.slot_index] = pythonize_action_result(result)
                    self._record_instrumentation("client_apply_action_jvm", perf_counter() - bucket_started)
                except Exception as exc:  # pragma: no cover - live bridge failures are integration-tested
                    raise BatchSlotExecutionError(spec.slot_index, "apply_action_jvm", str(exc)) from exc

        tick_started = perf_counter()
        self.client.tick(request.ticks_after)
        self._record_instrumentation("client_runtime_tick", perf_counter() - tick_started)
        collect_started = perf_counter()
        results = self._collect_step_results(
            request.actions,
            action_results=action_results,
            use_flat_observe=not self.config.include_future_leakage,
        )
        self._record_instrumentation("client_collect_step_results", perf_counter() - collect_started)
        self._record_instrumentation("client_step_batch_total", perf_counter() - stage_started)
        return BatchStepResponse(
            protocol=self.protocol,
            results=tuple(results),
            elapsed_nanos=perf_counter_ns() - started,
        )

    def run_action_trace(
        self,
        slot_index: int,
        actions: Sequence[int | str | Mapping[str, object] | NormalizedAction],
        *,
        ticks_after: int = 1,
        observe_every: int = 0,
    ) -> dict[str, Any]:
        slot = self._slot(slot_index)
        self._ensure_slot_ready(slot)
        return self.client.run_action_trace(
            slot.player,
            list(actions),
            ticks_after=ticks_after,
            observe_every=observe_every,
            include_future_leakage=self.config.include_future_leakage,
        )

    def close(self) -> None:
        if self._closed:
            return
        self.client.close()
        self._closed = True

    def _collect_step_results(
        self,
        action_specs: Sequence[Any],
        *,
        action_results: Mapping[int, dict[str, Any]],
        use_raw_observe: bool = False,
        use_flat_observe: bool = False,
    ) -> list[BatchSlotStepResult]:
        results: list[BatchSlotStepResult] = []
        minimal_info_payload = self._use_minimal_info_payload()
        flat_observation_rows: dict[int, np.ndarray] = {}
        if use_flat_observe and self._supports_batch_flat_observe():
            players = [self._slot(spec.slot_index).player for spec in action_specs]
            try:
                bucket_started = perf_counter()
                flat_batch = self.client.observe_flat_batch(players)
                elapsed = perf_counter() - bucket_started
                self._record_instrumentation("client_flat_observe_batch", elapsed)
                self._record_instrumentation(
                    "client_flat_observe",
                    elapsed,
                    calls=len(action_specs),
                )
                for action_index, spec in enumerate(action_specs):
                    flat_observation_rows[spec.slot_index] = flat_batch[action_index]
            except Exception as exc:  # pragma: no cover - live bridge failures are integration-tested
                raise BatchSlotExecutionError(-1, "observe_flat_batch", str(exc)) from exc
        for spec in action_specs:
            slot = self._slot(spec.slot_index)
            previous_observation = slot.last_observation
            try:
                if use_flat_observe:
                    if flat_observation_rows:
                        flat_observation = flat_observation_rows[spec.slot_index]
                    else:
                        bucket_started = perf_counter()
                        flat_observation = self.client.observe_flat(slot.player)
                        self._record_instrumentation("client_flat_observe", perf_counter() - bucket_started)
                    observation = None
                    observation_for_semantics: ObservationLike = flat_observation
                elif use_raw_observe:
                    bucket_started = perf_counter()
                    observation = pythonize_observation(
                        self.client.observe_jvm(
                            slot.player,
                            include_future_leakage=self.config.include_future_leakage,
                        )
                    )
                    self._record_instrumentation("client_raw_observe", perf_counter() - bucket_started)
                    flat_observation = None
                    observation_for_semantics = observation
                else:
                    bucket_started = perf_counter()
                    observation = self.client.observe(
                        slot.player,
                        include_future_leakage=self.config.include_future_leakage,
                    )
                    self._record_instrumentation(
                        "client_structured_observe",
                        perf_counter() - bucket_started,
                    )
                    flat_observation = None
                    observation_for_semantics = observation
            except Exception as exc:  # pragma: no cover - live bridge failures are integration-tested
                raise BatchSlotExecutionError(spec.slot_index, "observe", str(exc)) from exc

            slot.episode_steps += 1
            bucket_started = perf_counter()
            terminated, truncated, terminal_reason = infer_terminal_state(
                observation=observation_for_semantics,
                episode_start_tick=slot.episode_start_tick,
                tick_cap=self.config.tick_cap,
            )
            self._record_instrumentation("client_terminal_inference", perf_counter() - bucket_started)
            action_result = dict(action_results[spec.slot_index])
            bucket_started = perf_counter()
            reward = float(
                self.reward_fn(
                    previous_observation,
                    action_result,
                    observation_for_semantics,
                    terminated,
                    truncated,
                )
            )
            self._record_instrumentation("client_reward_eval", perf_counter() - bucket_started)
            slot.episode_return += reward
            slot.last_observation = observation_for_semantics
            bucket_started = perf_counter()
            visible_target_count = 0
            visible_targets: list[dict[str, Any]] | None = None
            info: dict[str, Any] = {}
            if not minimal_info_payload:
                visible_target_count = observation_visible_target_count(observation_for_semantics)
                visible_targets = observation_visible_targets(observation_for_semantics)
                info = {
                    "action_result": action_result,
                    "visible_targets": visible_targets,
                    "visible_target_count": visible_target_count,
                    "episode_steps": slot.episode_steps,
                    "episode_return": slot.episode_return,
                    "terminal_reason": terminal_reason,
                    "terminal_reason_inferred": terminal_reason is not None,
                    "observation_path_mode": "flat" if use_flat_observe else "raw",
                }
            self._record_instrumentation("client_info_assembly", perf_counter() - bucket_started)
            results.append(
                BatchSlotStepResult(
                    slot_index=spec.slot_index,
                    action=spec.action,
                    observation=observation,
                    flat_observation=flat_observation,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    action_result=action_result,
                    visible_target_count=visible_target_count,
                    episode_steps=slot.episode_steps,
                    episode_return=slot.episode_return,
                    terminal_reason=terminal_reason,
                    visible_targets=visible_targets,
                    info=info,
                )
            )
        return results

    def _record_instrumentation(self, bucket: str, seconds: float, *, calls: int = 1) -> None:
        if self._instrumentation is None:
            return
        self._instrumentation.record(bucket, seconds, calls=calls)

    def _use_minimal_info_payload(self) -> bool:
        return str(self.config.info_payload_mode) == INFO_PAYLOAD_MODE_MINIMAL

    def _supports_batch_apply_actions(self) -> bool:
        return self.protocol.batch_apply_actions_api == "applyActionsBatch"

    def _supports_batch_flat_observe(self) -> bool:
        return (
            self.protocol.batch_observe_flat_api == "observeFlatBatch"
            and self.protocol.batch_flat_observation_layout == "env_major_contiguous_float32"
        )

    def _create_slots(self) -> list[_SlotState]:
        slots: list[_SlotState] = []
        for slot_index in range(int(self.config.env_count)):
            account_name = f"{self.config.account_name_prefix}_{slot_index}"
            player = self.client.create_player_slot(
                HeadlessPlayerConfig(account_name=account_name)
            )
            slots.append(
                _SlotState(
                    slot_index=slot_index,
                    player=player,
                    account_name=account_name,
                )
            )
        return slots

    def _build_step_request(
        self,
        *,
        actions: Sequence[int | str | Mapping[str, object] | NormalizedAction],
        slot_indices: Sequence[int] | None,
        ticks_after: int,
    ):
        selected_indices = self._resolve_slot_indices(slot_indices)
        return build_batch_step_request(
            self.protocol,
            slot_indices=selected_indices,
            actions=actions,
            ticks_after=ticks_after,
        )

    def _build_episode_configs(
        self,
        *,
        slot_indices: Sequence[int],
        seeds: Sequence[int] | None,
        options: Sequence[Mapping[str, object] | None] | None,
    ) -> list[HeadlessEpisodeConfig]:
        if seeds is not None and len(seeds) != len(slot_indices):
            raise BridgeContractError(
                "Batch reset seeds length mismatch: "
                f"{len(seeds)} != {len(slot_indices)}"
            )
        if options is not None and len(options) != len(slot_indices):
            raise BridgeContractError(
                "Batch reset options length mismatch: "
                f"{len(options)} != {len(slot_indices)}"
            )

        episodes: list[HeadlessEpisodeConfig] = []
        for local_index, slot_index in enumerate(slot_indices):
            option_map = dict((options or [None] * len(slot_indices))[local_index] or {})
            seed = (
                int(seeds[local_index])
                if seeds is not None
                else self._rng.randrange(0, 2**31)
            )
            episodes.append(
                HeadlessEpisodeConfig(
                    seed=seed,
                    start_wave=int(option_map.get("start_wave", self.config.start_wave)),
                    ammo=int(option_map.get("ammo", self.config.ammo)),
                    prayer_potions=int(
                        option_map.get("prayer_potions", self.config.prayer_potions)
                    ),
                    sharks=int(option_map.get("sharks", self.config.sharks)),
                )
            )
        return episodes

    def _resolve_slot_indices(self, slot_indices: Sequence[int] | None) -> tuple[int, ...]:
        if slot_indices is None:
            return tuple(range(self.slot_count))
        resolved = tuple(int(index) for index in slot_indices)
        for index in resolved:
            if index < 0 or index >= self.slot_count:
                raise BridgeContractError(
                    f"Slot index {index} is outside the valid range [0, {self.slot_count})."
                )
        return resolved

    def _ensure_slot_ready(self, slot: _SlotState) -> None:
        if slot.last_observation is None or slot.episode_start_tick is None:
            raise BridgeContractError(
                f"Batch slot {slot.slot_index} has not been reset yet."
            )

    def _slot(self, slot_index: int) -> _SlotState:
        return self._slots[int(slot_index)]
