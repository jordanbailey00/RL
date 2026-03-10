from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pufferlib
import pufferlib.vector

from fight_caves_rl.bridge.batch_client import BatchClientConfig, HeadlessBatchClient
from fight_caves_rl.bridge.buffers import build_reset_buffers, build_step_buffers
from fight_caves_rl.bridge.contracts import HeadlessBootstrapConfig
from fight_caves_rl.envs.action_mapping import NormalizedAction
from fight_caves_rl.envs.observation_views import (
    observation_episode_seed,
    observation_remaining,
    observation_wave,
)
from fight_caves_rl.envs.puffer_encoding import (
    build_policy_action_space,
    build_policy_observation_space,
    decode_action_from_policy,
)
from fight_caves_rl.envs.shared_memory_transport import (
    INFO_PAYLOAD_MODE_FULL,
    INFO_PAYLOAD_MODE_MINIMAL,
    INFO_PAYLOAD_MODES,
)

RewardFn = Callable[[dict[str, Any] | None, dict[str, Any], dict[str, Any], bool, bool], float]
ResetOptionsProvider = Callable[[int, int], Mapping[str, object] | None]


@dataclass(frozen=True)
class HeadlessBatchVecEnvConfig:
    env_count: int
    account_name_prefix: str = "rl_vecenv"
    start_wave: int = 1
    ammo: int = 1000
    prayer_potions: int = 8
    sharks: int = 20
    tick_cap: int = 20_000
    include_future_leakage: bool = False
    info_payload_mode: str = INFO_PAYLOAD_MODE_FULL
    bootstrap: HeadlessBootstrapConfig = field(default_factory=HeadlessBootstrapConfig)
    reset_options_provider: ResetOptionsProvider | None = None


class HeadlessBatchVecEnv:
    reset = pufferlib.vector.reset
    step = pufferlib.vector.step

    def __init__(
        self,
        config: HeadlessBatchVecEnvConfig,
        *,
        reward_fn: RewardFn,
    ) -> None:
        if int(config.env_count) <= 0:
            raise ValueError(f"env_count must be > 0, got {config.env_count}.")

        self.config = config
        if str(config.info_payload_mode) not in INFO_PAYLOAD_MODES:
            raise ValueError(
                f"Unsupported info_payload_mode: {config.info_payload_mode!r}. "
                f"Expected one of {INFO_PAYLOAD_MODES!r}."
            )
        self.client = HeadlessBatchClient.create(
            BatchClientConfig(
                env_count=int(config.env_count),
                account_name_prefix=str(config.account_name_prefix),
                start_wave=int(config.start_wave),
                ammo=int(config.ammo),
                prayer_potions=int(config.prayer_potions),
                sharks=int(config.sharks),
                tick_cap=int(config.tick_cap),
                include_future_leakage=bool(config.include_future_leakage),
                bootstrap=config.bootstrap,
            ),
            reward_fn=reward_fn,
        )
        self.driver_env = self
        self.agents_per_batch = int(config.env_count)
        self.num_agents = self.agents_per_batch
        self.single_observation_space = build_policy_observation_space()
        self.single_action_space = build_policy_action_space()
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
        self._minimal_infos = tuple({} for _ in range(self.num_agents))

    @property
    def num_envs(self) -> int:
        return self.agents_per_batch

    @property
    def episode_counts(self) -> np.ndarray:
        return self._episodes_started.copy()

    def async_reset(self, seed: int | None = None) -> None:
        self.flag = pufferlib.vector.RECV
        self._seed_base = None if seed is None else int(seed)
        self._episodes_started.fill(0)
        slot_indices = tuple(range(self.num_agents))
        response = self.client.reset_batch(
            seeds=self._allocate_seeds(slot_indices),
            options=self._build_reset_options(slot_indices),
        )
        self._apply_reset_response(response.results)
        self._record_episode_starts(slot_indices)
        if self._use_minimal_infos():
            self.infos = list(self._minimal_infos)
            return
        self.infos = [
            self._build_reset_info(result)
            for result in sorted(response.results, key=lambda result: int(result.slot_index))
        ]

    def send(self, actions: np.ndarray) -> None:
        if not actions.flags.contiguous:
            actions = np.ascontiguousarray(actions)

        actions = pufferlib.vector.send_precheck(self, actions)
        if self._use_minimal_infos():
            ordered_infos: list[dict[str, Any]] = list(self._minimal_infos)
        else:
            ordered_infos = [{} for _ in range(self.num_agents)]

        done_indices = tuple(
            int(index) for index in np.flatnonzero(np.logical_or(self.terminals, self.truncations))
        )
        if done_indices:
            reset_response = self.client.reset_batch(
                slot_indices=done_indices,
                seeds=self._allocate_seeds(done_indices),
                options=self._build_reset_options(done_indices),
            )
            self._apply_reset_response(reset_response.results)
            self._record_episode_starts(done_indices)
            if not self._use_minimal_infos():
                for result in reset_response.results:
                    ordered_infos[int(result.slot_index)] = self._build_reset_info(result)

        active_indices = tuple(index for index in range(self.num_agents) if index not in done_indices)
        if active_indices:
            normalized_actions = [
                self._decode_joint_action(actions[int(slot_index)])
                for slot_index in active_indices
            ]
            step_response = self.client.step_batch(
                normalized_actions,
                slot_indices=active_indices,
            )
            self._apply_step_response(step_response.results, joint_actions=actions)
            if not self._use_minimal_infos():
                for result in step_response.results:
                    ordered_infos[int(result.slot_index)] = self._build_step_info(result)

        self.infos = ordered_infos

    def recv(self):
        pufferlib.vector.recv_precheck(self)
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            self.teacher_actions,
            self.infos,
            self.agent_ids,
            self.masks,
        )

    def notify(self) -> None:
        return None

    def close(self) -> None:
        self.client.close()

    def _allocate_seeds(self, slot_indices: Sequence[int]) -> list[int] | None:
        if self._seed_base is None:
            return None
        seeds: list[int] = []
        stride = self.num_agents
        for slot_index in slot_indices:
            seed = int(self._seed_base) + int(slot_index) + int(self._episodes_started[int(slot_index)]) * stride
            seeds.append(seed)
        return seeds

    def _build_reset_options(
        self,
        slot_indices: Sequence[int],
    ) -> list[Mapping[str, object] | None] | None:
        provider = self.config.reset_options_provider
        if provider is None:
            return None
        return [
            provider(
                slot_index=int(slot_index),
                episode_index=int(self._episodes_started[int(slot_index)]),
            )
            for slot_index in slot_indices
        ]

    def _record_episode_starts(self, slot_indices: Sequence[int]) -> None:
        for slot_index in slot_indices:
            self._episodes_started[int(slot_index)] += 1

    def _apply_reset_response(self, results: Sequence[Any]) -> None:
        buffers = build_reset_buffers(results)
        self.observations[buffers.slot_indices] = buffers.policy_observations
        self.rewards[buffers.slot_indices] = 0.0
        self.terminals[buffers.slot_indices] = False
        self.truncations[buffers.slot_indices] = False
        self.teacher_actions[buffers.slot_indices] = 0
        self.masks[buffers.slot_indices] = True

    def _apply_step_response(
        self,
        results: Sequence[Any],
        *,
        joint_actions: np.ndarray,
    ) -> None:
        buffers = build_step_buffers(results)
        self.observations[buffers.slot_indices] = buffers.policy_observations
        self.rewards[buffers.slot_indices] = buffers.rewards
        self.terminals[buffers.slot_indices] = buffers.terminated
        self.truncations[buffers.slot_indices] = buffers.truncated
        self.masks[buffers.slot_indices] = True
        self.actions[buffers.slot_indices] = joint_actions[buffers.slot_indices]

    def _decode_joint_action(self, action: np.ndarray) -> NormalizedAction:
        return decode_action_from_policy(np.asarray(action, dtype=np.int64))

    def _use_minimal_infos(self) -> bool:
        return str(self.config.info_payload_mode) == INFO_PAYLOAD_MODE_MINIMAL

    def _build_reset_info(self, result: Any) -> dict[str, Any]:
        observation = result.flat_observation if result.flat_observation is not None else result.observation
        return {
            "slot_index": int(result.slot_index),
            "episode_seed": observation_episode_seed(observation),
            "wave": observation_wave(observation),
            "remaining": observation_remaining(observation),
            "vecenv_event": "reset",
            **dict(result.info),
        }

    def _build_step_info(self, result: Any) -> dict[str, Any]:
        action_result = dict(result.info["action_result"])
        return {
            "slot_index": int(result.slot_index),
            "action_id": int(result.action.action_id),
            "visible_target_count": int(result.info["visible_target_count"]),
            "wave": observation_wave(
                result.flat_observation if result.flat_observation is not None else result.observation
            ),
            "remaining": observation_remaining(
                result.flat_observation if result.flat_observation is not None else result.observation
            ),
            "reward": float(result.reward),
            "action_applied": int(bool(action_result["action_applied"])),
            "rejection_reason": action_result["rejection_reason"],
            "terminal_reason": result.info["terminal_reason"],
            "episode_steps": int(result.info["episode_steps"]),
            "episode_return": float(result.info["episode_return"]),
            "terminated": int(bool(result.terminated)),
            "truncated": int(bool(result.truncated)),
            "vecenv_event": "step",
        }
