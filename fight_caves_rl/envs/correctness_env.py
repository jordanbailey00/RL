from __future__ import annotations

from dataclasses import dataclass, field
from random import Random
from typing import Any, Callable, Mapping

from fight_caves_rl.bridge.contracts import HeadlessBootstrapConfig, HeadlessEpisodeConfig, HeadlessPlayerConfig
from fight_caves_rl.bridge.debug_client import HeadlessDebugClient
from fight_caves_rl.envs.action_mapping import NormalizedAction, normalize_action

RewardFn = Callable[[dict[str, Any] | None, dict[str, Any], dict[str, Any], bool, bool], float]


def _zero_reward(
    previous_observation: dict[str, Any] | None,
    action_result: dict[str, Any],
    observation: dict[str, Any],
    terminated: bool,
    truncated: bool,
) -> float:
    return 0.0


@dataclass(frozen=True)
class CorrectnessEnvConfig:
    account_name: str = "rl_correctness_env"
    start_wave: int = 1
    ammo: int = 1000
    prayer_potions: int = 8
    sharks: int = 20
    tick_cap: int = 20_000
    include_future_leakage: bool = False
    bootstrap: HeadlessBootstrapConfig = field(default_factory=HeadlessBootstrapConfig)


class FightCavesCorrectnessEnv:
    def __init__(
        self,
        config: CorrectnessEnvConfig | None = None,
        reward_fn: RewardFn = _zero_reward,
    ) -> None:
        self.config = config or CorrectnessEnvConfig()
        self.reward_fn = reward_fn
        self.client = HeadlessDebugClient.create(bootstrap=self.config.bootstrap)
        self.player = self.client.create_player_slot(
            HeadlessPlayerConfig(account_name=self.config.account_name)
        )
        self._rng = Random()
        self._episode_steps = 0
        self._episode_start_tick: int | None = None
        self._last_observation: dict[str, Any] | None = None
        self._closed = False

    def reset(
        self,
        seed: int | None = None,
        options: Mapping[str, object] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_open()
        options_map = dict(options or {})
        episode_seed = int(seed if seed is not None else self._rng.randrange(0, 2**31))
        episode = HeadlessEpisodeConfig(
            seed=episode_seed,
            start_wave=int(options_map.get("start_wave", self.config.start_wave)),
            ammo=int(options_map.get("ammo", self.config.ammo)),
            prayer_potions=int(options_map.get("prayer_potions", self.config.prayer_potions)),
            sharks=int(options_map.get("sharks", self.config.sharks)),
        )
        episode_state = self.client.reset_episode(self.player, episode)
        observation = self.client.observe(
            self.player,
            include_future_leakage=self.config.include_future_leakage,
        )
        self._episode_steps = 0
        self._episode_start_tick = int(observation["tick"])
        self._last_observation = observation
        info = {
            "episode_state": episode_state,
            "bridge_handshake": dict(self.client.handshake.values),
        }
        return observation, info

    def step(
        self,
        action: int | str | Mapping[str, object] | NormalizedAction,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._ensure_open()
        normalized = normalize_action(action)
        snapshot = self.client.step_once(
            self.player,
            normalized,
            include_future_leakage=self.config.include_future_leakage,
        )
        self._episode_steps += 1

        terminated, truncated, terminal_reason = infer_terminal_state(
            observation=snapshot.observation,
            episode_start_tick=self._episode_start_tick,
            tick_cap=self.config.tick_cap,
        )
        reward = self.reward_fn(
            self._last_observation,
            snapshot.action_result,
            snapshot.observation,
            terminated,
            truncated,
        )
        self._last_observation = snapshot.observation
        info = {
            "action_result": snapshot.action_result,
            "visible_targets": snapshot.visible_targets,
            "episode_steps": self._episode_steps,
            "terminal_reason": terminal_reason,
            "terminal_reason_inferred": terminal_reason is not None,
        }
        return snapshot.observation, float(reward), terminated, truncated, info

    def close(self) -> None:
        if self._closed:
            return
        self.client.close()
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("FightCavesCorrectnessEnv is already closed.")


def infer_terminal_state(
    observation: Mapping[str, Any],
    episode_start_tick: int | None,
    tick_cap: int,
) -> tuple[bool, bool, str | None]:
    player = observation["player"]
    wave = observation["wave"]
    if int(player["hitpoints_current"]) <= 0:
        return True, False, "player_death"
    if int(wave["wave"]) == 63 and int(wave["remaining"]) == 0:
        return True, False, "cave_complete"

    if episode_start_tick is None:
        return False, False, None
    ticks_elapsed = int(observation["tick"]) - int(episode_start_tick)
    if ticks_elapsed >= tick_cap:
        return False, True, "max_tick_cap"
    return False, False, None
