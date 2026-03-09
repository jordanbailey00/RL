from __future__ import annotations

from typing import Any

import numpy as np

from fight_caves_rl.envs.observation_views import (
    observation_player_hitpoints_current,
    observation_remaining,
    observation_wave,
)
from fight_caves_rl.rewards.registry import RewardConfig


def build_reward_fn(config: RewardConfig):
    wave_progress = float(config.coefficients.get("wave_progress", 1.0))
    cave_complete = float(config.coefficients.get("cave_complete", 10.0))
    player_death = float(config.coefficients.get("player_death", -1.0))

    def reward_fn(
        previous_observation: dict[str, Any] | np.ndarray | None,
        action_result: dict[str, Any],
        observation: dict[str, Any] | np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> float:
        del action_result, truncated
        reward = 0.0
        reward += wave_progress * _wave_delta(previous_observation, observation)
        if terminated:
            if _is_success(observation):
                reward += cave_complete
            elif _is_death(observation):
                reward += player_death
        return float(reward)

    return reward_fn


def _wave_delta(
    previous_observation: dict[str, Any] | np.ndarray | None,
    observation: dict[str, Any] | np.ndarray,
) -> float:
    if previous_observation is None:
        return 0.0
    return float(observation_wave(observation) - observation_wave(previous_observation))


def _is_success(observation: dict[str, Any] | np.ndarray) -> bool:
    return (
        observation_wave(observation) == 63
        and observation_remaining(observation) == 0
        and observation_player_hitpoints_current(observation) > 0
    )


def _is_death(observation: dict[str, Any] | np.ndarray) -> bool:
    return observation_player_hitpoints_current(observation) <= 0
