from __future__ import annotations

from typing import Any

from fight_caves_rl.rewards.registry import RewardConfig


def build_reward_fn(config: RewardConfig):
    wave_progress = float(config.coefficients.get("wave_progress", 1.0))
    cave_complete = float(config.coefficients.get("cave_complete", 10.0))
    player_death = float(config.coefficients.get("player_death", -1.0))

    def reward_fn(
        previous_observation: dict[str, Any] | None,
        action_result: dict[str, Any],
        observation: dict[str, Any],
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


def _wave_delta(previous_observation: dict[str, Any] | None, observation: dict[str, Any]) -> float:
    if previous_observation is None:
        return 0.0
    return float(int(observation["wave"]["wave"]) - int(previous_observation["wave"]["wave"]))


def _is_success(observation: dict[str, Any]) -> bool:
    return (
        int(observation["wave"]["wave"]) == 63
        and int(observation["wave"]["remaining"]) == 0
        and int(observation["player"]["hitpoints_current"]) > 0
    )


def _is_death(observation: dict[str, Any]) -> bool:
    return int(observation["player"]["hitpoints_current"]) <= 0
