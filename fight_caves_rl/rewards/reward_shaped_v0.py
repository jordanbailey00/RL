from __future__ import annotations

from typing import Any

import numpy as np

from fight_caves_rl.envs.observation_views import (
    observation_consumable_value,
    observation_npc_health_projection,
    observation_player_hitpoints_current,
)
from fight_caves_rl.rewards.registry import RewardConfig
from fight_caves_rl.rewards.reward_sparse_v0 import build_reward_fn as build_sparse_reward_fn


def build_reward_fn(config: RewardConfig):
    sparse_reward = build_sparse_reward_fn(config)
    npc_damage = float(config.coefficients.get("npc_damage", 0.02))
    player_damage = float(config.coefficients.get("player_damage", -0.02))
    shark_use = float(config.coefficients.get("shark_use", -0.05))
    prayer_potion_use = float(config.coefficients.get("prayer_potion_use", -0.05))
    ammo_use = float(config.coefficients.get("ammo_use", -0.001))
    step_penalty = float(config.coefficients.get("step_penalty", -0.0005))

    def reward_fn(
        previous_observation: dict[str, Any] | np.ndarray | None,
        action_result: dict[str, Any],
        observation: dict[str, Any] | np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> float:
        reward = float(
            sparse_reward(
                previous_observation,
                action_result,
                observation,
                terminated,
                truncated,
            )
        )
        if previous_observation is None:
            return reward

        reward += npc_damage * _npc_damage_delta(previous_observation, observation)
        reward += player_damage * _player_damage_delta(previous_observation, observation)
        reward += shark_use * _consumable_delta(previous_observation, observation, "shark_count")
        reward += prayer_potion_use * _consumable_delta(
            previous_observation,
            observation,
            "prayer_potion_dose_count",
        )
        reward += ammo_use * _consumable_delta(previous_observation, observation, "ammo_count")
        reward += step_penalty
        return float(reward)

    return reward_fn

def _npc_damage_delta(
    previous_observation: dict[str, Any] | np.ndarray,
    observation: dict[str, Any] | np.ndarray,
) -> float:
    previous = observation_npc_health_projection(previous_observation)
    current = observation_npc_health_projection(observation)
    dealt = 0
    for key, previous_hp in previous.items():
        current_hp = current.get(key)
        if current_hp is None:
            continue
        dealt += max(0, previous_hp - current_hp)
    return float(dealt)


def _player_damage_delta(
    previous_observation: dict[str, Any] | np.ndarray,
    observation: dict[str, Any] | np.ndarray,
) -> float:
    previous_hp = observation_player_hitpoints_current(previous_observation)
    current_hp = observation_player_hitpoints_current(observation)
    return float(max(0, previous_hp - current_hp))


def _consumable_delta(
    previous_observation: dict[str, Any] | np.ndarray,
    observation: dict[str, Any] | np.ndarray,
    key: str,
) -> float:
    previous_value = observation_consumable_value(previous_observation, key)
    current_value = observation_consumable_value(observation, key)
    return float(max(0, previous_value - current_value))
