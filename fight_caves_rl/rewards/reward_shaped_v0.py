from __future__ import annotations

from typing import Any

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
        previous_observation: dict[str, Any] | None,
        action_result: dict[str, Any],
        observation: dict[str, Any],
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


def _npc_damage_delta(previous_observation: dict[str, Any], observation: dict[str, Any]) -> float:
    previous = {
        (int(npc["npc_index"]), str(npc["id"])): int(npc["hitpoints_current"])
        for npc in previous_observation.get("npcs", [])
    }
    current = {
        (int(npc["npc_index"]), str(npc["id"])): int(npc["hitpoints_current"])
        for npc in observation.get("npcs", [])
    }
    dealt = 0
    for key, previous_hp in previous.items():
        current_hp = current.get(key)
        if current_hp is None:
            continue
        dealt += max(0, previous_hp - current_hp)
    return float(dealt)


def _player_damage_delta(previous_observation: dict[str, Any], observation: dict[str, Any]) -> float:
    previous_hp = int(previous_observation["player"]["hitpoints_current"])
    current_hp = int(observation["player"]["hitpoints_current"])
    return float(max(0, previous_hp - current_hp))


def _consumable_delta(
    previous_observation: dict[str, Any],
    observation: dict[str, Any],
    key: str,
) -> float:
    previous_value = int(previous_observation["player"]["consumables"][key])
    current_value = int(observation["player"]["consumables"][key])
    return float(max(0, previous_value - current_value))
