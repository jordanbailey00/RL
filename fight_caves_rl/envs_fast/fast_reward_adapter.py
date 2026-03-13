from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fight_caves_rl.contracts.reward_feature_schema import (
    REWARD_FEATURE_INDEX,
    REWARD_FEATURE_NAMES,
)
from fight_caves_rl.rewards.registry import RewardConfig, load_reward_config

_REWARD_COEFFICIENT_TO_FEATURE = {
    "wave_progress": "wave_clear",
    "cave_complete": "cave_complete",
    "player_death": "player_death",
    "npc_damage": "damage_dealt",
    "player_damage": "damage_taken",
    "shark_use": "food_used",
    "prayer_potion_use": "prayer_potion_used",
    "step_penalty": "tick_penalty_base",
}


@dataclass(frozen=True)
class FastRewardAdapter:
    config_id: str
    mode: str
    weights: np.ndarray
    unsupported_coefficients: tuple[str, ...]

    @classmethod
    def from_config_id(cls, config_id: str) -> "FastRewardAdapter":
        return cls.from_config(load_reward_config(config_id))

    @classmethod
    def from_config(cls, config: RewardConfig) -> "FastRewardAdapter":
        weights = np.zeros((len(REWARD_FEATURE_NAMES),), dtype=np.float32)
        unsupported: list[str] = []
        for coefficient_name, coefficient_value in config.coefficients.items():
            feature_name = _resolve_feature_name(str(coefficient_name))
            if feature_name is None:
                if float(coefficient_value) != 0.0:
                    unsupported.append(str(coefficient_name))
                continue
            weights[REWARD_FEATURE_INDEX[feature_name]] += np.float32(coefficient_value)
        return cls(
            config_id=config.config_id,
            mode=config.mode,
            weights=weights,
            unsupported_coefficients=tuple(sorted(unsupported)),
        )

    def validate_supported(self) -> None:
        if self.unsupported_coefficients:
            joined = ", ".join(self.unsupported_coefficients)
            raise ValueError(
                "The current V2 fast reward adapter cannot reconstruct unsupported "
                f"coefficients in Python: {joined}. Use V2 reward configs that map "
                "directly onto emitted reward features."
            )

    def weight_batch(self, reward_features: np.ndarray) -> np.ndarray:
        self.validate_supported()
        batch = np.asarray(reward_features, dtype=np.float32)
        if batch.ndim != 2 or batch.shape[1] != self.weights.shape[0]:
            raise ValueError(
                "Fast reward feature layout drift: "
                f"expected (*, {self.weights.shape[0]}), got {batch.shape}."
            )
        return (batch @ self.weights).astype(np.float32, copy=False)


def _resolve_feature_name(coefficient_name: str) -> str | None:
    if coefficient_name in REWARD_FEATURE_INDEX:
        return coefficient_name
    return _REWARD_COEFFICIENT_TO_FEATURE.get(coefficient_name)
