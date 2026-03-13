from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from fight_caves_rl.utils.paths import repo_root


@dataclass(frozen=True)
class RewardConfig:
    config_id: str
    mode: str
    coefficients: dict[str, float]


def load_reward_config(config_id: str) -> RewardConfig:
    payload = yaml.safe_load(_config_path(config_id).read_text(encoding="utf-8")) or {}
    coefficients = {
        str(key): float(value)
        for key, value in dict(payload.get("coefficients", {})).items()
    }
    return RewardConfig(
        config_id=str(payload["config_id"]),
        mode=str(payload["mode"]),
        coefficients=coefficients,
    )


def resolve_reward_fn(config_id: str):
    config = load_reward_config(config_id)
    if config.config_id == "reward_sparse_v0":
        from fight_caves_rl.rewards.reward_sparse_v0 import build_reward_fn

        return build_reward_fn(config)
    if config.config_id == "reward_shaped_v0":
        from fight_caves_rl.rewards.reward_shaped_v0 import build_reward_fn

        return build_reward_fn(config)
    if config.config_id in {"reward_sparse_v2", "reward_shaped_v2"}:
        raise ValueError(
            f"{config.config_id!r} is a V2 fast-kernel reward config. "
            "It must be consumed through env_backend='v2_fast' reward-feature weighting, "
            "not the V1 observation-based reward-function path."
        )
    raise ValueError(f"Unsupported reward config: {config_id!r}")


def _config_path(config_id: str) -> Path:
    return repo_root() / "configs" / "reward" / f"{config_id}.yaml"
