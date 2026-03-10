from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
import yaml

import pufferlib.emulation
from fight_caves_rl.curriculum.registry import build_curriculum
from fight_caves_rl.envs.correctness_env import CorrectnessEnvConfig, FightCavesCorrectnessEnv
from fight_caves_rl.envs.puffer_encoding import (
    POLICY_MAX_VISIBLE_NPCS,
    build_policy_action_space,
    build_policy_observation_space,
    decode_action_from_policy,
    encode_observation_for_policy,
)
from fight_caves_rl.envs.schema import HEADLESS_ACTION_REJECT_REASONS
from fight_caves_rl.envs.subprocess_vector_env import (
    SubprocessHeadlessBatchVecEnv,
    SubprocessHeadlessBatchVecEnvConfig,
)
from fight_caves_rl.envs.shared_memory_transport import PIPE_PICKLE_TRANSPORT_MODE
from fight_caves_rl.envs.vector_env import HeadlessBatchVecEnv, HeadlessBatchVecEnvConfig
from fight_caves_rl.rewards.registry import resolve_reward_fn
from fight_caves_rl.utils.paths import repo_root

DEFAULT_SMOKE_TRAIN_CONFIG: dict[str, Any] = {
    "config_id": "smoke_ppo_v0",
    "trainer": "pufferlib",
    "policy": {
        "id": "mlp_v0",
        "hidden_size": 128,
    },
    "reward_config": "reward_sparse_v0",
    "curriculum_config": "curriculum_disabled_v0",
    "num_envs": 1,
    "env": {
        "start_wave": 1,
        "ammo": 1000,
        "prayer_potions": 8,
        "sharks": 20,
        "tick_cap": 256,
        "include_future_leakage": False,
        "info_payload_mode": "full",
        "account_name_prefix": "rl_puffer_smoke",
    },
    "train": {
        "seed": 11_001,
        "torch_deterministic": True,
        "device": "cpu",
        "cpu_offload": False,
        "compile": False,
        "compile_mode": "default",
        "compile_fullgraph": False,
        "optimizer": "adam",
        "learning_rate": 3.0e-4,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_eps": 1.0e-5,
        "precision": "float32",
        "bptt_horizon": 4,
        "batch_size": 4,
        "minibatch_size": 4,
        "max_minibatch_size": 4,
        "total_timesteps": 8,
        "update_epochs": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "vtrace_rho_clip": 1.0,
        "vtrace_c_clip": 1.0,
        "clip_coef": 0.2,
        "vf_clip_coef": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "prio_alpha": 0.0,
        "prio_beta0": 1.0,
        "anneal_lr": False,
        "checkpoint_interval": 1,
        "use_rnn": False,
    },
    "logging": {
        "dashboard": True,
    },
}

DEFAULT_REPLAY_EVAL_CONFIG: dict[str, Any] = {
    "config_id": "replay_eval_v0",
    "seed_pack": "bootstrap_smoke",
    "reward_config": "use_checkpoint",
    "curriculum_config": "curriculum_disabled_v0",
    "policy_mode": "greedy",
    "max_steps": 64,
    "replay_step_cadence": 1,
}

TERMINAL_REASON_TO_CODE = {
    None: 0,
    "player_death": 1,
    "cave_complete": 2,
    "max_tick_cap": 3,
}
REJECTION_REASON_TO_CODE = {
    None: 0,
    **{reason: index + 1 for index, reason in enumerate(HEADLESS_ACTION_REJECT_REASONS)},
}


class FightCavesPufferGymEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env_config: Mapping[str, Any] | None = None,
        reward_config_id: str = "reward_sparse_v0",
        curriculum_config_id: str = "curriculum_disabled_v0",
        env_index: int = 0,
    ) -> None:
        super().__init__()
        env_settings = dict(env_config or {})
        account_name_prefix = str(env_settings.get("account_name_prefix", "rl_puffer_smoke"))
        correctness_config = CorrectnessEnvConfig(
            account_name=f"{account_name_prefix}_{env_index}",
            start_wave=int(env_settings.get("start_wave", 1)),
            ammo=int(env_settings.get("ammo", 1000)),
            prayer_potions=int(env_settings.get("prayer_potions", 8)),
            sharks=int(env_settings.get("sharks", 20)),
            tick_cap=int(env_settings.get("tick_cap", 256)),
            include_future_leakage=bool(env_settings.get("include_future_leakage", False)),
        )
        self._reward_config_id = reward_config_id
        self._curriculum = build_curriculum(curriculum_config_id)
        self._env_index = int(env_index)
        self._episodes_started = 0
        self._env = FightCavesCorrectnessEnv(
            config=correctness_config,
            reward_fn=resolve_reward_fn(reward_config_id),
        )
        self.observation_space = build_policy_observation_space()
        self.action_space = build_policy_action_space()
        self.last_raw_observation: dict[str, Any] | None = None
        self.last_reset_info: dict[str, Any] | None = None
        self.last_step_info: dict[str, Any] | None = None
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        super().reset(seed=seed)
        reset_options = dict(
            self._curriculum.reset_overrides(
                slot_index=self._env_index,
                episode_index=self._episodes_started,
            )
        )
        if options:
            reset_options.update(dict(options))
        observation, info = self._env.reset(seed=seed, options=reset_options)
        self._episodes_started += 1
        self.last_raw_observation = observation
        self.last_reset_info = info
        self.last_step_info = None
        self._episode_return = 0.0
        self._episode_length = 0
        return encode_observation_for_policy(observation), {
            "episode_seed": float(observation["episode_seed"]),
            "wave": float(observation["wave"]["wave"]),
            "remaining": float(observation["wave"]["remaining"]),
        }

    def step(
        self,
        action: np.ndarray | list[int] | tuple[int, ...],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        normalized = decode_action_from_policy(action)
        observation, reward, terminated, truncated, step_info = self._env.step(normalized)
        self.last_raw_observation = observation
        self.last_step_info = step_info
        self._episode_return += float(reward)
        self._episode_length += 1

        action_result = step_info["action_result"]
        terminal_reason = step_info["terminal_reason"]
        info = {
            "action_id": float(normalized.action_id),
            "visible_target_count": float(len(step_info["visible_targets"])),
            "wave": float(observation["wave"]["wave"]),
            "remaining": float(observation["wave"]["remaining"]),
            "reward": float(reward),
            "action_applied": float(int(bool(action_result["action_applied"]))),
            "rejection_reason_code": float(
                REJECTION_REASON_TO_CODE[action_result["rejection_reason"]]
            ),
            "terminal_reason_code": float(TERMINAL_REASON_TO_CODE[terminal_reason]),
            "episode_steps": float(step_info["episode_steps"]),
            "episode_return": float(self._episode_return),
            "episode_length": float(self._episode_length),
            "terminated": float(int(terminated)),
            "truncated": float(int(truncated)),
        }
        return encode_observation_for_policy(observation), float(reward), terminated, truncated, info

    def close(self) -> None:
        self._env.close()


def load_smoke_train_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_SMOKE_TRAIN_CONFIG)
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded:
            config = _deep_merge(config, loaded)
    return config


def load_replay_eval_config(path: str | Path | None = None) -> dict[str, Any]:
    config = deepcopy(DEFAULT_REPLAY_EVAL_CONFIG)
    if path is not None:
        loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if loaded:
            config = _deep_merge(config, loaded)
    return config


def make_vecenv(config: Mapping[str, Any], *, backend: str = "embedded"):
    env_config = dict(config.get("env", {}))
    reward_config_id = str(config["reward_config"])
    curriculum = build_curriculum(str(config["curriculum_config"]))
    batch_config = HeadlessBatchVecEnvConfig(
        env_count=int(config["num_envs"]),
        account_name_prefix=str(env_config.get("account_name_prefix", "rl_vecenv")),
        start_wave=int(env_config.get("start_wave", 1)),
        ammo=int(env_config.get("ammo", 1000)),
        prayer_potions=int(env_config.get("prayer_potions", 8)),
        sharks=int(env_config.get("sharks", 20)),
        tick_cap=int(env_config.get("tick_cap", 20_000)),
        include_future_leakage=bool(env_config.get("include_future_leakage", False)),
        info_payload_mode=str(env_config.get("info_payload_mode", "full")),
        reset_options_provider=curriculum.reset_overrides,
    )
    if backend == "embedded":
        return HeadlessBatchVecEnv(
            batch_config,
            reward_fn=resolve_reward_fn(reward_config_id),
        )
    if backend == "subprocess":
        return SubprocessHeadlessBatchVecEnv(
            SubprocessHeadlessBatchVecEnvConfig(
                env_count=int(config["num_envs"]),
                reward_config_id=reward_config_id,
                curriculum_config_id=str(config["curriculum_config"]),
                transport_mode=str(
                    env_config.get("subprocess_transport_mode", PIPE_PICKLE_TRANSPORT_MODE)
                ),
                account_name_prefix=batch_config.account_name_prefix,
                start_wave=batch_config.start_wave,
                ammo=batch_config.ammo,
                prayer_potions=batch_config.prayer_potions,
                sharks=batch_config.sharks,
                tick_cap=batch_config.tick_cap,
                include_future_leakage=batch_config.include_future_leakage,
                info_payload_mode=batch_config.info_payload_mode,
                bootstrap=batch_config.bootstrap,
            )
        )
    raise ValueError(f"Unsupported vecenv backend: {backend!r}")


def build_train_output_dir(
    config_id: str,
    data_dir: str | Path | None = None,
) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    return repo_root() / "artifacts" / "train" / str(config_id)


def build_puffer_train_config(
    config: Mapping[str, Any],
    *,
    data_dir: Path,
    total_timesteps: int | None = None,
) -> dict[str, Any]:
    train_config = deepcopy(dict(config["train"]))
    if total_timesteps is not None:
        train_config["total_timesteps"] = int(total_timesteps)
    train_config["data_dir"] = str(data_dir)
    train_config["env"] = str(config["config_id"])
    return train_config


def build_policy_episode_env(
    env_config: Mapping[str, Any],
    reward_config_id: str,
    curriculum_config_id: str = "curriculum_disabled_v0",
) -> FightCavesPufferGymEnv:
    return FightCavesPufferGymEnv(
        env_config=env_config,
        reward_config_id=reward_config_id,
        curriculum_config_id=curriculum_config_id,
        env_index=0,
    )


def scripted_action_space_shape() -> tuple[int, ...]:
    return tuple(int(value) for value in build_policy_action_space().nvec)


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged
