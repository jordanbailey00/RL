from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pufferlib.pufferl
from fight_caves_rl.policies.checkpointing import (
    build_checkpoint_metadata,
    load_policy_checkpoint,
    metadata_path_for_checkpoint,
    write_checkpoint_metadata,
)
from fight_caves_rl.policies.mlp import MultiDiscreteMLPPolicy
from fight_caves_rl.puffer.callbacks import SmokeLogger
from fight_caves_rl.puffer.factory import (
    build_policy_episode_env,
    build_puffer_train_config,
    build_train_output_dir,
    load_replay_eval_config,
    load_smoke_train_config,
    make_vecenv,
)
from fight_caves_rl.replay.seed_packs import resolve_seed_pack
from fight_caves_rl.replay.trace_packs import project_observation_for_determinism, semantic_digest


@dataclass(frozen=True)
class TrainRunResult:
    config_id: str
    checkpoint_path: str
    checkpoint_metadata_path: str
    global_step: int
    log_records: int
    puffer_logs: list[dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_smoke_training(
    *,
    config_path: str | Path | None = None,
    total_timesteps: int | None = None,
    data_dir: str | Path | None = None,
) -> TrainRunResult:
    config = load_smoke_train_config(config_path)
    output_dir = build_train_output_dir(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vecenv = make_vecenv(config)
    policy = MultiDiscreteMLPPolicy.from_spaces(
        vecenv.single_observation_space,
        vecenv.single_action_space,
        hidden_size=int(config["policy"]["hidden_size"]),
    )
    puffer_train_config = build_puffer_train_config(
        config,
        data_dir=output_dir,
        total_timesteps=total_timesteps,
    )
    logger = SmokeLogger(args=puffer_train_config)
    trainer = pufferlib.pufferl.PuffeRL(puffer_train_config, vecenv, policy, logger)

    try:
        while trainer.global_step < puffer_train_config["total_timesteps"]:
            trainer.evaluate()
            trainer.train()

        trainer.evaluate()
        trainer.mean_and_log()
        checkpoint_path = Path(trainer.close())
        trainer.logger.close(str(checkpoint_path))
    finally:
        if hasattr(trainer, "vecenv"):
            try:
                trainer.vecenv.close()
            except Exception:
                pass

    metadata = build_checkpoint_metadata(
        train_config_id=str(config["config_id"]),
        policy_id=str(config["policy"]["id"]),
        reward_config_id=str(config["reward_config"]),
        curriculum_config_id=str(config["curriculum_config"]),
    )
    metadata_path = write_checkpoint_metadata(checkpoint_path, metadata)
    puffer_logs = [record.payload for record in logger.records]
    return TrainRunResult(
        config_id=str(config["config_id"]),
        checkpoint_path=str(checkpoint_path),
        checkpoint_metadata_path=str(metadata_path),
        global_step=int(trainer.global_step),
        log_records=len(logger.records),
        puffer_logs=puffer_logs,
    )


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    config = load_replay_eval_config(config_path)
    reward_config_id = "reward_sparse_v0"
    env = build_policy_episode_env({"tick_cap": int(config["max_steps"])}, reward_config_id)
    try:
        policy = MultiDiscreteMLPPolicy.from_spaces(
            env.observation_space,
            env.action_space,
            hidden_size=128,
        )
        metadata = load_policy_checkpoint(Path(checkpoint_path), policy)
        policy.eval()
        seed_pack = resolve_seed_pack(str(config["seed_pack"]))
        per_seed: list[dict[str, Any]] = []
        step_cap = int(max_steps if max_steps is not None else config["max_steps"])

        for seed in seed_pack.seeds:
            observation, reset_info = env.reset(seed=int(seed))
            if env.last_raw_observation is None:
                raise RuntimeError("Expected raw observation after reset.")
            if env.last_reset_info is None:
                raise RuntimeError("Expected raw reset info after reset.")
            episode_start_tick = int(env.last_raw_observation["tick"])
            episode_start_tile = dict(env.last_raw_observation["player"]["tile"])
            terminated = False
            truncated = False
            step_count = 0
            trajectory: list[dict[str, Any]] = []

            while not terminated and not truncated and step_count < step_cap:
                action = greedy_policy_action(policy, observation)
                observation, reward, terminated, truncated, info = env.step(action)
                if env.last_raw_observation is None:
                    raise RuntimeError("Expected raw observation after step.")
                trajectory.append(
                    {
                        "step_index": step_count,
                        "action": np.asarray(action, dtype=np.int64).tolist(),
                        "reward": float(reward),
                        "terminal_reason_code": float(info["terminal_reason_code"]),
                        "semantic_observation": project_observation_for_determinism(
                            env.last_raw_observation,
                            episode_start_tick=episode_start_tick,
                            episode_start_tile=episode_start_tile,
                        ),
                    }
                )
                step_count += 1

            if env.last_raw_observation is None:
                raise RuntimeError("Expected raw observation at end of eval.")
            per_seed.append(
                {
                    "seed": int(seed),
                    "episode_reset_summary": reset_info,
                    "episode_state": env.last_reset_info["episode_state"],
                    "steps_taken": step_count,
                    "terminated": terminated,
                    "truncated": truncated,
                    "trajectory_digest": semantic_digest(trajectory),
                    "final_semantic_observation": project_observation_for_determinism(
                        env.last_raw_observation,
                        episode_start_tick=episode_start_tick,
                        episode_start_tile=episode_start_tile,
                    ),
                }
            )

        return {
            "config_id": str(config["config_id"]),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_metadata_path": str(metadata_path_for_checkpoint(Path(checkpoint_path))),
            "checkpoint_metadata": metadata.to_dict(),
            "seed_pack": str(seed_pack.identity.contract_id),
            "seed_pack_version": int(seed_pack.identity.version),
            "policy_mode": str(config["policy_mode"]),
            "max_steps": step_cap,
            "per_seed": per_seed,
            "summary_digest": semantic_digest(per_seed),
        }
    finally:
        env.close()


def greedy_policy_action(policy: MultiDiscreteMLPPolicy, observation: np.ndarray) -> np.ndarray:
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _values = policy.forward_eval(obs_tensor)
    return np.asarray(
        [int(torch.argmax(head, dim=1).item()) for head in logits],
        dtype=np.int64,
    )
