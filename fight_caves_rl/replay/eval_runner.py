from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fight_caves_rl.envs.puffer_encoding import build_policy_action_space, build_policy_observation_space
from fight_caves_rl.logging.metrics import build_eval_summary_metrics
from fight_caves_rl.logging.wandb_client import WandbRunLogger
from fight_caves_rl.manifests.run_manifest import build_eval_run_manifest, write_run_manifest
from fight_caves_rl.policies.checkpointing import (
    load_policy_checkpoint,
    metadata_path_for_checkpoint,
)
from fight_caves_rl.policies.mlp import MultiDiscreteMLPPolicy
from fight_caves_rl.puffer.factory import (
    build_policy_episode_env,
    load_replay_eval_config,
)
from fight_caves_rl.replay.replay_export import (
    build_replay_episode,
    build_replay_pack,
    write_replay_pack,
)
from fight_caves_rl.replay.replay_index import build_replay_index, write_replay_index
from fight_caves_rl.replay.seed_packs import resolve_seed_pack
from fight_caves_rl.replay.trace_packs import project_observation_for_determinism, semantic_digest
from fight_caves_rl.utils.config import load_bootstrap_config
from fight_caves_rl.utils.paths import repo_root


def run_replay_eval(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    bootstrap_config = load_bootstrap_config()
    config = load_replay_eval_config(config_path)
    logger = WandbRunLogger(
        config=bootstrap_config,
        run_kind="eval",
        config_id=str(config["config_id"]),
        tags=(str(config["config_id"]), "replay-eval"),
    )
    try:
        policy = MultiDiscreteMLPPolicy.from_spaces(
            build_policy_observation_space(),
            build_policy_action_space(),
            hidden_size=128,
        )
        checkpoint = Path(checkpoint_path)
        metadata = load_policy_checkpoint(checkpoint, policy)
        reward_config_id = (
            metadata.reward_config_id
            if str(config.get("reward_config", "use_checkpoint")) == "use_checkpoint"
            else str(config["reward_config"])
        )
        curriculum_config_id = str(config.get("curriculum_config", "curriculum_disabled_v0"))
        step_cap = int(max_steps if max_steps is not None else config["max_steps"])
        env = build_policy_episode_env(
            {"tick_cap": step_cap},
            reward_config_id,
            curriculum_config_id,
        )
        policy.eval()
        seed_pack = resolve_seed_pack(str(config["seed_pack"]))
        per_seed: list[dict[str, Any]] = []
        replay_episodes = []
        replay_step_cadence = int(config.get("replay_step_cadence", 1))

        if replay_step_cadence <= 0:
            raise ValueError(
                f"replay_step_cadence must be >= 1, got {replay_step_cadence}."
            )

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
            terminal_reason: str | None = None
            step_count = 0
            trajectory: list[dict[str, Any]] = []

            while not terminated and not truncated and step_count < step_cap:
                action = greedy_policy_action(policy, observation)
                observation, reward, terminated, truncated, info = env.step(action)
                if env.last_raw_observation is None:
                    raise RuntimeError("Expected raw observation after step.")
                terminal_reason = (
                    env.last_step_info["terminal_reason"]
                    if env.last_step_info is not None
                    else None
                )
                trajectory.append(
                    {
                        "step_index": step_count,
                        "action": np.asarray(action, dtype=np.int64).tolist(),
                        "reward": float(reward),
                        "terminal_reason_code": float(info["terminal_reason_code"]),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "terminal_reason": terminal_reason,
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
            final_semantic_observation = project_observation_for_determinism(
                env.last_raw_observation,
                episode_start_tick=episode_start_tick,
                episode_start_tile=episode_start_tile,
            )
            trajectory_digest = semantic_digest(trajectory)
            per_seed_entry = {
                "seed": int(seed),
                "episode_reset_summary": reset_info,
                "episode_state": env.last_reset_info["episode_state"],
                "steps_taken": step_count,
                "terminated": terminated,
                "truncated": truncated,
                "terminal_reason": terminal_reason,
                "trajectory_digest": trajectory_digest,
                "final_semantic_observation": final_semantic_observation,
            }
            per_seed.append(per_seed_entry)
            replay_episodes.append(
                build_replay_episode(
                    seed=int(seed),
                    episode_reset_summary=reset_info,
                    episode_state=env.last_reset_info["episode_state"],
                    steps_taken=step_count,
                    terminated=terminated,
                    truncated=truncated,
                    terminal_reason=terminal_reason,
                    trajectory_digest=trajectory_digest,
                    final_semantic_observation=final_semantic_observation,
                    full_steps=trajectory,
                    replay_step_cadence=replay_step_cadence,
                )
            )

        run_artifact_dir = build_eval_output_dir(str(config["config_id"]), str(logger.run_id))
        run_artifact_dir.mkdir(parents=True, exist_ok=True)
        eval_summary_path = run_artifact_dir / "eval_summary.json"
        replay_pack_path = run_artifact_dir / "replay_pack.json"
        replay_index_path = run_artifact_dir / "replay_index.json"
        manifest_path = run_artifact_dir / "run_manifest.json"
        checkpoint_metadata_path = metadata_path_for_checkpoint(checkpoint)
        summary_digest = semantic_digest(per_seed)
        replay_pack = build_replay_pack(
            config_id=str(config["config_id"]),
            checkpoint_path=checkpoint,
            checkpoint_metadata_path=checkpoint_metadata_path,
            checkpoint_metadata=metadata.to_dict(),
            seed_pack=str(seed_pack.identity.contract_id),
            seed_pack_version=int(seed_pack.identity.version),
            policy_mode=str(config["policy_mode"]),
            reward_config_id=reward_config_id,
            curriculum_config_id=curriculum_config_id,
            replay_step_cadence=replay_step_cadence,
            summary_digest=summary_digest,
            episodes=replay_episodes,
        )
        write_replay_pack(replay_pack_path, replay_pack)
        replay_index = build_replay_index(
            replay_pack=replay_pack,
            replay_pack_filename=replay_pack_path.name,
            eval_summary_filename=eval_summary_path.name,
            checkpoint_format_id=metadata.checkpoint_format_id,
            checkpoint_format_version=metadata.checkpoint_format_version,
            policy_id=metadata.policy_id,
        )
        write_replay_index(replay_index_path, replay_index)
        payload = {
            "config_id": str(config["config_id"]),
            "checkpoint_path": str(checkpoint),
            "checkpoint_metadata_path": str(checkpoint_metadata_path),
            "checkpoint_metadata": metadata.to_dict(),
            "seed_pack": str(seed_pack.identity.contract_id),
            "seed_pack_version": int(seed_pack.identity.version),
            "policy_mode": str(config["policy_mode"]),
            "max_steps": step_cap,
            "replay_pack_schema_id": replay_pack.schema_id,
            "replay_pack_schema_version": replay_pack.schema_version,
            "replay_index_schema_id": replay_index.schema_id,
            "replay_index_schema_version": replay_index.schema_version,
            "replay_step_cadence": replay_step_cadence,
            "per_seed": per_seed,
            "summary_digest": summary_digest,
            "wandb_run_id": str(logger.run_id),
            "eval_summary_path": str(eval_summary_path),
            "replay_pack_path": str(replay_pack_path),
            "replay_index_path": str(replay_index_path),
            "run_manifest_path": str(manifest_path),
        }
        eval_summary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        eval_summary_record = logger.build_artifact_record(
            category="eval_summary",
            path=eval_summary_path,
        )
        replay_pack_record = logger.build_artifact_record(
            category="replay_pack",
            path=replay_pack_path,
        )
        replay_index_record = logger.build_artifact_record(
            category="replay_index",
            path=replay_index_path,
        )
        manifest_record = logger.build_artifact_record(
            category="run_manifest",
            path=manifest_path,
        )
        manifest = build_eval_run_manifest(
            bootstrap_config=bootstrap_config,
            config_id=str(config["config_id"]),
            run_id=str(logger.run_id),
            run_output_dir=run_artifact_dir,
            reward_config_id=reward_config_id,
            curriculum_config_id=curriculum_config_id,
            policy_id=metadata.policy_id,
            env_count=1,
            wandb_tags=logger.effective_tags,
            checkpoint_metadata=metadata,
            checkpoint_path=checkpoint,
            checkpoint_metadata_path=checkpoint_metadata_path,
            seed_pack=str(seed_pack.identity.contract_id),
            seed_pack_version=int(seed_pack.identity.version),
            summary_digest=str(payload["summary_digest"]),
            artifacts=(
                eval_summary_record,
                replay_pack_record,
                replay_index_record,
                manifest_record,
            ),
        )
        write_run_manifest(manifest_path, manifest)
        logger.update_config(manifest.to_dict())
        logger.log_metrics(
            build_eval_summary_metrics(payload),
            step=int(step_cap),
        )
        logger.log_artifact(
            eval_summary_record,
            metadata={
                "run_kind": "eval",
                "config_id": str(config["config_id"]),
                "artifact_category": "eval_summary",
            },
        )
        logger.log_artifact(
            replay_pack_record,
            metadata={
                "run_kind": "eval",
                "config_id": str(config["config_id"]),
                "artifact_category": "replay_pack",
                "replay_pack_schema_id": replay_pack.schema_id,
                "replay_pack_schema_version": replay_pack.schema_version,
                "replay_step_cadence": replay_step_cadence,
            },
        )
        logger.log_artifact(
            replay_index_record,
            metadata={
                "run_kind": "eval",
                "config_id": str(config["config_id"]),
                "artifact_category": "replay_index",
                "replay_index_schema_id": replay_index.schema_id,
                "replay_index_schema_version": replay_index.schema_version,
            },
        )
        logger.log_artifact(
            manifest_record,
            metadata={
                "run_kind": "eval",
                "config_id": str(config["config_id"]),
                "artifact_category": "run_manifest",
            },
        )
        logger.finish()
        payload["artifacts"] = [record.to_dict() for record in logger.artifact_records]
        return payload
    finally:
        if "env" in locals():
            env.close()
        logger.finish()


def greedy_policy_action(policy: MultiDiscreteMLPPolicy, observation: np.ndarray) -> np.ndarray:
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _values = policy.forward_eval(obs_tensor)
    return np.asarray(
        [int(torch.argmax(head, dim=1).item()) for head in logits],
        dtype=np.int64,
    )


def build_eval_output_dir(config_id: str, run_id: str) -> Path:
    return repo_root() / "artifacts" / "eval" / config_id / run_id
