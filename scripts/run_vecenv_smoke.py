from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from fight_caves_rl.puffer.factory import load_smoke_train_config, make_vecenv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live vecenv smoke checks in an isolated subprocess.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train/train_baseline_v0.yaml"),
    )
    parser.add_argument(
        "--mode",
        choices=("reset-step", "long-run"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_smoke_train_config(args.config)
    if args.mode == "long-run":
        config = deepcopy(config)
        config["num_envs"] = 4
        config["env"]["tick_cap"] = 16

    payload = _run_reset_step(config) if args.mode == "reset-step" else _run_long_run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_reset_step(config: dict[str, object]) -> dict[str, object]:
    vecenv = make_vecenv(config, backend="embedded")
    try:
        env_count = int(config["num_envs"])
        vecenv.async_reset(seed=int(config["train"]["seed"]))
        observations, rewards, terminals, truncations, teacher_actions, infos, agent_ids, masks = vecenv.recv()
        actions = np.zeros((env_count, len(vecenv.single_action_space.nvec)), dtype=np.int32)
        vecenv.send(actions)
        next_observations, next_rewards, next_terminals, next_truncations, _ta, next_infos, _ids, _masks = vecenv.recv()
        return {
            "mode": "reset-step",
            "env_count": env_count,
            "initial_observation_shape": list(observations.shape),
            "initial_reward_shape": list(rewards.shape),
            "initial_terminal_shape": list(terminals.shape),
            "initial_truncation_shape": list(truncations.shape),
            "teacher_action_shape": list(teacher_actions.shape),
            "info_count": len(infos),
            "agent_ids": [int(value) for value in agent_ids.tolist()],
            "mask_dtype": str(masks.dtype),
            "next_observation_shape": list(next_observations.shape),
            "next_reward_shape": list(next_rewards.shape),
            "next_terminal_shape": list(next_terminals.shape),
            "next_truncation_shape": list(next_truncations.shape),
            "next_info_count": len(next_infos),
            "next_events": [str(info["vecenv_event"]) for info in next_infos],
        }
    finally:
        vecenv.close()


def _run_long_run(config: dict[str, object]) -> dict[str, object]:
    vecenv = make_vecenv(config, backend="embedded")
    try:
        env_count = int(config["num_envs"])
        vecenv.async_reset(seed=int(config["train"]["seed"]))
        vecenv.recv()
        actions = np.zeros((env_count, len(vecenv.single_action_space.nvec)), dtype=np.int32)
        last_reward_shape: list[int] = []
        for _ in range(48):
            vecenv.send(actions)
            observations, rewards, terminals, truncations, _ta, infos, _ids, masks = vecenv.recv()
            if not np.isfinite(observations).all():
                raise AssertionError("Vecenv observations contained non-finite values.")
            if not np.isfinite(rewards).all():
                raise AssertionError("Vecenv rewards contained non-finite values.")
            if len(infos) != env_count:
                raise AssertionError(f"Expected {env_count} infos, got {len(infos)}.")
            if terminals.shape != (env_count,) or truncations.shape != (env_count,) or masks.shape != (env_count,):
                raise AssertionError("Vecenv terminal/truncation/mask shapes drifted during long-run smoke.")
            last_reward_shape = list(rewards.shape)
        return {
            "mode": "long-run",
            "env_count": env_count,
            "episodes_started": [int(value) for value in vecenv.episode_counts.tolist()],
            "last_reward_shape": last_reward_shape,
        }
    finally:
        vecenv.close()


if __name__ == "__main__":
    main()
