from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fight_caves_rl.bridge.contracts import HeadlessEpisodeConfig, HeadlessPlayerConfig
from fight_caves_rl.bridge.debug_client import HeadlessDebugClient
from fight_caves_rl.envs.correctness_env import FightCavesCorrectnessEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a single-step fight-caves trace as JSON.")
    parser.add_argument("--mode", choices=("wrapper", "raw"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--action", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = collect_trace(mode=args.mode, seed=args.seed, action=args.action)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_trace(mode: str, seed: int, action: int) -> dict[str, Any]:
    if mode == "wrapper":
        return collect_wrapper_trace(seed=seed, action=action)
    if mode == "raw":
        return collect_raw_trace(seed=seed, action=action)
    raise ValueError(f"Unsupported trace mode: {mode}")


def collect_wrapper_trace(seed: int, action: int) -> dict[str, Any]:
    env = FightCavesCorrectnessEnv()
    try:
        observation, reset_info = env.reset(seed=seed)
        next_observation, reward, terminated, truncated, step_info = env.step(action)
        return {
            "mode": "wrapper",
            "seed": seed,
            "action": action,
            "episode_state": reset_info["episode_state"],
            "bridge_handshake": reset_info["bridge_handshake"],
            "observation": observation,
            "next_observation": next_observation,
            "action_result": step_info["action_result"],
            "visible_targets": step_info["visible_targets"],
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason": step_info["terminal_reason"],
            "terminal_reason_inferred": step_info["terminal_reason_inferred"],
            "episode_steps": step_info["episode_steps"],
        }
    finally:
        env.close()


def collect_raw_trace(seed: int, action: int) -> dict[str, Any]:
    client = HeadlessDebugClient.create()
    player = client.create_player_slot(HeadlessPlayerConfig(account_name="rl_raw_trace"))
    try:
        episode_state = client.reset_episode(player, HeadlessEpisodeConfig(seed=seed))
        observation = client.observe(player)
        snapshot = client.step_once(player, action)
        return {
            "mode": "raw",
            "seed": seed,
            "action": action,
            "episode_state": episode_state,
            "bridge_handshake": dict(client.handshake.values),
            "observation": observation,
            "next_observation": snapshot.observation,
            "action_result": snapshot.action_result,
            "visible_targets": snapshot.visible_targets,
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "terminal_reason": None,
            "terminal_reason_inferred": False,
            "episode_steps": 1,
        }
    finally:
        client.close()


if __name__ == "__main__":
    main()
