from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fight_caves_rl.bridge.contracts import HeadlessEpisodeConfig, HeadlessPlayerConfig
from fight_caves_rl.bridge.debug_client import HeadlessDebugClient
from fight_caves_rl.envs.correctness_env import (
    CorrectnessEnvConfig,
    FightCavesCorrectnessEnv,
    infer_terminal_state,
)
from fight_caves_rl.replay.trace_packs import (
    TracePack,
    project_episode_state_for_determinism,
    project_observation_for_determinism,
    project_visible_targets_for_determinism,
    resolve_trace_pack,
    semantic_digest,
    serialize_action,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a deterministic trace rollout as JSON.")
    parser.add_argument("--mode", choices=("wrapper", "raw"), required=True)
    parser.add_argument("--trace-pack", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace_pack = resolve_trace_pack(args.trace_pack)
    seed = int(args.seed if args.seed is not None else trace_pack.default_seed)
    payload = collect_trajectory(mode=args.mode, trace_pack=trace_pack, seed=seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_trajectory(mode: str, trace_pack: TracePack, seed: int) -> dict[str, Any]:
    if mode == "wrapper":
        return collect_wrapper_trajectory(trace_pack=trace_pack, seed=seed)
    if mode == "raw":
        return collect_raw_trajectory(trace_pack=trace_pack, seed=seed)
    raise ValueError(f"Unsupported trajectory mode: {mode}")


def collect_wrapper_trajectory(trace_pack: TracePack, seed: int) -> dict[str, Any]:
    tick_cap = int(trace_pack.tick_cap if trace_pack.tick_cap is not None else 20_000)
    env = FightCavesCorrectnessEnv(
        CorrectnessEnvConfig(start_wave=trace_pack.start_wave, tick_cap=tick_cap)
    )
    try:
        initial_observation, reset_info = env.reset(seed=seed)
        episode_start_tick = int(initial_observation["tick"])
        episode_start_tile = initial_observation["player"]["tile"]
        steps: list[dict[str, Any]] = []

        for index, trace_step in enumerate(trace_pack.steps):
            observation, reward, terminated, truncated, step_info = env.step(trace_step.action)
            steps.append(
                {
                    "step_index": index,
                    "action": serialize_action(trace_step.action),
                    "observation": observation,
                    "semantic_observation": project_observation_for_determinism(
                        observation,
                        episode_start_tick=episode_start_tick,
                        episode_start_tile=episode_start_tile,
                    ),
                    "action_result": step_info["action_result"],
                    "visible_targets": step_info["visible_targets"],
                    "semantic_visible_targets": project_visible_targets_for_determinism(
                        step_info["visible_targets"],
                        episode_start_tile=episode_start_tile,
                    ),
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "terminal_reason": step_info["terminal_reason"],
                }
            )
            if terminated or truncated:
                break

        return _build_trajectory_payload(
            mode="wrapper",
            trace_pack=trace_pack,
            seed=seed,
            episode_state=reset_info["episode_state"],
            initial_observation=initial_observation,
            episode_start_tick=episode_start_tick,
            episode_start_tile=episode_start_tile,
            steps=steps,
        )
    finally:
        env.close()


def collect_raw_trajectory(trace_pack: TracePack, seed: int) -> dict[str, Any]:
    tick_cap = int(trace_pack.tick_cap if trace_pack.tick_cap is not None else 20_000)
    client = HeadlessDebugClient.create()
    player = client.create_player_slot(HeadlessPlayerConfig(account_name="rl_raw_trajectory"))
    try:
        episode_state = client.reset_episode(
            player,
            HeadlessEpisodeConfig(seed=seed, start_wave=trace_pack.start_wave),
        )
        initial_observation = client.observe(player)
        episode_start_tick = int(initial_observation["tick"])
        episode_start_tile = initial_observation["player"]["tile"]
        steps: list[dict[str, Any]] = []

        for index, trace_step in enumerate(trace_pack.steps):
            snapshot = client.step_once(player, trace_step.action)
            terminated, truncated, terminal_reason = infer_terminal_state(
                observation=snapshot.observation,
                episode_start_tick=episode_start_tick,
                tick_cap=tick_cap,
            )
            steps.append(
                {
                    "step_index": index,
                    "action": serialize_action(trace_step.action),
                    "observation": snapshot.observation,
                    "semantic_observation": project_observation_for_determinism(
                        snapshot.observation,
                        episode_start_tick=episode_start_tick,
                        episode_start_tile=episode_start_tile,
                    ),
                    "action_result": snapshot.action_result,
                    "visible_targets": snapshot.visible_targets,
                    "semantic_visible_targets": project_visible_targets_for_determinism(
                        snapshot.visible_targets,
                        episode_start_tile=episode_start_tile,
                    ),
                    "reward": 0.0,
                    "terminated": terminated,
                    "truncated": truncated,
                    "terminal_reason": terminal_reason,
                }
            )
            if terminated or truncated:
                break

        return _build_trajectory_payload(
            mode="raw",
            trace_pack=trace_pack,
            seed=seed,
            episode_state=episode_state,
            initial_observation=initial_observation,
            episode_start_tick=episode_start_tick,
            episode_start_tile=episode_start_tile,
            steps=steps,
        )
    finally:
        client.close()


def _build_trajectory_payload(
    *,
    mode: str,
    trace_pack: TracePack,
    seed: int,
    episode_state: dict[str, Any],
    initial_observation: dict[str, Any],
    episode_start_tick: int,
    episode_start_tile: dict[str, Any],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    semantic_initial = project_observation_for_determinism(
        initial_observation,
        episode_start_tick=episode_start_tick,
        episode_start_tile=episode_start_tile,
    )
    semantic_episode_state = project_episode_state_for_determinism(episode_state)
    digest_payload = {
        "trace_pack": trace_pack.identity.contract_id,
        "seed": seed,
        "initial_observation": semantic_initial,
        "steps": [
            {
                "action": step["action"],
                "semantic_observation": step["semantic_observation"],
                "action_result": step["action_result"],
                "semantic_visible_targets": step["semantic_visible_targets"],
                "terminated": step["terminated"],
                "truncated": step["truncated"],
                "terminal_reason": step["terminal_reason"],
            }
            for step in steps
        ],
    }
    final_observation = steps[-1]["semantic_observation"] if steps else semantic_initial
    final_step = steps[-1] if steps else None
    return {
        "mode": mode,
        "trace_pack": trace_pack.identity.contract_id,
        "trace_pack_version": trace_pack.identity.version,
        "seed": seed,
        "start_wave": trace_pack.start_wave,
        "episode_state": episode_state,
        "semantic_episode_state": semantic_episode_state,
        "initial_observation": initial_observation,
        "semantic_initial_observation": semantic_initial,
        "steps": steps,
        "summary": {
            "total_steps": len(steps),
            "completed_all_steps": len(steps) == len(trace_pack.steps),
            "final_relative_tick": int(final_observation["tick"]),
            "final_wave": int(final_observation["wave"]["wave"]),
            "final_remaining": int(final_observation["wave"]["remaining"]),
            "final_terminal_reason": None if final_step is None else final_step["terminal_reason"],
            "final_terminated": False if final_step is None else final_step["terminated"],
            "final_truncated": False if final_step is None else final_step["truncated"],
            "semantic_digest": semantic_digest(digest_payload),
        },
    }


if __name__ == "__main__":
    main()
