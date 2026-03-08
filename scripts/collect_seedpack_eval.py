from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fight_caves_rl.envs.correctness_env import FightCavesCorrectnessEnv
from fight_caves_rl.replay.seed_packs import resolve_seed_pack
from fight_caves_rl.replay.trace_packs import project_observation_for_determinism, semantic_digest


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a deterministic seed-pack evaluation summary as JSON.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-pack", required=True)
    parser.add_argument("--max-steps", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = collect_seedpack_eval(
        checkpoint_path=args.checkpoint,
        seed_pack_id=args.seed_pack,
        max_steps=args.max_steps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_seedpack_eval(
    *,
    checkpoint_path: Path,
    seed_pack_id: str,
    max_steps: int,
) -> dict[str, Any]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    policy_id = str(checkpoint["policy_id"])
    seed_pack = resolve_seed_pack(seed_pack_id)
    env = FightCavesCorrectnessEnv()
    try:
        per_seed: list[dict[str, Any]] = []
        for seed in seed_pack.seeds:
            observation, info = env.reset(seed=int(seed))
            episode_start_tick = int(observation["tick"])
            episode_start_tile = observation["player"]["tile"]
            step_records: list[dict[str, Any]] = []
            terminated = False
            truncated = False
            terminal_reason = None
            steps_taken = 0
            while not terminated and not truncated and steps_taken < max_steps:
                action = select_policy_action(policy_id, observation)
                observation, reward, terminated, truncated, step_info = env.step(action)
                steps_taken += 1
                terminal_reason = step_info["terminal_reason"]
                step_records.append(
                    {
                        "step_index": steps_taken - 1,
                        "action": int(action) if isinstance(action, int) else dict(action),
                        "semantic_observation": project_observation_for_determinism(
                            observation,
                            episode_start_tick=episode_start_tick,
                            episode_start_tile=episode_start_tile,
                        ),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                        "terminal_reason": terminal_reason,
                    }
                )

            per_seed.append(
                {
                    "seed": int(seed),
                    "episode_state": info["episode_state"],
                    "steps_taken": steps_taken,
                    "terminated": terminated,
                    "truncated": truncated,
                    "terminal_reason": terminal_reason,
                    "final_semantic_observation": project_observation_for_determinism(
                        observation,
                        episode_start_tick=episode_start_tick,
                        episode_start_tile=episode_start_tile,
                    ),
                    "trajectory_digest": semantic_digest(step_records),
                }
            )

        return {
            "checkpoint": checkpoint,
            "seed_pack": seed_pack.identity.contract_id,
            "seed_pack_version": seed_pack.identity.version,
            "max_steps": max_steps,
            "per_seed": per_seed,
            "summary_digest": semantic_digest(per_seed),
        }
    finally:
        env.close()


def select_policy_action(policy_id: str, observation: dict[str, Any]) -> int | dict[str, object]:
    if policy_id == "wait_only_v0":
        return 0
    if policy_id == "attack_first_visible_or_wait_v0":
        npcs = observation["npcs"]
        if npcs:
            return {"action_id": 2, "visible_npc_index": int(npcs[0]["visible_index"])}
        return 0
    raise ValueError(f"Unsupported scripted policy id: {policy_id!r}")


if __name__ == "__main__":
    main()
