from __future__ import annotations

import argparse
import random
import sys

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.envs.correctness_env import FightCavesCorrectnessEnv


def choose_action(observation: dict[str, object], rng: random.Random) -> dict[str, object] | int:
    player = observation["player"]
    if player["hitpoints_current"] < player["hitpoints_max"] // 2 and player["consumables"]["shark_count"] > 0:
        return 4
    if player["prayer_current"] < max(10, player["prayer_max"] // 3) and player["consumables"]["prayer_potion_dose_count"] > 0:
        return 5
    npcs = observation["npcs"]
    if npcs:
        if rng.random() < 0.8:
            return {"action_id": 2, "visible_npc_index": npcs[0]["visible_index"]}
        return rng.choice(
            [
                {"action_id": 3, "prayer": "protect_from_magic"},
                {"action_id": 3, "prayer": "protect_from_missiles"},
                {"action_id": 3, "prayer": "protect_from_melee"},
            ]
        )
    if rng.random() < 0.1:
        return 6
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Random-policy smoke for the correctness env.")
    parser.add_argument("--seed", type=int, default=123456789)
    parser.add_argument("--max-steps", type=int, default=2_000)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    try:
        env = FightCavesCorrectnessEnv()
    except BridgeError as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        observation, info = env.reset(seed=args.seed)
        terminated = False
        truncated = False
        terminal_reason = None
        step_count = 0
        while not terminated and not truncated and step_count < args.max_steps:
            action = choose_action(observation, rng)
            observation, reward, terminated, truncated, step_info = env.step(action)
            step_count += 1
            terminal_reason = step_info["terminal_reason"]
        print(
            {
                "seed": args.seed,
                "steps": step_count,
                "tick": observation["tick"],
                "terminated": terminated,
                "truncated": truncated,
                "terminal_reason": terminal_reason,
                "episode_state": info["episode_state"],
            }
        )
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
