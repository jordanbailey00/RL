from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.envs.correctness_env import FightCavesCorrectnessEnv
from fight_caves_rl.replay.trace_packs import (
    project_episode_state_for_determinism,
    project_observation_for_determinism,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect semantic reset reproducibility payloads as JSON.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = collect_reset_repro(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def collect_reset_repro(seed: int) -> dict[str, object]:
    env = FightCavesCorrectnessEnv()
    try:
        first_observation, first_info = env.reset(seed=seed)
        for _ in range(3):
            env.step(0)
        second_observation, second_info = env.reset(seed=seed)
        return {
            "seed": seed,
            "first_episode_state": project_episode_state_for_determinism(first_info["episode_state"]),
            "second_episode_state": project_episode_state_for_determinism(second_info["episode_state"]),
            "first_observation": project_observation_for_determinism(
                first_observation,
                episode_start_tick=int(first_observation["tick"]),
                episode_start_tile=first_observation["player"]["tile"],
            ),
            "second_observation": project_observation_for_determinism(
                second_observation,
                episode_start_tick=int(second_observation["tick"]),
                episode_start_tile=second_observation["player"]["tile"],
            ),
        }
    finally:
        env.close()


if __name__ == "__main__":
    main()
