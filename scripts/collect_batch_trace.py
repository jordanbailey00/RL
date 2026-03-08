from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fight_caves_rl.bridge.batch_client import BatchClientConfig, HeadlessBatchClient
from fight_caves_rl.replay.trace_packs import (
    project_episode_state_for_determinism,
    project_observation_for_determinism,
    project_visible_targets_for_determinism,
    semantic_digest,
    serialize_action,
)

TRACE_ACTIONS = (
    0,
    6,
    {"action_id": 3, "prayer": "protect_from_missiles"},
    4,
    5,
    0,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a PR7 batch/reference bridge trace.")
    parser.add_argument("--mode", choices=("reference", "batch"), required=True)
    parser.add_argument("--env-count", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = collect_trace(mode=args.mode, env_count=args.env_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def collect_trace(*, mode: str, env_count: int) -> dict[str, Any]:
    client = HeadlessBatchClient.create(
        BatchClientConfig(
            env_count=env_count,
            account_name_prefix=f"batch_trace_{mode}",
            tick_cap=20_000,
        )
    )
    try:
        seeds = [70_000 + slot_index for slot_index in range(env_count)]
        reset = client.reset_batch(seeds=seeds)
        episode_starts = {
            result.slot_index: {
                "tick": int(result.observation["tick"]),
                "tile": dict(result.observation["player"]["tile"]),
                "episode_state": result.info["episode_state"],
            }
            for result in reset.results
        }

        steps: list[dict[str, Any]] = []
        for trace_index, action in enumerate(TRACE_ACTIONS):
            actions = [action] * env_count
            response = (
                client.step_reference(actions)
                if mode == "reference"
                else client.step_batch(actions)
            )
            slot_payloads: list[dict[str, Any]] = []
            for result in response.results:
                episode_start = episode_starts[result.slot_index]
                slot_payloads.append(
                    {
                        "slot_index": result.slot_index,
                        "action": serialize_action(result.action),
                        "action_result": result.info["action_result"],
                        "semantic_observation": project_observation_for_determinism(
                            result.observation,
                            episode_start_tick=episode_start["tick"],
                            episode_start_tile=episode_start["tile"],
                        ),
                        "semantic_visible_targets": project_visible_targets_for_determinism(
                            result.info["visible_targets"],
                            episode_start_tile=episode_start["tile"],
                        ),
                        "reward": result.reward,
                        "terminated": result.terminated,
                        "truncated": result.truncated,
                        "terminal_reason": result.info["terminal_reason"],
                    }
                )
            steps.append(
                {
                    "trace_index": trace_index,
                    "slots": slot_payloads,
                }
            )

        semantic_episode_states = {
            str(slot_index): project_episode_state_for_determinism(start["episode_state"])
            for slot_index, start in episode_starts.items()
        }
        digest_payload = {
            "env_count": env_count,
            "episode_states": semantic_episode_states,
            "steps": steps,
        }
        return {
            "mode": mode,
            "env_count": env_count,
            "bridge_protocol": client.protocol.to_dict(),
            "semantic_episode_states": semantic_episode_states,
            "steps": steps,
            "semantic_digest": semantic_digest(digest_payload),
        }
    finally:
        client.close()


if __name__ == "__main__":
    main()
