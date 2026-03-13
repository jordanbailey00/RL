from __future__ import annotations

import argparse
import json
from pathlib import Path

from fight_caves_rl.envs.puffer_encoding import encode_action_for_policy
from fight_caves_rl.puffer.factory import build_policy_episode_env
from fight_caves_rl.replay.trace_packs import (
    project_observation_for_determinism,
    project_visible_targets_for_determinism,
    resolve_trace_pack,
    semantic_digest,
    serialize_action,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a scripted trace against the PR5 Gym/Puffer wrapper.")
    parser.add_argument("--trace-pack", default="parity_single_wave_v0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    trace_pack = resolve_trace_pack(args.trace_pack)
    tick_cap = int(
        trace_pack.tick_cap
        if trace_pack.tick_cap is not None
        else max(args.max_steps or len(trace_pack.steps) + 8, len(trace_pack.steps))
    )
    env = build_policy_episode_env(
        {
            "start_wave": trace_pack.start_wave,
            "tick_cap": tick_cap,
        },
        reward_config_id="reward_sparse_v0",
    )
    try:
        seed = int(trace_pack.default_seed if args.seed is None else args.seed)
        env.reset(seed=seed)
        if env.last_raw_observation is None:
            raise RuntimeError("Expected raw observation after reset.")
        episode_start_tick = int(env.last_raw_observation["tick"])
        episode_start_tile = dict(env.last_raw_observation["player"]["tile"])
        semantic_initial_observation = project_observation_for_determinism(
            env.last_raw_observation,
            episode_start_tick=episode_start_tick,
            episode_start_tile=episode_start_tile,
        )
        records: list[dict[str, object]] = []
        max_steps = len(trace_pack.steps) if args.max_steps is None else min(len(trace_pack.steps), args.max_steps)
        terminated = False
        truncated = False
        terminal_reason_code = 0.0
        terminal_reason = None

        for step_index, step in enumerate(trace_pack.steps[:max_steps]):
            action_vector = encode_action_for_policy(step.action)
            _observation, reward, terminated, truncated, info = env.step(action_vector)
            if env.last_raw_observation is None:
                raise RuntimeError("Expected raw observation after step.")
            if env.last_step_info is None:
                raise RuntimeError("Expected raw step info after step.")
            terminal_reason_code = float(info["terminal_reason_code"])
            terminal_reason = env.last_step_info["terminal_reason"]
            records.append(
                {
                    "step_index": step_index,
                    "action": serialize_action(step.action),
                    "semantic_observation": project_observation_for_determinism(
                        env.last_raw_observation,
                        episode_start_tick=episode_start_tick,
                        episode_start_tile=episode_start_tile,
                    ),
                    "action_result": env.last_step_info["action_result"],
                    "semantic_visible_targets": project_visible_targets_for_determinism(
                        env.last_step_info["visible_targets"],
                        episode_start_tile=episode_start_tile,
                    ),
                    "terminated": terminated,
                    "truncated": truncated,
                    "terminal_reason": terminal_reason,
                }
            )
            if terminated or truncated:
                break

        digest_payload = {
            "trace_pack": trace_pack.identity.contract_id,
            "seed": seed,
            "initial_observation": semantic_initial_observation,
            "steps": [
                {
                    "action": record["action"],
                    "semantic_observation": record["semantic_observation"],
                    "action_result": record["action_result"],
                    "semantic_visible_targets": record["semantic_visible_targets"],
                    "terminated": record["terminated"],
                    "truncated": record["truncated"],
                    "terminal_reason": record["terminal_reason"],
                }
                for record in records
            ],
        }
        final_observation = records[-1]["semantic_observation"] if records else semantic_initial_observation
        payload = {
            "trace_pack": trace_pack.identity.contract_id,
            "trace_pack_version": trace_pack.identity.version,
            "seed": seed,
            "steps_executed": len(records),
            "completed_all_steps": len(records) == len(trace_pack.steps[:max_steps]),
            "final_relative_tick": int(final_observation["tick"]),
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason_code": terminal_reason_code,
            "semantic_digest": semantic_digest(digest_payload),
            "expected_semantic_digest": trace_pack.expected_semantic_digest,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
