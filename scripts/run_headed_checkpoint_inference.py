from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fight_caves_rl.envs.action_mapping import normalize_action
from fight_caves_rl.envs.observation_mapping import visible_targets_from_observation
from fight_caves_rl.envs.puffer_encoding import (
    build_policy_action_space,
    build_policy_observation_space,
    decode_action_from_policy,
    encode_observation_for_policy,
)
from fight_caves_rl.headed_backend_control import issue_action, timestamp_id
from fight_caves_rl.policies.checkpointing import load_checkpoint_metadata, load_policy_checkpoint
from fight_caves_rl.policies.registry import build_policy_from_metadata
from fight_caves_rl.utils.paths import repo_root, workspace_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run live trained-checkpoint inference against the RSPS-backed headed Fight Caves demo."
    )
    parser.add_argument("--account", default="fcdemo01")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--sampling-seed", type=int, default=123)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--max-preflight-waits", type=int, default=8)
    parser.add_argument(
        "--rsps-root",
        type=Path,
        default=workspace_root() / "RSPS",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    metadata = load_checkpoint_metadata(checkpoint_path)
    policy = build_policy_from_metadata(
        metadata,
        build_policy_observation_space(),
        build_policy_action_space(),
    )
    load_policy_checkpoint(checkpoint_path, policy)
    policy.eval()

    if args.policy_mode == "sample":
        torch.manual_seed(args.sampling_seed)
        np.random.seed(args.sampling_seed)
        random.seed(args.sampling_seed)

    rsps_root = args.rsps_root.resolve()
    backend_root = rsps_root / "data" / "fight_caves_demo" / "backend_control"
    inbox_dir = backend_root / "inbox"
    results_dir = backend_root / "results"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"{args.account}-{timestamp_id()}"
    output = (
        args.output.resolve()
        if args.output is not None
        else repo_root() / "artifacts" / "headed_live_inference" / f"{run_id}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    preflight_record = issue_action(
        inbox_dir=inbox_dir,
        results_dir=results_dir,
        account=args.account,
        action={"name": "wait"},
        timeout_seconds=args.timeout_seconds,
        poll_interval=args.poll_interval,
        request_context={
            "source": "live_inference_preflight",
            "run_id": run_id,
        },
    )
    current_observation = dict(preflight_record["result"]["observation_after"])
    policy_state = new_policy_state(policy)

    records: list[dict[str, Any]] = []
    for step_index in range(args.max_steps):
        action_vector = policy_action_vector(
            policy=policy,
            observation=current_observation,
            state=policy_state,
            policy_mode=args.policy_mode,
        )
        normalized = decode_action_from_policy(action_vector)
        record = issue_action(
            inbox_dir=inbox_dir,
            results_dir=results_dir,
            account=args.account,
            action=normalized,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            request_context={
                "source": "live_inference",
                "run_id": run_id,
                "policy_mode": args.policy_mode,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_policy_id": str(metadata.policy_id),
                "checkpoint_train_config_id": str(metadata.train_config_id),
                "step_index": step_index,
            },
        )
        record["policy_action_vector"] = [int(value) for value in np.asarray(action_vector, dtype=np.int64).tolist()]
        record["policy_action"] = normalized_to_map(normalized)
        records.append(record)
        current_observation = dict(record["result"]["observation_after"])

    output_payload = build_output_payload(
        run_id=run_id,
        account=args.account,
        checkpoint_path=checkpoint_path,
        metadata=metadata.to_dict(),
        policy_mode=args.policy_mode,
        sampling_seed=args.sampling_seed if args.policy_mode == "sample" else None,
        rsps_root=rsps_root,
        preflight_record=preflight_record,
        records=records,
    )
    output.write_text(json.dumps(output_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote headed live inference artifact: {output}")


def resolve_checkpoint_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidates = (
        repo_root() / "artifacts" / "acceptance" / "pr52_wsl_acceptance_20260311" / "train_run" / "fc-rl-train-1773247469-b81341f34265.pt",
        repo_root() / "artifacts" / "benchmarks" / "train" / "fast_train_v2" / "disabled" / "run_data" / "fc-rl-train-1773246426-dc100d1bb7c4.pt",
        repo_root() / "artifacts" / "benchmarks" / "train" / "train_1024env_v0" / "standard" / "run_data" / "fc-rl-train-1773243116-da2d2e385d61.pt",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No default headed live-inference checkpoint candidate exists. "
        "Pass --checkpoint explicitly."
    )


def new_policy_state(policy: torch.nn.Module) -> dict[str, object] | None:
    if getattr(policy, "hidden_size", None) is not None and policy.__class__.__name__.endswith("LSTMPolicy"):
        return {"lstm_h": None, "lstm_c": None}
    return None


def policy_action_vector(
    *,
    policy: torch.nn.Module,
    observation: dict[str, Any],
    state: dict[str, object] | None,
    policy_mode: str,
) -> np.ndarray:
    observation_vector = encode_observation_for_policy(observation)
    obs_tensor = torch.as_tensor(observation_vector, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _values = policy.forward_eval(obs_tensor, state=state)
    if policy_mode == "greedy":
        return np.asarray([int(torch.argmax(head, dim=1).item()) for head in logits], dtype=np.int64)
    sampled: list[int] = []
    for head in logits:
        probabilities = torch.softmax(head, dim=1)
        sampled.append(int(torch.multinomial(probabilities, 1).item()))
    return np.asarray(sampled, dtype=np.int64)


def build_output_payload(
    *,
    run_id: str,
    account: str,
    checkpoint_path: Path,
    metadata: dict[str, Any],
    policy_mode: str,
    sampling_seed: int | None,
    rsps_root: Path,
    preflight_record: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    processed_ticks = [record["result"]["processed_tick"] for record in records]
    tick_deltas = [
        processed_ticks[index] - processed_ticks[index - 1]
        for index in range(1, len(processed_ticks))
    ]
    rejected_steps = [
        {
            "step_index": index,
            "action_name": record["normalized_action"]["name"],
            "rejection_reason": record["result"]["action_result"]["rejection_reason"],
        }
        for index, record in enumerate(records)
        if not record["result"]["action_result"]["action_applied"]
    ]
    issued_action_counts = Counter(record["normalized_action"]["name"] for record in records)
    applied_action_counts = Counter(
        record["normalized_action"]["name"]
        for record in records
        if record["result"]["action_result"]["action_applied"]
    )
    attack_records = [record for record in records if record["normalized_action"]["name"] == "attack_visible_npc"]
    target_checks = [validate_attack_target_resolution(record) for record in attack_records]
    observed_visible_target_counts = [
        len(record["result"]["visible_targets_before"])
        for record in records
    ]
    session_log_path = latest_session_log_for_account(rsps_root, account)

    return {
        "schema_id": "fight_caves_headed_live_inference_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "account": account,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_metadata": metadata,
        "policy_mode": policy_mode,
        "sampling_seed": sampling_seed,
        "shared_action_schema_id": "headless_action_v1",
        "shared_action_schema_version": 1,
        "preflight": preflight_record,
        "records": records,
        "action_family_summary": {
            "issued_action_counts": dict(sorted(issued_action_counts.items())),
            "applied_action_counts": dict(sorted(applied_action_counts.items())),
        },
        "target_ordering_validation": {
            "attack_step_count": len(attack_records),
            "applied_attack_step_count": sum(1 for check in target_checks if check["action_applied"]),
            "all_applied_attack_steps_resolved_expected_target": all(
                check["passed"] for check in target_checks if check["action_applied"]
            ),
            "multi_target_observed": any(count > 1 for count in observed_visible_target_counts),
            "max_visible_target_count": max(observed_visible_target_counts, default=0),
            "checks": target_checks,
        },
        "timing_validation": {
            "processed_ticks": processed_ticks,
            "tick_deltas": tick_deltas,
            "all_actions_processed_on_forward_ticks": all(delta >= 1 for delta in tick_deltas),
            "result_latencies_ms": [record["latency_ms"] for record in records],
            "rejected_steps": rejected_steps,
        },
        "artifact_paths": {
            "session_log": str(session_log_path) if session_log_path is not None else None,
            "backend_control_results_dir": str(rsps_root / "data" / "fight_caves_demo" / "backend_control" / "results"),
        },
    }


def validate_attack_target_resolution(record: dict[str, Any]) -> dict[str, Any]:
    action = record["normalized_action"]
    result = record["result"]
    visible_targets_before = result["visible_targets_before"]
    selected_index = int(action["visible_npc_index"])
    target_count = len(visible_targets_before)
    target_before = visible_targets_before[selected_index] if 0 <= selected_index < target_count else None
    action_result = result["action_result"]
    selected_target_index = action_result["metadata"].get("target_npc_index")
    selected_target_id = action_result["metadata"].get("target_npc_id")
    action_applied = bool(action_result["action_applied"])
    passed = (
        action_applied
        and target_before is not None
        and str(target_before["npc_index"]) == str(selected_target_index)
        and str(target_before["id"]) == str(selected_target_id)
    )
    return {
        "step_request_id": record["request_id"],
        "selected_visible_index": selected_index,
        "visible_target_count": target_count,
        "visible_target_before": target_before,
        "resolved_target_npc_index": selected_target_index,
        "resolved_target_id": selected_target_id,
        "action_applied": action_applied,
        "rejection_reason": action_result["rejection_reason"],
        "passed": passed,
    }


def latest_session_log_for_account(rsps_root: Path, account: str) -> Path | None:
    session_logs = sorted(
        (rsps_root / "data" / "fight_caves_demo" / "artifacts" / "session_logs").glob(f"{account}-*.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    if not session_logs:
        return None
    return session_logs[-1]


def normalized_to_map(action: Any) -> dict[str, Any]:
    normalized = normalize_action(action)
    payload: dict[str, Any] = {
        "action_id": int(normalized.action_id),
        "name": str(normalized.name),
    }
    if normalized.tile is not None:
        payload["tile"] = {
            "x": int(normalized.tile.x),
            "y": int(normalized.tile.y),
            "level": int(normalized.tile.level),
        }
    if normalized.visible_npc_index is not None:
        payload["visible_npc_index"] = int(normalized.visible_npc_index)
    if normalized.prayer is not None:
        payload["prayer"] = str(normalized.prayer)
    return payload


if __name__ == "__main__":
    main()
