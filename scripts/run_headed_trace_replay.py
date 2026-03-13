from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fight_caves_rl.envs.action_mapping import normalize_action
from fight_caves_rl.headed_backend_control import issue_action, timestamp_id
from fight_caves_rl.utils.paths import repo_root, workspace_root

_TRACE_PACKS_PATH = REPO_ROOT / "fight_caves_rl" / "replay" / "trace_packs.py"
_TRACE_PACKS_SPEC = importlib.util.spec_from_file_location("fight_caves_rl_trace_packs_direct", _TRACE_PACKS_PATH)
if _TRACE_PACKS_SPEC is None or _TRACE_PACKS_SPEC.loader is None:
    raise RuntimeError(f"Unable to load trace pack module from {_TRACE_PACKS_PATH}")
_TRACE_PACKS_MODULE = importlib.util.module_from_spec(_TRACE_PACKS_SPEC)
sys.modules[_TRACE_PACKS_SPEC.name] = _TRACE_PACKS_MODULE
_TRACE_PACKS_SPEC.loader.exec_module(_TRACE_PACKS_MODULE)
resolve_trace_pack = _TRACE_PACKS_MODULE.resolve_trace_pack
serialize_action = _TRACE_PACKS_MODULE.serialize_action


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay shared-schema action traces against the live RSPS-headed Fight Caves demo."
    )
    parser.add_argument("--account", default="fcdemo01")
    parser.add_argument("--trace-pack", default=None)
    parser.add_argument("--trace-json", type=Path, default=None)
    parser.add_argument(
        "--trace-template",
        choices=("headed_demo_all_actions_v0",),
        default="headed_demo_all_actions_v0",
        help=(
            "Generate a session-valid input trace when a built-in trace pack or explicit trace JSON "
            "is not supplied. The generated trace still uses the shared action schema."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--max-preflight-waits", type=int, default=8)
    parser.add_argument(
        "--rsps-root",
        type=Path,
        default=workspace_root() / "RSPS",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--input-trace-output", type=Path, default=None)
    args = parser.parse_args()

    if args.trace_pack and args.trace_json:
        raise SystemExit("Specify only one of --trace-pack or --trace-json.")

    rsps_root = args.rsps_root.resolve()
    backend_root = rsps_root / "data" / "fight_caves_demo" / "backend_control"
    inbox_dir = backend_root / "inbox"
    results_dir = backend_root / "results"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    replay_id = f"{args.account}-{timestamp_id()}"
    output = (
        args.output.resolve()
        if args.output is not None
        else repo_root() / "artifacts" / "headed_replay" / f"{replay_id}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    input_trace_output = (
        args.input_trace_output.resolve()
        if args.input_trace_output is not None
        else output.with_name(f"{output.stem}-input-trace.json")
    )

    preflight_records, preflight_state = preflight_live_state(
        inbox_dir=inbox_dir,
        results_dir=results_dir,
        account=args.account,
        timeout_seconds=args.timeout_seconds,
        poll_interval=args.poll_interval,
        max_preflight_waits=args.max_preflight_waits,
    )
    trace_input = resolve_trace_input(
        args=args,
        preflight_state=preflight_state,
        replay_id=replay_id,
    )
    input_trace_output.write_text(
        json.dumps(trace_input, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    records: list[dict[str, Any]] = []
    for step_index, step in enumerate(trace_input["steps"]):
        record = issue_action(
            inbox_dir=inbox_dir,
            results_dir=results_dir,
            account=args.account,
            action=step,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            request_context={
                "source": "replay",
                "replay_id": replay_id,
                "trace_id": trace_input["trace_id"],
                "step_index": step_index,
            },
        )
        records.append(record)

    processed_ticks = [record["result"]["processed_tick"] for record in records]
    tick_deltas = [
        processed_ticks[index] - processed_ticks[index - 1]
        for index in range(1, len(processed_ticks))
    ]
    attack_steps = [record for record in records if record["normalized_action"]["name"] == "attack_visible_npc"]
    rejected_steps = [
        {
            "step_index": index,
            "action_name": record["normalized_action"]["name"],
            "rejection_reason": record["result"]["action_result"]["rejection_reason"],
        }
        for index, record in enumerate(records)
        if not record["result"]["action_result"]["action_applied"]
    ]
    target_resolution_checks = [validate_attack_target_resolution(record) for record in attack_steps]

    payload = {
        "schema_id": "fight_caves_headed_replay_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "account": args.account,
        "replay_id": replay_id,
        "rsps_root": str(rsps_root),
        "backend_root": str(backend_root),
        "input_trace_path": str(input_trace_output),
        "output_path": str(output),
        "trace_boundary": {
            "input_mode": trace_input["input_mode"],
            "trace_id": trace_input["trace_id"],
            "trace_source": trace_input["trace_source"],
            "action_schema_id": trace_input["action_schema_id"],
            "action_schema_version": trace_input["action_schema_version"],
            "step_count": len(trace_input["steps"]),
        },
        "preflight": {
            "records": preflight_records,
            "state": preflight_state,
        },
        "records": records,
        "target_ordering_validation": {
            "attack_step_count": len(attack_steps),
            "all_attack_steps_resolved_expected_target": all(check["passed"] for check in target_resolution_checks),
            "checks": target_resolution_checks,
        },
        "timing_validation": {
            "processed_ticks": processed_ticks,
            "tick_deltas": tick_deltas,
            "all_actions_processed_on_forward_ticks": all(delta >= 1 for delta in tick_deltas),
            "result_latencies_ms": [record["latency_ms"] for record in records],
            "rejected_steps": rejected_steps,
        },
        "replay_summary": {
            "all_actions_applied": not rejected_steps,
            "move_steps": count_actions(records, "walk_to_tile"),
            "attack_steps": count_actions(records, "attack_visible_npc"),
            "eat_steps": count_actions(records, "eat_shark"),
            "drink_steps": count_actions(records, "drink_prayer_potion"),
            "prayer_steps": count_actions(records, "toggle_protection_prayer"),
        },
    }

    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote headed replay artifact: {output}")
    print(f"Wrote replay input trace: {input_trace_output}")


def resolve_trace_input(
    *,
    args: argparse.Namespace,
    preflight_state: dict[str, Any],
    replay_id: str,
) -> dict[str, Any]:
    if args.trace_pack:
        pack = resolve_trace_pack(args.trace_pack)
        return {
            "input_mode": "trace_pack",
            "trace_id": pack.identity.contract_id,
            "trace_source": pack.source_ref,
            "action_schema_id": "headless_action_v1",
            "action_schema_version": 1,
            "expected_start_wave": pack.start_wave,
            "steps": [serialize_action(step.action) for step in pack.steps],
        }

    if args.trace_json is not None:
        payload = json.loads(args.trace_json.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            steps = payload
            trace_id = args.trace_json.stem
            trace_source = str(args.trace_json)
        else:
            steps = payload["steps"]
            trace_id = str(payload.get("trace_id", args.trace_json.stem))
            trace_source = str(payload.get("trace_source", args.trace_json))
        return {
            "input_mode": "trace_json",
            "trace_id": trace_id,
            "trace_source": trace_source,
            "action_schema_id": str(payload.get("action_schema_id", "headless_action_v1")) if isinstance(payload, Mapping) else "headless_action_v1",
            "action_schema_version": int(payload.get("action_schema_version", 1)) if isinstance(payload, Mapping) else 1,
            "steps": [serialize_action(normalize_action(step)) for step in steps],
        }

    return materialize_demo_trace(preflight_state=preflight_state, replay_id=replay_id)


def materialize_demo_trace(*, preflight_state: dict[str, Any], replay_id: str) -> dict[str, Any]:
    initial_tile = preflight_state["player_state"]["tile"]
    attack_visible_index = int(preflight_state["first_attack_visible_index"])
    move_tile = {
        "x": int(initial_tile["x"]) + 1,
        "y": int(initial_tile["y"]),
        "level": int(initial_tile["level"]),
    }
    steps = [
        {"name": "walk_to_tile", "tile": move_tile},
        {"name": "wait"},
        {"name": "toggle_protection_prayer", "prayer": "protect_from_missiles"},
        {"name": "wait"},
        {"name": "attack_visible_npc", "visible_npc_index": attack_visible_index},
        {"name": "wait"},
        {"name": "eat_shark"},
        {"name": "wait"},
        {"name": "drink_prayer_potion"},
    ]
    return {
        "input_mode": "materialized_template",
        "trace_id": f"headed_demo_all_actions_v0-{replay_id}",
        "trace_source": "materialized_from_live_headed_state",
        "action_schema_id": "headless_action_v1",
        "action_schema_version": 1,
        "preflight_wave": preflight_state["player_state"]["wave"],
        "preflight_player_tile": deepcopy(initial_tile),
        "steps": [serialize_action(normalize_action(step)) for step in steps],
    }


def preflight_live_state(
    *,
    inbox_dir: Path,
    results_dir: Path,
    account: str,
    timeout_seconds: float,
    poll_interval: float,
    max_preflight_waits: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    candidate_visible_index: int | None = None
    player_state: dict[str, Any] | None = None
    visible_targets: list[dict[str, Any]] = []
    for wait_index in range(max_preflight_waits):
        record = issue_action(
            inbox_dir=inbox_dir,
            results_dir=results_dir,
            account=account,
            action={"name": "wait"},
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            request_context={
                "source": "replay_preflight",
                "preflight_step": wait_index,
            },
        )
        records.append(record)
        result = record["result"]
        player_state = result["player_state_before"]
        visible_targets = result["visible_targets_before"]
        candidate_visible_index = first_attackable_visible_index(visible_targets)
        if candidate_visible_index is not None:
            break
    if player_state is None:
        raise RuntimeError("Failed to capture preflight player state for headed replay.")
    if candidate_visible_index is None:
        raise RuntimeError(
            "No attackable visible target was observed during headed replay preflight. "
            "Make sure the player is in an active Fight Caves wave before replaying."
        )
    return records, {
        "player_state": player_state,
        "visible_targets": visible_targets,
        "first_attack_visible_index": candidate_visible_index,
    }


def first_attackable_visible_index(visible_targets: Sequence[Mapping[str, Any]]) -> int | None:
    for target in visible_targets:
        target_id = str(target["id"])
        if not target_id.endswith("_spawn_point"):
            return int(target["visible_index"])
    return None


def validate_attack_target_resolution(record: Mapping[str, Any]) -> dict[str, Any]:
    action = record["normalized_action"]
    result = record["result"]
    visible_targets_before = result["visible_targets_before"]
    selected_index = int(action["visible_npc_index"])
    target_count = len(visible_targets_before)
    target_before = visible_targets_before[selected_index] if 0 <= selected_index < target_count else None
    selected_target_index = result["action_result"]["metadata"].get("target_npc_index")
    selected_target_id = result["action_result"]["metadata"].get("target_npc_id")
    passed = (
        target_before is not None
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
        "passed": passed,
    }


def count_actions(records: Sequence[Mapping[str, Any]], action_name: str) -> int:
    return sum(1 for record in records if record["normalized_action"]["name"] == action_name)


if __name__ == "__main__":
    main()
