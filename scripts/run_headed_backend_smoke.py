from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fight_caves_rl.envs.action_mapping import normalize_action
from fight_caves_rl.headed_backend_control import (
    first_action_result,
    issue_action,
    timestamp_id,
)
from fight_caves_rl.utils.paths import repo_root, workspace_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue shared-schema backend actions against the live RSPS-headed Fight Caves demo."
    )
    parser.add_argument("--account", default="fcdemo01")
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument(
        "--rsps-root",
        type=Path,
        default=workspace_root() / "RSPS",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the smoke artifact JSON.",
    )
    args = parser.parse_args()

    rsps_root = args.rsps_root.resolve()
    backend_root = rsps_root / "data" / "fight_caves_demo" / "backend_control"
    inbox_dir = backend_root / "inbox"
    results_dir = backend_root / "results"
    inbox_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    output = (
        args.output.resolve()
        if args.output is not None
        else repo_root() / "artifacts" / "headed_backend_smoke" / f"{args.account}-{timestamp_id()}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    initial_result = issue_action(
        inbox_dir=inbox_dir,
        results_dir=results_dir,
        account=args.account,
        action={"name": "wait"},
        timeout_seconds=args.timeout_seconds,
        poll_interval=args.poll_interval,
        request_context={"source": "backend_smoke"},
    )
    records.append(initial_result)

    initial_targets = initial_result["result"]["visible_targets_before"]
    initial_tile = initial_result["result"]["player_state_before"]["tile"]
    if not initial_targets:
        raise RuntimeError(
            "No visible targets were reported by the headed backend smoke. "
            "Make sure the demo client is already in the Fight Caves scene."
        )

    move_tile = {
        "x": int(initial_tile["x"]) + 1,
        "y": int(initial_tile["y"]),
        "level": int(initial_tile["level"]),
    }
    actions = [
        {"name": "walk_to_tile", "tile": move_tile},
        {"name": "wait"},
        {"name": "toggle_protection_prayer", "prayer": "protect_from_missiles"},
        {"name": "wait"},
        {"name": "attack_visible_npc", "visible_npc_index": int(initial_targets[0]["visible_index"])},
        {"name": "wait"},
        {"name": "eat_shark"},
        {"name": "wait"},
        {"name": "drink_prayer_potion"},
    ]

    for action in actions:
        record = issue_action(
            inbox_dir=inbox_dir,
            results_dir=results_dir,
            account=args.account,
            action=action,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            request_context={"source": "backend_smoke"},
        )
        records.append(record)

    processed_ticks = [record["result"]["processed_tick"] for record in records]
    tick_deltas = [
        processed_ticks[index] - processed_ticks[index - 1]
        for index in range(1, len(processed_ticks))
    ]
    attack_record = next(record for record in records if record["normalized_action"]["name"] == "attack_visible_npc")
    prayer_record = next(record for record in records if record["normalized_action"]["name"] == "toggle_protection_prayer")
    eat_record = next(record for record in records if record["normalized_action"]["name"] == "eat_shark")
    drink_record = next(record for record in records if record["normalized_action"]["name"] == "drink_prayer_potion")

    payload = {
        "schema_id": "fight_caves_headed_backend_smoke_v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "account": args.account,
        "rsps_root": str(rsps_root),
        "backend_root": str(backend_root),
        "output_path": str(output),
        "records": records,
        "backend_control_smoke": {
            "move_action_applied": first_action_result(records, "walk_to_tile")["action_applied"],
            "attack_action_applied": attack_record["result"]["action_result"]["action_applied"],
            "eat_action_applied": eat_record["result"]["action_result"]["action_applied"],
            "drink_action_applied": drink_record["result"]["action_result"]["action_applied"],
            "prayer_toggle_applied": prayer_record["result"]["action_result"]["action_applied"],
            "target_ordering_visible_count": len(initial_targets),
            "selected_visible_index": attack_record["normalized_action"]["visible_npc_index"],
            "selected_target_metadata": attack_record["result"]["action_result"]["metadata"],
        },
        "timing_validation": {
            "processed_ticks": processed_ticks,
            "tick_deltas": tick_deltas,
            "all_actions_processed_on_forward_ticks": all(delta >= 1 for delta in tick_deltas),
            "result_latencies_ms": [record["latency_ms"] for record in records],
        },
    }

    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote headed backend smoke artifact: {output}")
if __name__ == "__main__":
    main()
