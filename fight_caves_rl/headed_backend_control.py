from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from fight_caves_rl.envs.action_mapping import normalize_action


def issue_action(
    *,
    inbox_dir: Path,
    results_dir: Path,
    account: str,
    action: int | str | Mapping[str, object],
    timeout_seconds: float,
    poll_interval: float,
    request_context: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    normalized = normalize_action(action)
    request_id = f"{account}-{normalized.name}-{timestamp_id()}"
    request_path = inbox_dir / f"{request_id}.properties"
    result_path = results_dir / f"{request_id}.json"
    if result_path.exists():
        result_path.unlink()

    submitted_at_ms = int(time.time() * 1000)
    temp_path = request_path.with_suffix(".tmp")
    temp_path.write_text(
        build_properties_payload(
            request_id=request_id,
            account=account,
            submitted_at_ms=submitted_at_ms,
            normalized=normalized,
            request_context=request_context or {},
        ),
        encoding="utf-8",
    )
    temp_path.replace(request_path)

    started = time.monotonic()
    while time.monotonic() - started <= timeout_seconds:
        if result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            latency_ms = int((time.monotonic() - started) * 1000)
            return {
                "request_id": request_id,
                "request_path": str(request_path),
                "result_path": str(result_path),
                "normalized_action": asdict(normalized),
                "latency_ms": latency_ms,
                "result": result,
            }
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Timed out waiting for backend-control result {result_path.name}. "
        f"Make sure the RSPS demo server is running and the headed client is already in-game for account {account!r}."
    )


def build_properties_payload(
    *,
    request_id: str,
    account: str,
    submitted_at_ms: int,
    normalized: Any,
    request_context: Mapping[str, object],
) -> str:
    lines = [
        f"request_id={request_id}",
        f"account={account}",
        f"submitted_at_ms={submitted_at_ms}",
        f"action_id={normalized.action_id}",
        f"name={normalized.name}",
    ]
    if normalized.tile is not None:
        lines.append(f"x={normalized.tile.x}")
        lines.append(f"y={normalized.tile.y}")
        lines.append(f"level={normalized.tile.level}")
    if normalized.visible_npc_index is not None:
        lines.append(f"visible_npc_index={normalized.visible_npc_index}")
    if normalized.prayer is not None:
        lines.append(f"prayer={normalized.prayer}")
    for key, value in sorted(request_context.items()):
        if value is None:
            continue
        lines.append(f"{key}={value}")
    return "\n".join(lines) + "\n"


def first_action_result(records: list[dict[str, Any]], action_name: str) -> dict[str, Any]:
    record = next(record for record in records if record["normalized_action"]["name"] == action_name)
    return record["result"]["action_result"]


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S%f")
