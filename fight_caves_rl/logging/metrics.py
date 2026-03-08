from __future__ import annotations

from statistics import mean
from typing import Any, Mapping


def coerce_numeric_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    numeric_logs: dict[str, float] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            numeric_logs[str(key)] = float(int(value))
        elif isinstance(value, (int, float)):
            numeric_logs[str(key)] = float(value)
    return numeric_logs


def namespace_metrics(namespace: str, payload: Mapping[str, Any]) -> dict[str, float]:
    return {
        f"{namespace}/{key}": value
        for key, value in coerce_numeric_metrics(payload).items()
    }


def build_eval_summary_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    per_seed = list(payload.get("per_seed", []))
    if not per_seed:
        return {
            "eval/episode_count": 0.0,
        }

    steps = [float(item["steps_taken"]) for item in per_seed]
    terminated = [float(bool(item["terminated"])) for item in per_seed]
    truncated = [float(bool(item["truncated"])) for item in per_seed]
    return {
        "eval/episode_count": float(len(per_seed)),
        "eval/mean_steps": float(mean(steps)),
        "eval/terminated_rate": float(mean(terminated)),
        "eval/truncated_rate": float(mean(truncated)),
        "eval/seed_pack_version": float(payload["seed_pack_version"]),
    }
