from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LogRecord:
    step: int
    payload: dict[str, float]


@dataclass
class SmokeLogger:
    args: dict[str, Any]
    run_id: str | None = None
    records: list[LogRecord] = field(default_factory=list)
    closed_model_path: str | None = None

    def __post_init__(self) -> None:
        if self.run_id is None:
            self.run_id = f"smoke_{int(time.time() * 1000)}"

    def log(self, logs: dict[str, Any], step: int) -> None:
        numeric_logs: dict[str, float] = {}
        for key, value in logs.items():
            if isinstance(value, bool):
                numeric_logs[key] = float(int(value))
            elif isinstance(value, (int, float)):
                numeric_logs[key] = float(value)
        self.records.append(LogRecord(step=int(step), payload=numeric_logs))

    def close(self, model_path: str) -> None:
        self.closed_model_path = model_path

