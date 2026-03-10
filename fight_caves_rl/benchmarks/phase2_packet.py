from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
from typing import Any

from fight_caves_rl.envs.shared_memory_transport import (
    PIPE_PICKLE_TRANSPORT_MODE,
    SHARED_MEMORY_TRANSPORT_MODE,
)


PHASE2_TRANSPORT_ENV_COUNTS = (16, 64)
PHASE2_TRAIN_ENV_COUNTS = (16, 64)


@dataclass(frozen=True)
class Phase2GateStatus:
    benchmark_host_class: str
    transport_rows_complete: bool
    train_rows_complete: bool
    transport_64_pipe_env_steps_per_second: float | None
    transport_64_shared_env_steps_per_second: float | None
    transport_64_speedup_vs_pipe: float | None
    train_16_pipe_sps: float | None
    train_16_shared_sps: float | None
    train_64_pipe_sps: float | None
    train_64_shared_sps: float | None
    train_64_speedup_vs_pipe: float | None
    shared_train_scaling_ratio_64_vs_16: float | None
    transport_signal_strong_enough: bool
    train_signal_strong_enough: bool
    scaling_signal_strong_enough: bool
    wc_p2_03_unblocked: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase2PacketReport:
    created_at: str
    output_dir: str
    transport_reports: dict[str, str]
    train_reports: dict[str, str]
    gate_status: Phase2GateStatus

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate_status"] = self.gate_status.to_dict()
        return payload


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_phase2_gate(
    *,
    transport_reports: dict[int, dict[str, Any]],
    train_reports: dict[tuple[str, int], dict[str, Any]],
) -> Phase2GateStatus:
    host_class = _detect_host_class(transport_reports, train_reports)
    transport_rows_complete = _transport_rows_complete(transport_reports)
    train_rows_complete = _train_rows_complete(train_reports)

    transport_measurements_64 = _transport_measurements(transport_reports.get(64, {}))
    train_pipe_16 = _train_sps(train_reports.get((PIPE_PICKLE_TRANSPORT_MODE, 16), {}))
    train_shared_16 = _train_sps(train_reports.get((SHARED_MEMORY_TRANSPORT_MODE, 16), {}))
    train_pipe_64 = _train_sps(train_reports.get((PIPE_PICKLE_TRANSPORT_MODE, 64), {}))
    train_shared_64 = _train_sps(train_reports.get((SHARED_MEMORY_TRANSPORT_MODE, 64), {}))

    transport_pipe_64 = transport_measurements_64.get(PIPE_PICKLE_TRANSPORT_MODE)
    transport_shared_64 = transport_measurements_64.get(SHARED_MEMORY_TRANSPORT_MODE)
    transport_64_speedup = _ratio(transport_shared_64, transport_pipe_64)
    train_64_speedup = _ratio(train_shared_64, train_pipe_64)
    shared_scaling_ratio = _ratio(train_shared_64, train_shared_16)

    transport_signal_strong_enough = (
        transport_64_speedup is not None and transport_64_speedup >= 1.20
    )
    train_signal_strong_enough = (
        train_64_speedup is not None and train_64_speedup >= 1.10
    )
    scaling_signal_strong_enough = (
        shared_scaling_ratio is not None and shared_scaling_ratio >= 1.25
    )

    blockers: list[str] = []
    if host_class != "linux_native":
        blockers.append("native_linux_source_of_truth_missing")
    if not transport_rows_complete:
        blockers.append("transport_packet_incomplete")
    if not train_rows_complete:
        blockers.append("train_packet_incomplete")
    if transport_64_speedup is None:
        blockers.append("transport_64_speedup_missing")
    if train_64_speedup is None:
        blockers.append("train_64_speedup_missing")
    if shared_scaling_ratio is None:
        blockers.append("shared_train_scaling_ratio_missing")
    if not transport_signal_strong_enough:
        blockers.append("transport_signal_too_weak")
    if not train_signal_strong_enough:
        blockers.append("train_signal_too_weak")
    if not scaling_signal_strong_enough:
        blockers.append("shared_train_scaling_too_weak")

    deduped_blockers = tuple(dict.fromkeys(blockers))
    return Phase2GateStatus(
        benchmark_host_class=host_class,
        transport_rows_complete=transport_rows_complete,
        train_rows_complete=train_rows_complete,
        transport_64_pipe_env_steps_per_second=transport_pipe_64,
        transport_64_shared_env_steps_per_second=transport_shared_64,
        transport_64_speedup_vs_pipe=transport_64_speedup,
        train_16_pipe_sps=train_pipe_16,
        train_16_shared_sps=train_shared_16,
        train_64_pipe_sps=train_pipe_64,
        train_64_shared_sps=train_shared_64,
        train_64_speedup_vs_pipe=train_64_speedup,
        shared_train_scaling_ratio_64_vs_16=shared_scaling_ratio,
        transport_signal_strong_enough=transport_signal_strong_enough,
        train_signal_strong_enough=train_signal_strong_enough,
        scaling_signal_strong_enough=scaling_signal_strong_enough,
        wc_p2_03_unblocked=not deduped_blockers,
        blockers=deduped_blockers,
    )


def build_phase2_packet_report(
    *,
    output_dir: Path,
    transport_reports: dict[int, Path],
    train_reports: dict[tuple[str, int], Path],
) -> Phase2PacketReport:
    loaded_transport = {env_count: load_json(path) for env_count, path in transport_reports.items()}
    loaded_train = {key: load_json(path) for key, path in train_reports.items()}
    gate_status = evaluate_phase2_gate(
        transport_reports=loaded_transport,
        train_reports=loaded_train,
    )
    return Phase2PacketReport(
        created_at=datetime.now(UTC).isoformat(),
        output_dir=str(output_dir),
        transport_reports={str(key): str(value) for key, value in transport_reports.items()},
        train_reports={
            f"{transport_mode}_{env_count}": str(path)
            for (transport_mode, env_count), path in train_reports.items()
        },
        gate_status=gate_status,
    )


def _detect_host_class(
    transport_reports: dict[int, dict[str, Any]],
    train_reports: dict[tuple[str, int], dict[str, Any]],
) -> str:
    for payload in (*transport_reports.values(), *train_reports.values()):
        context = payload.get("context", {})
        hardware = context.get("hardware_profile", {})
        host_class = hardware.get("host_class")
        if host_class:
            return str(host_class)
    return "unknown"


def _transport_rows_complete(reports: dict[int, dict[str, Any]]) -> bool:
    return all(env_count in reports for env_count in PHASE2_TRANSPORT_ENV_COUNTS)


def _train_rows_complete(reports: dict[tuple[str, int], dict[str, Any]]) -> bool:
    required = {
        (PIPE_PICKLE_TRANSPORT_MODE, env_count) for env_count in PHASE2_TRAIN_ENV_COUNTS
    } | {
        (SHARED_MEMORY_TRANSPORT_MODE, env_count) for env_count in PHASE2_TRAIN_ENV_COUNTS
    }
    return required.issubset(reports)


def _transport_measurements(payload: dict[str, Any]) -> dict[str, float]:
    measurements = {}
    for entry in payload.get("measurements", []):
        mode = str(entry.get("transport_mode"))
        value = _optional_float(entry.get("env_steps_per_second"))
        if value is not None:
            measurements[mode] = value
    return measurements


def _train_sps(payload: dict[str, Any]) -> float | None:
    for entry in payload.get("measurements", []):
        if str(entry.get("logging_mode")) == "disabled":
            return _optional_float(entry.get("env_steps_per_second"))
    return None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0.0):
        return None
    return float(numerator) / float(denominator)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
