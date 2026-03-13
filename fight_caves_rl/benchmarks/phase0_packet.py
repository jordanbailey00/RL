from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from math import ceil
from pathlib import Path
from typing import Any


PHASE0_BRIDGE_ENV_COUNTS = (1, 16, 64)
PHASE0_VECENV_ENV_COUNTS = (1, 16, 64)
PHASE0_TRAIN_ENV_COUNTS = (4, 16, 64)


@dataclass(frozen=True)
class Phase0GateStatus:
    benchmark_host_class: str
    performance_source_of_truth: bool
    benchmark_source_of_truth: bool
    clean_pure_jvm_artifact_present: bool
    clean_batched_headless_artifact_present: bool
    bridge_rows_complete: bool
    vecenv_rows_complete: bool
    train_rows_complete: bool
    per_worker_sim_env_steps_per_second: float | None
    workers_needed_for_100k: int | None
    phase1_unblocked: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase0PacketReport:
    created_at: str
    output_dir: str
    sim_report_path: str
    bridge_reports: dict[str, str]
    vecenv_reports: dict[str, str]
    train_reports: dict[str, str]
    gate_status: Phase0GateStatus

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate_status"] = self.gate_status.to_dict()
        return payload


def load_json(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_phase0_gate(
    *,
    sim_report: dict[str, Any],
    bridge_reports: dict[int, dict[str, Any]],
    vecenv_reports: dict[int, dict[str, Any]],
    train_reports: dict[int, dict[str, Any]],
) -> Phase0GateStatus:
    runtime_metadata = dict(sim_report.get("runtime_metadata", {}))
    host_class = str(runtime_metadata.get("host_class", "unknown"))
    performance_source_of_truth = bool(runtime_metadata.get("performance_source_of_truth", False))
    benchmark_source_of_truth = performance_source_of_truth

    per_worker_payload = dict(sim_report.get("per_worker_ceiling", {}))
    per_worker_env_steps_per_second = _optional_float(
        per_worker_payload.get("batched_env_steps_per_second")
    )
    workers_needed_for_100k = (
        None
        if per_worker_env_steps_per_second is None or per_worker_env_steps_per_second <= 0.0
        else int(ceil(100_000.0 / per_worker_env_steps_per_second))
    )

    clean_pure_jvm_artifact_present = _optional_float(
        dict(sim_report.get("throughput", {})).get("ticks_per_second")
    ) is not None
    clean_batched_headless_artifact_present = per_worker_env_steps_per_second is not None
    bridge_rows_complete = _rows_complete(bridge_reports, PHASE0_BRIDGE_ENV_COUNTS)
    vecenv_rows_complete = _rows_complete(vecenv_reports, PHASE0_VECENV_ENV_COUNTS)
    train_rows_complete = _rows_complete(train_reports, PHASE0_TRAIN_ENV_COUNTS)

    blockers: list[str] = []
    if not benchmark_source_of_truth:
        blockers.append("benchmark_source_of_truth_missing")
    if not clean_pure_jvm_artifact_present:
        blockers.append("clean_pure_jvm_artifact_missing")
    if not clean_batched_headless_artifact_present:
        blockers.append("clean_batched_headless_artifact_missing")
    if not bridge_rows_complete:
        blockers.append("bridge_packet_incomplete")
    if not vecenv_rows_complete:
        blockers.append("vecenv_packet_incomplete")
    if not train_rows_complete:
        blockers.append("train_packet_incomplete")

    return Phase0GateStatus(
        benchmark_host_class=host_class,
        performance_source_of_truth=performance_source_of_truth,
        benchmark_source_of_truth=benchmark_source_of_truth,
        clean_pure_jvm_artifact_present=clean_pure_jvm_artifact_present,
        clean_batched_headless_artifact_present=clean_batched_headless_artifact_present,
        bridge_rows_complete=bridge_rows_complete,
        vecenv_rows_complete=vecenv_rows_complete,
        train_rows_complete=train_rows_complete,
        per_worker_sim_env_steps_per_second=per_worker_env_steps_per_second,
        workers_needed_for_100k=workers_needed_for_100k,
        phase1_unblocked=not blockers,
        blockers=tuple(blockers),
    )


def build_phase0_packet_report(
    *,
    output_dir: Path,
    sim_report_path: Path,
    bridge_reports: dict[int, Path],
    vecenv_reports: dict[int, Path],
    train_reports: dict[int, Path],
) -> Phase0PacketReport:
    sim_report = load_json(sim_report_path)
    loaded_bridge = {env_count: load_json(path) for env_count, path in bridge_reports.items()}
    loaded_vecenv = {env_count: load_json(path) for env_count, path in vecenv_reports.items()}
    loaded_train = {env_count: load_json(path) for env_count, path in train_reports.items()}
    gate_status = evaluate_phase0_gate(
        sim_report=sim_report,
        bridge_reports=loaded_bridge,
        vecenv_reports=loaded_vecenv,
        train_reports=loaded_train,
    )
    return Phase0PacketReport(
        created_at=datetime.now(UTC).isoformat(),
        output_dir=str(output_dir),
        sim_report_path=str(sim_report_path),
        bridge_reports={str(key): str(value) for key, value in bridge_reports.items()},
        vecenv_reports={str(key): str(value) for key, value in vecenv_reports.items()},
        train_reports={str(key): str(value) for key, value in train_reports.items()},
        gate_status=gate_status,
    )


def _rows_complete(reports: dict[int, dict[str, Any]], required: tuple[int, ...]) -> bool:
    return all(env_count in reports for env_count in required)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
