from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
from typing import Any


PHASE2_PROTOTYPE_ENV_COUNTS = (16, 64)


@dataclass(frozen=True)
class Phase2PrototypeGateStatus:
    benchmark_host_class: str
    benchmark_source_of_truth: bool
    production_rows_complete: bool
    learner_ceiling_rows_complete: bool
    prototype_16_sps: float | None
    prototype_64_sps: float | None
    prototype_scaling_ratio_64_vs_16: float | None
    learner_ceiling_16_sps: float | None
    learner_ceiling_64_sps: float | None
    learner_ceiling_scaling_ratio_64_vs_16: float | None
    next_move: str
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase2PrototypePacketReport:
    created_at: str
    output_dir: str
    production_reports: dict[str, str]
    learner_ceiling_report: str
    gate_status: Phase2PrototypeGateStatus

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["gate_status"] = self.gate_status.to_dict()
        return payload


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_phase2_prototype_gate(
    *,
    production_reports: dict[int, dict[str, Any]],
    learner_ceiling_report: dict[str, Any],
) -> Phase2PrototypeGateStatus:
    host_class = _detect_host_class(production_reports, learner_ceiling_report)
    benchmark_source_of_truth = _detect_source_of_truth(
        production_reports,
        learner_ceiling_report,
    )
    production_rows_complete = _production_rows_complete(production_reports)
    learner_ceiling_rows_complete = _learner_ceiling_rows_complete(learner_ceiling_report)

    prototype_16_sps = _production_sps(production_reports.get(16, {}))
    prototype_64_sps = _production_sps(production_reports.get(64, {}))
    prototype_scaling_ratio = _ratio(prototype_64_sps, prototype_16_sps)

    learner_16_sps = _learner_ceiling_sps(learner_ceiling_report, env_count=16)
    learner_64_sps = _learner_ceiling_sps(learner_ceiling_report, env_count=64)
    learner_scaling_ratio = _ratio(learner_64_sps, learner_16_sps)

    blockers: list[str] = []
    if not benchmark_source_of_truth:
        blockers.append("benchmark_source_of_truth_missing")
    if not production_rows_complete:
        blockers.append("production_packet_incomplete")
    if not learner_ceiling_rows_complete:
        blockers.append("learner_ceiling_packet_incomplete")
    if prototype_64_sps is None:
        blockers.append("prototype_64_sps_missing")
    if prototype_scaling_ratio is None:
        blockers.append("prototype_scaling_ratio_missing")
    if learner_scaling_ratio is None:
        blockers.append("learner_ceiling_scaling_ratio_missing")

    next_move = "incomplete"
    if not blockers:
        if float(prototype_64_sps) < 250.0:
            next_move = "deeper_trainer_replacement"
            blockers.append("prototype_64_below_trainer_bar")
        elif float(prototype_scaling_ratio) < 1.10:
            next_move = "continue_trainer_redesign"
        elif float(learner_scaling_ratio) < 1.05:
            next_move = "continue_trainer_redesign"
        else:
            next_move = "review_transport_or_topology"

    return Phase2PrototypeGateStatus(
        benchmark_host_class=host_class,
        benchmark_source_of_truth=benchmark_source_of_truth,
        production_rows_complete=production_rows_complete,
        learner_ceiling_rows_complete=learner_ceiling_rows_complete,
        prototype_16_sps=prototype_16_sps,
        prototype_64_sps=prototype_64_sps,
        prototype_scaling_ratio_64_vs_16=prototype_scaling_ratio,
        learner_ceiling_16_sps=learner_16_sps,
        learner_ceiling_64_sps=learner_64_sps,
        learner_ceiling_scaling_ratio_64_vs_16=learner_scaling_ratio,
        next_move=next_move,
        blockers=tuple(dict.fromkeys(blockers)),
    )


def build_phase2_prototype_packet_report(
    *,
    output_dir: Path,
    production_reports: dict[int, Path],
    learner_ceiling_report: Path,
) -> Phase2PrototypePacketReport:
    loaded_production = {
        env_count: load_json(path) for env_count, path in production_reports.items()
    }
    loaded_ceiling = load_json(learner_ceiling_report)
    gate_status = evaluate_phase2_prototype_gate(
        production_reports=loaded_production,
        learner_ceiling_report=loaded_ceiling,
    )
    return Phase2PrototypePacketReport(
        created_at=datetime.now(UTC).isoformat(),
        output_dir=str(output_dir),
        production_reports={
            str(env_count): str(path) for env_count, path in production_reports.items()
        },
        learner_ceiling_report=str(learner_ceiling_report),
        gate_status=gate_status,
    )


def _detect_host_class(
    production_reports: dict[int, dict[str, Any]],
    learner_ceiling_report: dict[str, Any],
) -> str:
    for payload in (*production_reports.values(), learner_ceiling_report):
        context = payload.get("context", {})
        hardware = context.get("hardware_profile", {})
        host_class = hardware.get("host_class")
        if host_class:
            return str(host_class)
    return "unknown"


def _detect_source_of_truth(
    production_reports: dict[int, dict[str, Any]],
    learner_ceiling_report: dict[str, Any],
) -> bool:
    for payload in (*production_reports.values(), learner_ceiling_report):
        context = payload.get("context", {})
        hardware = context.get("hardware_profile", {})
        explicit = hardware.get("performance_source_of_truth")
        if isinstance(explicit, bool):
            return explicit
        host_class = hardware.get("host_class")
        if host_class:
            return str(host_class) in {"linux_native", "wsl2"}
    return False


def _production_rows_complete(production_reports: dict[int, dict[str, Any]]) -> bool:
    return all(env_count in production_reports for env_count in PHASE2_PROTOTYPE_ENV_COUNTS)


def _learner_ceiling_rows_complete(payload: dict[str, Any]) -> bool:
    measurements = {
        int(entry.get("env_count", -1))
        for entry in payload.get("measurements", [])
    }
    return set(PHASE2_PROTOTYPE_ENV_COUNTS).issubset(measurements)


def _production_sps(payload: dict[str, Any]) -> float | None:
    for entry in payload.get("measurements", []):
        if str(entry.get("logging_mode")) == "disabled":
            value = _optional_float(entry.get("production_env_steps_per_second"))
            if value is not None:
                return value
            return _optional_float(entry.get("env_steps_per_second"))
    return None


def _learner_ceiling_sps(payload: dict[str, Any], *, env_count: int) -> float | None:
    for entry in payload.get("measurements", []):
        if int(entry.get("env_count", -1)) != int(env_count):
            continue
        value = _optional_float(entry.get("diagnostic_env_steps_per_second"))
        if value is not None:
            return value
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
