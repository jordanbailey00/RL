from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import pstats
from typing import Any


PHASE1_BRIDGE_ENV_COUNTS = (1, 16, 64)
PHASE1_VECENV_ENV_COUNTS = (16, 64)


@dataclass(frozen=True)
class Phase1ProfileSummary:
    total_time_seconds: float
    step_batch_cumulative_seconds: float
    raw_conversion_cumulative_seconds: float
    flat_observe_cumulative_seconds: float
    build_step_buffers_cumulative_seconds: float
    raw_object_conversion_still_dominant: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase1GateStatus:
    benchmark_host_class: str
    phase0_baseline_host_class: str | None
    bridge_rows_complete: bool
    vecenv_rows_complete: bool
    python_profile_present: bool
    bridge_64_env_steps_per_second: float | None
    vecenv_64_env_steps_per_second: float | None
    bridge_64_improvement_ratio: float | None
    vecenv_64_improvement_ratio: float | None
    bridge_threshold_met: bool
    vecenv_threshold_met: bool
    raw_object_conversion_still_dominant: bool
    phase2_unblocked: bool
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase1PacketReport:
    created_at: str
    output_dir: str
    phase0_baseline_dir: str | None
    bridge_reports: dict[str, str]
    vecenv_reports: dict[str, str]
    python_profile_path: str
    python_profile_top_path: str
    python_profile_summary: Phase1ProfileSummary
    gate_status: Phase1GateStatus

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["python_profile_summary"] = self.python_profile_summary.to_dict()
        payload["gate_status"] = self.gate_status.to_dict()
        return payload


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize_python_profile(profile_path: str | Path) -> Phase1ProfileSummary:
    stats = pstats.Stats(str(profile_path))
    total_time = float(stats.total_tt)
    step_batch_cumulative = _cumulative_for(stats, "step_batch")
    raw_conversion_cumulative = _cumulative_for(stats, "pythonize_observation") + _cumulative_for(stats, "_pythonize")
    flat_observe_cumulative = _cumulative_for(stats, "observe_flat") + _cumulative_for(stats, "observe_flat_jvm")
    build_step_buffers_cumulative = _cumulative_for(stats, "build_step_buffers")
    raw_still_dominant = (
        step_batch_cumulative > 0.0
        and raw_conversion_cumulative >= (0.5 * step_batch_cumulative)
    )
    return Phase1ProfileSummary(
        total_time_seconds=total_time,
        step_batch_cumulative_seconds=step_batch_cumulative,
        raw_conversion_cumulative_seconds=raw_conversion_cumulative,
        flat_observe_cumulative_seconds=flat_observe_cumulative,
        build_step_buffers_cumulative_seconds=build_step_buffers_cumulative,
        raw_object_conversion_still_dominant=raw_still_dominant,
    )


def write_profile_top_table(
    profile_path: str | Path,
    output_path: str | Path,
    *,
    limit: int = 40,
) -> Path:
    stats = pstats.Stats(str(profile_path))
    stats.sort_stats("cumulative")
    lines: list[str] = []
    stats.stream = _ListStream(lines)
    stats.print_stats(limit)
    output = Path(output_path)
    output.write_text("".join(lines), encoding="utf-8")
    return output


def evaluate_phase1_gate(
    *,
    phase0_baseline_dir: Path | None,
    bridge_reports: dict[int, dict[str, Any]],
    vecenv_reports: dict[int, dict[str, Any]],
    python_profile_summary: Phase1ProfileSummary,
) -> Phase1GateStatus:
    current_host_class = _detect_host_class(bridge_reports, vecenv_reports)
    baseline_host_class: str | None = None
    baseline_bridge_64 = None
    baseline_vecenv_64 = None
    if phase0_baseline_dir is not None:
        gate_summary = load_json(phase0_baseline_dir / "gate_summary.json")
        baseline_host_class = str(gate_summary.get("benchmark_host_class"))
        baseline_bridge_64 = _phase0_bridge_sps(phase0_baseline_dir / "bridge_64env.json")
        baseline_vecenv_64 = _phase0_vecenv_sps(phase0_baseline_dir / "vecenv_64env.json")

    bridge_rows_complete = _rows_complete(bridge_reports, PHASE1_BRIDGE_ENV_COUNTS)
    vecenv_rows_complete = _rows_complete(vecenv_reports, PHASE1_VECENV_ENV_COUNTS)
    bridge_64 = _optional_float(bridge_reports.get(64, {}).get("batch", {}).get("env_steps_per_second"))
    vecenv_64 = _optional_float(vecenv_reports.get(64, {}).get("measurement", {}).get("env_steps_per_second"))

    bridge_ratio = None
    if bridge_64 is not None and baseline_bridge_64 not in (None, 0.0):
        bridge_ratio = float(bridge_64) / float(baseline_bridge_64)
    vecenv_ratio = None
    if vecenv_64 is not None and baseline_vecenv_64 not in (None, 0.0):
        vecenv_ratio = float(vecenv_64) / float(baseline_vecenv_64)

    blockers: list[str] = []
    if current_host_class != "linux_native":
        blockers.append("native_linux_source_of_truth_missing")
    if phase0_baseline_dir is None:
        blockers.append("phase0_baseline_missing")
    if baseline_host_class not in (None, "linux_native"):
        blockers.append("phase0_baseline_host_class_mismatch")
    if not bridge_rows_complete:
        blockers.append("bridge_packet_incomplete")
    if not vecenv_rows_complete:
        blockers.append("vecenv_packet_incomplete")
    if bridge_ratio is None:
        blockers.append("bridge_ratio_missing")
    if vecenv_ratio is None:
        blockers.append("vecenv_ratio_missing")
    if bridge_64 is None:
        blockers.append("bridge_64_row_missing")
    if vecenv_64 is None:
        blockers.append("vecenv_64_row_missing")
    if python_profile_summary.raw_object_conversion_still_dominant:
        blockers.append("raw_object_conversion_still_dominant")
    if bridge_64 is not None and bridge_64 < 5_000.0:
        blockers.append("bridge_64_below_reconsider_threshold")
    if vecenv_64 is not None and vecenv_64 < 4_000.0:
        blockers.append("vecenv_64_below_reconsider_threshold")

    bridge_threshold_met = (
        bridge_ratio is not None and bridge_ratio >= 5.0 and bridge_64 is not None and 8_000.0 <= bridge_64 <= 15_000.0
    )
    vecenv_threshold_met = (
        vecenv_ratio is not None and vecenv_ratio >= 4.0 and vecenv_64 is not None and 6_000.0 <= vecenv_64 <= 12_000.0
    )
    if not bridge_threshold_met:
        blockers.append("bridge_threshold_not_met")
    if not vecenv_threshold_met:
        blockers.append("vecenv_threshold_not_met")

    deduped_blockers = tuple(dict.fromkeys(blockers))
    return Phase1GateStatus(
        benchmark_host_class=current_host_class,
        phase0_baseline_host_class=baseline_host_class,
        bridge_rows_complete=bridge_rows_complete,
        vecenv_rows_complete=vecenv_rows_complete,
        python_profile_present=True,
        bridge_64_env_steps_per_second=bridge_64,
        vecenv_64_env_steps_per_second=vecenv_64,
        bridge_64_improvement_ratio=bridge_ratio,
        vecenv_64_improvement_ratio=vecenv_ratio,
        bridge_threshold_met=bridge_threshold_met,
        vecenv_threshold_met=vecenv_threshold_met,
        raw_object_conversion_still_dominant=python_profile_summary.raw_object_conversion_still_dominant,
        phase2_unblocked=not deduped_blockers,
        blockers=deduped_blockers,
    )


def build_phase1_packet_report(
    *,
    output_dir: Path,
    phase0_baseline_dir: Path | None,
    bridge_reports: dict[int, Path],
    vecenv_reports: dict[int, Path],
    python_profile_path: Path,
    python_profile_top_path: Path,
) -> Phase1PacketReport:
    loaded_bridge = {env_count: load_json(path) for env_count, path in bridge_reports.items()}
    loaded_vecenv = {env_count: load_json(path) for env_count, path in vecenv_reports.items()}
    profile_summary = summarize_python_profile(python_profile_path)
    gate_status = evaluate_phase1_gate(
        phase0_baseline_dir=phase0_baseline_dir,
        bridge_reports=loaded_bridge,
        vecenv_reports=loaded_vecenv,
        python_profile_summary=profile_summary,
    )
    return Phase1PacketReport(
        created_at=datetime.now(UTC).isoformat(),
        output_dir=str(output_dir),
        phase0_baseline_dir=None if phase0_baseline_dir is None else str(phase0_baseline_dir),
        bridge_reports={str(key): str(value) for key, value in bridge_reports.items()},
        vecenv_reports={str(key): str(value) for key, value in vecenv_reports.items()},
        python_profile_path=str(python_profile_path),
        python_profile_top_path=str(python_profile_top_path),
        python_profile_summary=profile_summary,
        gate_status=gate_status,
    )


def _phase0_bridge_sps(path: Path) -> float | None:
    payload = load_json(path)
    return _optional_float(payload.get("batch", {}).get("env_steps_per_second"))


def _phase0_vecenv_sps(path: Path) -> float | None:
    payload = load_json(path)
    return _optional_float(payload.get("measurement", {}).get("env_steps_per_second"))


def _detect_host_class(
    bridge_reports: dict[int, dict[str, Any]],
    vecenv_reports: dict[int, dict[str, Any]],
) -> str:
    for payload in (*bridge_reports.values(), *vecenv_reports.values()):
        context = payload.get("context", {})
        hardware = context.get("hardware_profile", {})
        host_class = hardware.get("host_class")
        if host_class:
            return str(host_class)
    return "unknown"


def _rows_complete(reports: dict[int, dict[str, Any]], required: tuple[int, ...]) -> bool:
    return all(env_count in reports for env_count in required)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cumulative_for(stats: pstats.Stats, function_name: str) -> float:
    total = 0.0
    for (_, _, candidate_name), entry in stats.stats.items():
        if candidate_name == function_name:
            total += float(entry[3])
    return total


class _ListStream:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    def write(self, value: str) -> None:
        self._lines.append(value)

    def flush(self) -> None:
        return None
