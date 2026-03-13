import json

from fight_caves_rl.benchmarks.phase1_packet import (
    Phase1ProfileSummary,
    evaluate_phase1_gate,
)


def _bridge_report(sps: float, *, host_class: str = "wsl2", source_of_truth: bool = True) -> dict:
    return {
        "context": {
            "hardware_profile": {
                "host_class": host_class,
                "performance_source_of_truth": source_of_truth,
            }
        },
        "batch": {"env_steps_per_second": sps},
    }


def _vecenv_report(sps: float, *, host_class: str = "wsl2", source_of_truth: bool = True) -> dict:
    return {
        "context": {
            "hardware_profile": {
                "host_class": host_class,
                "performance_source_of_truth": source_of_truth,
            }
        },
        "measurement": {"env_steps_per_second": sps},
    }


def test_phase1_gate_allows_wsl_source_of_truth_with_phase0_packet(tmp_path):
    baseline_dir = tmp_path / "phase0"
    baseline_dir.mkdir()
    (baseline_dir / "phase0_packet.json").write_text(
        json.dumps(
            {
                "gate_status": {
                    "benchmark_host_class": "wsl2",
                    "performance_source_of_truth": True,
                    "benchmark_source_of_truth": True,
                }
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "bridge_64env.json").write_text(
        json.dumps({"batch": {"env_steps_per_second": 1000.0}}),
        encoding="utf-8",
    )
    (baseline_dir / "vecenv_64env.json").write_text(
        json.dumps({"measurement": {"env_steps_per_second": 1000.0}}),
        encoding="utf-8",
    )

    gate = evaluate_phase1_gate(
        phase0_baseline_dir=baseline_dir,
        bridge_reports={
            1: _bridge_report(2000.0),
            16: _bridge_report(4000.0),
            64: _bridge_report(9000.0),
        },
        vecenv_reports={
            16: _vecenv_report(7000.0),
            64: _vecenv_report(8000.0),
        },
        python_profile_summary=Phase1ProfileSummary(
            total_time_seconds=1.0,
            step_batch_cumulative_seconds=1.0,
            raw_conversion_cumulative_seconds=0.1,
            flat_observe_cumulative_seconds=0.1,
            build_step_buffers_cumulative_seconds=0.1,
            raw_object_conversion_still_dominant=False,
        ),
    )

    assert gate.benchmark_source_of_truth is True
    assert gate.phase0_baseline_source_of_truth is True
    assert gate.phase2_unblocked is True
    assert gate.blockers == ()


def test_phase1_gate_blocks_when_current_run_is_not_source_of_truth():
    gate = evaluate_phase1_gate(
        phase0_baseline_dir=None,
        bridge_reports={
            1: _bridge_report(2000.0, source_of_truth=False),
            16: _bridge_report(4000.0, source_of_truth=False),
            64: _bridge_report(9000.0, source_of_truth=False),
        },
        vecenv_reports={
            16: _vecenv_report(7000.0, source_of_truth=False),
            64: _vecenv_report(8000.0, source_of_truth=False),
        },
        python_profile_summary=Phase1ProfileSummary(
            total_time_seconds=1.0,
            step_batch_cumulative_seconds=1.0,
            raw_conversion_cumulative_seconds=0.1,
            flat_observe_cumulative_seconds=0.1,
            build_step_buffers_cumulative_seconds=0.1,
            raw_object_conversion_still_dominant=False,
        ),
    )

    assert gate.benchmark_source_of_truth is False
    assert gate.phase2_unblocked is False
    assert "benchmark_source_of_truth_missing" in gate.blockers


def test_phase1_gate_accepts_legacy_wsl_host_without_explicit_flag(tmp_path):
    baseline_dir = tmp_path / "phase0"
    baseline_dir.mkdir()
    (baseline_dir / "phase0_packet.json").write_text(
        json.dumps(
            {
                "gate_status": {
                    "benchmark_host_class": "wsl2",
                    "benchmark_source_of_truth": True,
                }
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "bridge_64env.json").write_text(
        json.dumps({"batch": {"env_steps_per_second": 1000.0}}),
        encoding="utf-8",
    )
    (baseline_dir / "vecenv_64env.json").write_text(
        json.dumps({"measurement": {"env_steps_per_second": 1000.0}}),
        encoding="utf-8",
    )

    gate = evaluate_phase1_gate(
        phase0_baseline_dir=baseline_dir,
        bridge_reports={
            1: {"context": {"hardware_profile": {"host_class": "wsl2"}}, "batch": {"env_steps_per_second": 2000.0}},
            16: {"context": {"hardware_profile": {"host_class": "wsl2"}}, "batch": {"env_steps_per_second": 4000.0}},
            64: {"context": {"hardware_profile": {"host_class": "wsl2"}}, "batch": {"env_steps_per_second": 9000.0}},
        },
        vecenv_reports={
            16: {
                "context": {"hardware_profile": {"host_class": "wsl2"}},
                "measurement": {"env_steps_per_second": 7000.0},
            },
            64: {
                "context": {"hardware_profile": {"host_class": "wsl2"}},
                "measurement": {"env_steps_per_second": 8000.0},
            },
        },
        python_profile_summary=Phase1ProfileSummary(
            total_time_seconds=1.0,
            step_batch_cumulative_seconds=1.0,
            raw_conversion_cumulative_seconds=0.1,
            flat_observe_cumulative_seconds=0.1,
            build_step_buffers_cumulative_seconds=0.1,
            raw_object_conversion_still_dominant=False,
        ),
    )

    assert gate.benchmark_source_of_truth is True
    assert gate.phase2_unblocked is True


def test_phase1_gate_does_not_fail_when_vecenv_64_exceeds_old_upper_band(tmp_path):
    baseline_dir = tmp_path / "phase0"
    baseline_dir.mkdir()
    (baseline_dir / "phase0_packet.json").write_text(
        json.dumps(
            {
                "gate_status": {
                    "benchmark_host_class": "wsl2",
                    "benchmark_source_of_truth": True,
                }
            }
        ),
        encoding="utf-8",
    )
    (baseline_dir / "bridge_64env.json").write_text(
        json.dumps({"batch": {"env_steps_per_second": 1600.0}}),
        encoding="utf-8",
    )
    (baseline_dir / "vecenv_64env.json").write_text(
        json.dumps({"measurement": {"env_steps_per_second": 1400.0}}),
        encoding="utf-8",
    )

    gate = evaluate_phase1_gate(
        phase0_baseline_dir=baseline_dir,
        bridge_reports={
            1: _bridge_report(2000.0),
            16: _bridge_report(4000.0),
            64: _bridge_report(11000.0),
        },
        vecenv_reports={
            16: _vecenv_report(8000.0),
            64: _vecenv_report(12402.49),
        },
        python_profile_summary=Phase1ProfileSummary(
            total_time_seconds=1.0,
            step_batch_cumulative_seconds=1.0,
            raw_conversion_cumulative_seconds=0.1,
            flat_observe_cumulative_seconds=0.1,
            build_step_buffers_cumulative_seconds=0.1,
            raw_object_conversion_still_dominant=False,
        ),
    )

    assert gate.vecenv_threshold_met is True
    assert "vecenv_threshold_not_met" not in gate.blockers
    assert gate.phase2_unblocked is True
