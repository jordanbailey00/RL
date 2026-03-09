from fight_caves_rl.benchmarks.phase0_packet import evaluate_phase0_gate


def test_phase0_gate_blocks_when_native_linux_source_of_truth_is_missing():
    sim_report = {
        "runtime_metadata": {
            "host_class": "wsl2",
            "performance_source_of_truth": False,
        },
        "throughput": {"ticks_per_second": 9000.0},
        "per_worker_ceiling": {"batched_env_steps_per_second": 18000.0},
    }
    bridge_reports = {1: {}, 16: {}, 64: {}}
    vecenv_reports = {1: {}, 16: {}, 64: {}}
    train_reports = {4: {}, 16: {}, 64: {}}

    gate = evaluate_phase0_gate(
        sim_report=sim_report,
        bridge_reports=bridge_reports,
        vecenv_reports=vecenv_reports,
        train_reports=train_reports,
    )

    assert gate.native_linux_source_of_truth is False
    assert gate.phase1_unblocked is False
    assert "native_linux_source_of_truth_missing" in gate.blockers
    assert gate.workers_needed_for_100k == 6


def test_phase0_gate_unblocks_when_required_artifacts_exist_on_native_linux():
    sim_report = {
        "runtime_metadata": {
            "host_class": "linux_native",
            "performance_source_of_truth": True,
        },
        "throughput": {"ticks_per_second": 12000.0},
        "per_worker_ceiling": {"batched_env_steps_per_second": 25000.0},
    }
    bridge_reports = {1: {}, 16: {}, 64: {}}
    vecenv_reports = {1: {}, 16: {}, 64: {}}
    train_reports = {4: {}, 16: {}, 64: {}}

    gate = evaluate_phase0_gate(
        sim_report=sim_report,
        bridge_reports=bridge_reports,
        vecenv_reports=vecenv_reports,
        train_reports=train_reports,
    )

    assert gate.native_linux_source_of_truth is True
    assert gate.phase1_unblocked is True
    assert gate.blockers == ()
    assert gate.workers_needed_for_100k == 4
