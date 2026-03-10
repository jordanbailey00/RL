from fight_caves_rl.benchmarks.phase2_prototype_packet import evaluate_phase2_prototype_gate


def _production_reports(*, host_class: str, sps_16: float, sps_64: float) -> dict[int, dict[str, object]]:
    return {
        16: {
            "context": {"hardware_profile": {"host_class": host_class}},
            "measurements": [
                {
                    "logging_mode": "disabled",
                    "production_env_steps_per_second": sps_16,
                }
            ],
        },
        64: {
            "context": {"hardware_profile": {"host_class": host_class}},
            "measurements": [
                {
                    "logging_mode": "disabled",
                    "production_env_steps_per_second": sps_64,
                }
            ],
        },
    }


def _learner_ceiling_report(
    *,
    host_class: str,
    sps_16: float,
    sps_64: float,
) -> dict[str, object]:
    return {
        "context": {"hardware_profile": {"host_class": host_class}},
        "measurements": [
            {
                "env_count": 16,
                "diagnostic_env_steps_per_second": sps_16,
            },
            {
                "env_count": 64,
                "diagnostic_env_steps_per_second": sps_64,
            },
        ],
    }


def test_phase2_prototype_gate_continues_trainer_redesign_when_scaling_is_flat():
    gate = evaluate_phase2_prototype_gate(
        production_reports=_production_reports(
            host_class="linux_native",
            sps_16=420.0,
            sps_64=400.0,
        ),
        learner_ceiling_report=_learner_ceiling_report(
            host_class="linux_native",
            sps_16=145.0,
            sps_64=144.0,
        ),
    )

    assert gate.next_move == "continue_trainer_redesign"
    assert gate.blockers == ()
    assert gate.prototype_scaling_ratio_64_vs_16 is not None
    assert gate.prototype_scaling_ratio_64_vs_16 < 1.10


def test_phase2_prototype_gate_escalates_when_prototype_64_is_below_bar():
    gate = evaluate_phase2_prototype_gate(
        production_reports=_production_reports(
            host_class="linux_native",
            sps_16=220.0,
            sps_64=200.0,
        ),
        learner_ceiling_report=_learner_ceiling_report(
            host_class="linux_native",
            sps_16=145.0,
            sps_64=144.0,
        ),
    )

    assert gate.next_move == "deeper_trainer_replacement"
    assert "prototype_64_below_trainer_bar" in gate.blockers


def test_phase2_prototype_gate_can_recommend_transport_or_topology_review():
    gate = evaluate_phase2_prototype_gate(
        production_reports=_production_reports(
            host_class="linux_native",
            sps_16=300.0,
            sps_64=360.0,
        ),
        learner_ceiling_report=_learner_ceiling_report(
            host_class="linux_native",
            sps_16=200.0,
            sps_64=220.0,
        ),
    )

    assert gate.next_move == "review_transport_or_topology"
    assert gate.blockers == ()
