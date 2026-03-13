from fight_caves_rl.benchmarks.phase2_packet import evaluate_phase2_gate
from fight_caves_rl.envs.shared_memory_transport import (
    PIPE_PICKLE_TRANSPORT_MODE,
    SHARED_MEMORY_TRANSPORT_MODE,
)


def _transport_report(host_class: str, *, pipe_64: float, shared_64: float) -> dict[int, dict[str, object]]:
    return {
        16: {
            "context": {"hardware_profile": {"host_class": host_class}},
            "measurements": [
                {"transport_mode": PIPE_PICKLE_TRANSPORT_MODE, "env_steps_per_second": 5000.0},
                {"transport_mode": SHARED_MEMORY_TRANSPORT_MODE, "env_steps_per_second": 6000.0},
            ],
        },
        64: {
            "context": {"hardware_profile": {"host_class": host_class}},
            "measurements": [
                {"transport_mode": PIPE_PICKLE_TRANSPORT_MODE, "env_steps_per_second": pipe_64},
                {"transport_mode": SHARED_MEMORY_TRANSPORT_MODE, "env_steps_per_second": shared_64},
            ],
        },
    }


def _train_reports(
    *,
    pipe_16: float,
    shared_16: float,
    pipe_64: float,
    shared_64: float,
) -> dict[tuple[str, int], dict[str, object]]:
    return {
        (PIPE_PICKLE_TRANSPORT_MODE, 16): {
            "context": {"hardware_profile": {"host_class": "linux_native"}},
            "measurements": [
                {
                    "logging_mode": "disabled",
                    "production_env_steps_per_second": pipe_16,
                }
            ],
        },
        (SHARED_MEMORY_TRANSPORT_MODE, 16): {
            "context": {"hardware_profile": {"host_class": "linux_native"}},
            "measurements": [
                {
                    "logging_mode": "disabled",
                    "production_env_steps_per_second": shared_16,
                }
            ],
        },
        (PIPE_PICKLE_TRANSPORT_MODE, 64): {
            "context": {"hardware_profile": {"host_class": "linux_native"}},
            "measurements": [
                {
                    "logging_mode": "disabled",
                    "production_env_steps_per_second": pipe_64,
                }
            ],
        },
        (SHARED_MEMORY_TRANSPORT_MODE, 64): {
            "context": {"hardware_profile": {"host_class": "linux_native"}},
            "measurements": [
                {
                    "logging_mode": "disabled",
                    "production_env_steps_per_second": shared_64,
                }
            ],
        },
    }


def test_phase2_gate_blocks_when_train_signal_is_too_weak():
    gate = evaluate_phase2_gate(
        transport_reports=_transport_report("linux_native", pipe_64=8000.0, shared_64=10000.0),
        train_reports=_train_reports(
            pipe_16=150.0,
            shared_16=150.0,
            pipe_64=180.0,
            shared_64=190.0,
        ),
    )

    assert gate.transport_signal_strong_enough is True
    assert gate.train_signal_strong_enough is False
    assert gate.wc_p2_03_unblocked is False
    assert "train_signal_too_weak" in gate.blockers


def test_phase2_gate_unblocks_when_transport_and_train_signals_clear_thresholds():
    gate = evaluate_phase2_gate(
        transport_reports=_transport_report("linux_native", pipe_64=8000.0, shared_64=10400.0),
        train_reports=_train_reports(
            pipe_16=400.0,
            shared_16=450.0,
            pipe_64=600.0,
            shared_64=900.0,
        ),
    )

    assert gate.transport_signal_strong_enough is True
    assert gate.train_signal_strong_enough is True
    assert gate.scaling_signal_strong_enough is True
    assert gate.wc_p2_03_unblocked is True
    assert gate.blockers == ()


def test_phase2_gate_accepts_legacy_train_metric_field():
    gate = evaluate_phase2_gate(
        transport_reports=_transport_report("linux_native", pipe_64=8000.0, shared_64=10400.0),
        train_reports={
            (PIPE_PICKLE_TRANSPORT_MODE, 16): {
                "context": {"hardware_profile": {"host_class": "linux_native"}},
                "measurements": [{"logging_mode": "disabled", "env_steps_per_second": 400.0}],
            },
            (SHARED_MEMORY_TRANSPORT_MODE, 16): {
                "context": {"hardware_profile": {"host_class": "linux_native"}},
                "measurements": [{"logging_mode": "disabled", "env_steps_per_second": 450.0}],
            },
            (PIPE_PICKLE_TRANSPORT_MODE, 64): {
                "context": {"hardware_profile": {"host_class": "linux_native"}},
                "measurements": [{"logging_mode": "disabled", "env_steps_per_second": 600.0}],
            },
            (SHARED_MEMORY_TRANSPORT_MODE, 64): {
                "context": {"hardware_profile": {"host_class": "linux_native"}},
                "measurements": [{"logging_mode": "disabled", "env_steps_per_second": 900.0}],
            },
        },
    )

    assert gate.train_signal_strong_enough is True
    assert gate.scaling_signal_strong_enough is True
