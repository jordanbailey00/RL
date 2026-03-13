from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_fast_env_benchmark_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "fast_env_benchmark.json"
    result = run_script(
        "benchmark_env.py",
        "--config",
        "configs/benchmark/fast_env_v2.yaml",
        "--env-count",
        "8",
        "--rounds",
        "16",
        "--mode",
        "vecenv",
        "--output",
        str(output),
        timeout_seconds=180.0,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Fast env benchmark smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["label"] == "vecenv_lockstep"
    assert payload["vecenv_topology"]["env_backend"] == "v2_fast"
    assert payload["vecenv_topology"]["backend"] == "subprocess"
    assert payload["vecenv_topology"]["transport_mode"] == "shared_memory_v1"
    assert int(payload["vecenv_topology"]["worker_count"]) > 0
    assert float(payload["env_steps_per_second"]) > 0.0


def test_fast_train_benchmark_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "fast_train_benchmark.json"
    result = run_script(
        "benchmark_train.py",
        "--config",
        "configs/benchmark/fast_train_v2.yaml",
        "--env-count",
        "4",
        "--total-timesteps",
        "16",
        "--logging-modes",
        "disabled",
        "--output",
        str(output),
        timeout_seconds=240.0,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Fast train benchmark smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    measurement = payload["measurements"][0]
    assert measurement["runner_mode"] == "smoke_subprocess_v1"
    assert measurement["vecenv_topology"]["env_backend"] == "v2_fast"
    assert measurement["vecenv_topology"]["backend"] == "subprocess"
    assert measurement["vecenv_topology"]["transport_mode"] == "shared_memory_v1"
    assert int(measurement["vecenv_topology"]["worker_count"]) > 0
    assert float(measurement["production_env_steps_per_second"]) > 0.0
