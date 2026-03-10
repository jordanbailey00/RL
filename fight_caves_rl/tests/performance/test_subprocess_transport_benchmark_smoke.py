from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.envs.shared_memory_transport import (
    PIPE_PICKLE_TRANSPORT_MODE,
    SHARED_MEMORY_TRANSPORT_MODE,
)
from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_subprocess_transport_benchmark_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "subprocess_transport_benchmark.json"
    result = run_script(
        "benchmark_subprocess_transport.py",
        "--config",
        "configs/benchmark/vecenv_256env_v0.yaml",
        "--env-count",
        "8",
        "--rounds",
        "16",
        "--output",
        str(output),
        timeout_seconds=180.0,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Subprocess transport benchmark smoke failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    payload = load_json(output)
    measurements = {entry["transport_mode"]: entry for entry in payload["measurements"]}
    assert payload["context"]["benchmark_profile_id"] == "official_profile_v0"
    assert set(measurements) == {
        PIPE_PICKLE_TRANSPORT_MODE,
        SHARED_MEMORY_TRANSPORT_MODE,
    }
    assert float(measurements[PIPE_PICKLE_TRANSPORT_MODE]["env_steps_per_second"]) > 0.0
    assert float(measurements[SHARED_MEMORY_TRANSPORT_MODE]["env_steps_per_second"]) > 0.0
    assert float(payload["speedup_vs_pipe_pickle"][SHARED_MEMORY_TRANSPORT_MODE]) > 0.0
