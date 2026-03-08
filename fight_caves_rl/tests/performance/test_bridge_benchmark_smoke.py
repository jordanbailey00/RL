from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_bridge_benchmark_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "bridge_benchmark.json"
    result = run_script(
        "benchmark_bridge.py",
        "--config",
        "configs/benchmark/bridge_1env_v0.yaml",
        "--rounds",
        "256",
        "--output",
        str(output),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Bridge benchmark smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["bridge_protocol_id"] == "fight_caves_bridge_v1"
    assert int(payload["bridge_protocol_version"]) == 1
    assert float(payload["reference"]["env_steps_per_second"]) > 0.0
    assert float(payload["batch"]["env_steps_per_second"]) > 0.0
    assert float(payload["speedup_vs_reference"]) > 1.0
