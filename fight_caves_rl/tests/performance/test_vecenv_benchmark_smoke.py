from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_vecenv_benchmark_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "vecenv_benchmark.json"
    result = run_script(
        "benchmark_env.py",
        "--config",
        "configs/benchmark/vecenv_256env_v0.yaml",
        "--env-count",
        "8",
        "--rounds",
        "16",
        "--output",
        str(output),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"VecEnv benchmark smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["bridge_protocol_id"] == "fight_caves_bridge_v1"
    assert int(payload["bridge_protocol_version"]) == 1
    assert int(payload["env_count"]) == 8
    assert float(payload["measurement"]["env_steps_per_second"]) > 0.0
