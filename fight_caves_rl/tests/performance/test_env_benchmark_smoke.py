from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_env_benchmark_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "env_benchmark.json"
    result = run_script(
        "benchmark_env.py",
        "--config",
        "configs/benchmark/vecenv_256env_v0.yaml",
        "--env-count",
        "8",
        "--wrapper-env-count",
        "1",
        "--rounds",
        "16",
        "--output",
        str(output),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Env benchmark smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["context"]["benchmark_profile_id"] == "official_profile_v0"
    assert payload["context"]["logging_mode"] == "benchmark_standard"
    assert payload["context"]["replay_mode"] == "disabled"
    assert float(payload["wrapper"]["env_steps_per_second"]) > 0.0
    assert float(payload["measurement"]["env_steps_per_second"]) > 0.0
    assert float(payload["speedup_vs_wrapper"]) > 0.0
