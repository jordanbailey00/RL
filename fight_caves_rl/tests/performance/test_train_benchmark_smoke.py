from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_train_benchmark_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "train_benchmark.json"
    result = run_script(
        "benchmark_train.py",
        "--config",
        "configs/benchmark/train_1024env_v0.yaml",
        "--env-count",
        "2",
        "--total-timesteps",
        "8",
        "--logging-modes",
        "disabled,standard",
        "--output",
        str(output),
        timeout_seconds=180.0,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Train benchmark smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["context"]["benchmark_profile_id"] == "official_profile_v0"
    assert payload["context"]["logging_mode"] == "recorded_per_measurement"
    assert payload["context"]["replay_mode"] == "disabled"
    measurements = {entry["logging_mode"]: entry for entry in payload["measurements"]}
    assert set(measurements) == {"disabled", "standard"}
    assert float(measurements["disabled"]["env_steps_per_second"]) > 0.0
    assert float(measurements["standard"]["env_steps_per_second"]) > 0.0
    assert float(payload["sps_ratio_vs_disabled"]["standard"]) > 0.0
