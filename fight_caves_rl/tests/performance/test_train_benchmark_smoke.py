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
    assert str(payload["metric_contract_id"]) == "train_benchmark_production_v1"
    assert payload["context"]["benchmark_profile_id"] == "official_profile_v0"
    assert payload["context"]["logging_mode"] == "recorded_per_measurement"
    assert payload["context"]["replay_mode"] == "disabled"
    measurements = {entry["logging_mode"]: entry for entry in payload["measurements"]}
    assert set(measurements) == {"disabled", "standard"}
    assert str(measurements["disabled"]["metric_scope"]) == "production_fast_path_v1"
    assert float(measurements["disabled"]["production_env_steps_per_second"]) > 0.0
    assert float(measurements["standard"]["production_env_steps_per_second"]) > 0.0
    assert float(measurements["disabled"]["wall_clock_env_steps_per_second"]) > 0.0
    assert measurements["disabled"]["runner_stage_seconds"] == {}
    assert measurements["disabled"]["trainer_bucket_totals"] == {}
    assert float(payload["sps_ratio_vs_disabled"]["standard"]) > 0.0


def test_train_benchmark_core_runner_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "train_benchmark_core.json"
    result = run_script(
        "benchmark_train.py",
        "--config",
        "configs/benchmark/train_1024env_v0.yaml",
        "--runner-mode",
        "core_inprocess_v1",
        "--env-count",
        "2",
        "--total-timesteps",
        "8",
        "--logging-modes",
        "disabled",
        "--output",
        str(output),
        timeout_seconds=180.0,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Core train benchmark smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert str(payload["metric_contract_id"]) == "train_benchmark_production_v1"
    measurement = payload["measurements"][0]
    assert measurement["runner_mode"] == "core_inprocess_v1"
    assert measurement["logging_mode"] == "disabled"
    assert str(measurement["metric_scope"]) == "production_fast_path_v1"
    assert float(measurement["production_env_steps_per_second"]) > 0.0
    assert float(measurement["wall_clock_env_steps_per_second"]) > 0.0
    assert int(measurement["wall_clock_elapsed_nanos"]) >= int(measurement["elapsed_nanos"])
    assert float(measurement["runner_stage_seconds"]["evaluate_seconds"]) >= 0.0
    assert float(measurement["runner_stage_seconds"]["train_seconds"]) >= 0.0
    assert "eval_tensor_copy" in measurement["trainer_bucket_totals"]
    assert "train_policy_forward" in measurement["trainer_bucket_totals"]
    assert int(measurement["artifact_count"]) == 0
    assert str(measurement["checkpoint_path"]) == ""


def test_train_benchmark_prototype_runner_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "train_benchmark_prototype.json"
    result = run_script(
        "benchmark_train.py",
        "--config",
        "configs/benchmark/train_1024env_v0.yaml",
        "--runner-mode",
        "prototype_sync_v1",
        "--env-count",
        "2",
        "--total-timesteps",
        "8",
        "--logging-modes",
        "disabled",
        "--output",
        str(output),
        timeout_seconds=180.0,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Prototype train benchmark smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert str(payload["metric_contract_id"]) == "train_benchmark_production_v1"
    measurement = payload["measurements"][0]
    assert measurement["runner_mode"] == "prototype_sync_v1"
    assert measurement["logging_mode"] == "disabled"
    assert str(measurement["metric_scope"]) == "production_fast_path_v1"
    assert float(measurement["production_env_steps_per_second"]) > 0.0
    assert float(measurement["wall_clock_env_steps_per_second"]) > 0.0
    assert float(measurement["final_evaluate_seconds"]) == 0.0
    assert float(measurement["runner_stage_seconds"]["evaluate_seconds"]) > 0.0
    assert float(measurement["runner_stage_seconds"]["train_seconds"]) > 0.0
    assert "rollout_policy_forward" in measurement["trainer_bucket_totals"]
    assert "update_policy_forward" in measurement["trainer_bucket_totals"]
    assert int(measurement["artifact_count"]) == 0
    assert str(measurement["checkpoint_path"]) == ""
