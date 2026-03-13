from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_fast_vecenv_reset_step_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "fast_vecenv_reset_step.json"
    result = run_script(
        "run_vecenv_smoke.py",
        "--config",
        "configs/train/smoke_fast_v2.yaml",
        "--mode",
        "reset-step",
        "--output",
        str(output),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Fast vecenv reset/step smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["mode"] == "reset-step"
    assert payload["backend"] == "embedded"
    assert int(payload["env_count"]) == 4
    assert payload["initial_observation_shape"] == [4, 134]
    assert payload["initial_nonempty_info_count"] == 0
    assert payload["next_nonempty_info_count"] == 0
    assert payload["next_events"] == []
    assert payload["topology"]["env_backend"] == "v2_fast"
    assert payload["topology"]["worker_count"] == 1
    assert payload["topology"]["transport_mode"] == "embedded_jvm"


def test_fast_vecenv_subprocess_shared_memory_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "fast_vecenv_subprocess_reset_step.json"
    result = run_script(
        "run_vecenv_smoke.py",
        "--config",
        "configs/train/smoke_fast_v2.yaml",
        "--mode",
        "long-run",
        "--backend",
        "subprocess",
        "--output",
        str(output),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Fast subprocess vecenv smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["mode"] == "long-run"
    assert payload["backend"] == "subprocess"
    assert int(payload["env_count"]) == 4
    assert payload["topology"]["env_backend"] == "v2_fast"
    assert payload["topology"]["transport_mode"] == "shared_memory_v1"
    assert payload["topology"]["worker_count"] == 2
    assert payload["topology"]["worker_env_counts"] == [2, 2]
    assert payload["last_reward_shape"] == [4]
