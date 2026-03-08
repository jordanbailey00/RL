from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_vecenv_reset_step_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "vecenv_reset_step.json"
    result = run_script(
        "run_vecenv_smoke.py",
        "--config",
        "configs/train/train_baseline_v0.yaml",
        "--mode",
        "reset-step",
        "--output",
        str(output),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"VecEnv reset/step smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["mode"] == "reset-step"
    assert int(payload["env_count"]) == 4
    assert payload["initial_observation_shape"][0] == 4
    assert payload["initial_reward_shape"] == [4]
    assert payload["initial_terminal_shape"] == [4]
    assert payload["initial_truncation_shape"] == [4]
    assert payload["teacher_action_shape"] == [4]
    assert payload["info_count"] == 4
    assert payload["agent_ids"] == [0, 1, 2, 3]
    assert payload["mask_dtype"] == "bool"
    assert payload["next_observation_shape"][0] == 4
    assert payload["next_info_count"] == 4
    assert payload["next_events"] == ["step", "step", "step", "step"]
