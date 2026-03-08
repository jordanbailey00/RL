from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_long_run_vector_stability(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "vecenv_long_run.json"
    result = run_script(
        "run_vecenv_smoke.py",
        "--config",
        "configs/train/train_baseline_v0.yaml",
        "--mode",
        "long-run",
        "--output",
        str(output),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"VecEnv long-run smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["mode"] == "long-run"
    assert int(payload["env_count"]) == 4
    assert payload["last_reward_shape"] == [4]
    assert min(int(value) for value in payload["episodes_started"]) >= 2
