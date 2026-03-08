from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import (
    load_json,
    offline_wandb_env,
    require_live_runtime,
    run_script,
)

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_scripted_baseline_smoke(tmp_path: Path):
    require_live_runtime()

    output_path = tmp_path / "scripted_smoke.json"
    result = run_script(
        "smoke_scripted.py",
        "--trace-pack",
        "parity_single_wave_v0",
        "--output",
        str(output_path),
        env=offline_wandb_env(tmp_path, tags="smoke,scripted"),
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Scripted smoke failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output_path)
    assert payload["trace_pack"] == "parity_single_wave_v0"
    assert payload["steps_executed"] > 0
    assert payload["semantic_digest"] == payload["expected_semantic_digest"]
