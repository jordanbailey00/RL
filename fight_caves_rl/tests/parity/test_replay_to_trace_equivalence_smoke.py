from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_replay_to_trace_equivalence_smoke(tmp_path: Path):
    require_live_runtime()

    output = tmp_path / "parity_canary_report.json"
    result = run_script(
        "run_parity_canary.py",
        "--config",
        "configs/eval/parity_canary_v0.yaml",
        "--output",
        str(output),
        timeout_seconds=300.0,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Replay-to-trace parity run failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["all_passed"] is True
    for scenario in payload["scenarios"]:
        assert scenario["scripted_matches_wrapper"] is True
        assert scenario["scripted_summary"]["semantic_digest"] == scenario["wrapper_semantic_digest"]
        assert (
            scenario["scripted_summary"]["final_relative_tick"]
            == scenario["wrapper_summary"]["final_relative_tick"]
        )
        assert (
            scenario["scripted_summary"]["completed_all_steps"]
            == scenario["wrapper_summary"]["completed_all_steps"]
        )
