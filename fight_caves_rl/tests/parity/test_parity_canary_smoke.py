from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_parity_canary_smoke(tmp_path: Path):
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
            f"Parity canary run failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )

    payload = load_json(output)
    assert payload["config_id"] == "parity_canary_v0"
    assert payload["seed_pack"] == "parity_reference_v0"
    assert payload["comparison_mode"] == "semantic_digest"
    assert payload["all_passed"] is True
    scenarios = {entry["scenario_id"]: entry for entry in payload["scenarios"]}
    assert set(scenarios) == {
        "parity_single_wave",
        "parity_jad_healer",
        "parity_tzkek_split",
    }
    for scenario in scenarios.values():
        assert scenario["passed"] is True
        assert scenario["wrapper_matches_raw"] is True
        assert scenario["expected_digest_matches"] is True
        assert scenario["final_relative_tick_matches"] is True
