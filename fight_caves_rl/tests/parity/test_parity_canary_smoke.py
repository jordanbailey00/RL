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
        "parity_action_rejection",
        "parity_prayer_toggle_timing",
        "parity_single_wave",
        "parity_jad_healer",
        "parity_terminal_tick_cap",
        "parity_tzkek_split",
    }
    for scenario in scenarios.values():
        assert scenario["passed"] is True
        assert scenario["wrapper_matches_raw"] is True
        assert scenario["scripted_matches_wrapper"] is True
        assert scenario["oracle_matches_v2_fast"] is True
        assert (
            scenario["oracle_mechanics_digest"]
            == scenario["v2_fast_mechanics_digest"]
        )
        if scenario["expected_digest_matches"] is not None:
            assert scenario["expected_digest_matches"] is True
        if scenario["expected_mechanics_digest_matches"] is not None:
            assert scenario["expected_mechanics_digest_matches"] is True
        if scenario["final_relative_tick_matches"] is not None:
            assert scenario["final_relative_tick_matches"] is True
        assert scenario["mechanics_first_mismatch"] is None
        assert scenario["mechanics_divergence_artifact"] is None
