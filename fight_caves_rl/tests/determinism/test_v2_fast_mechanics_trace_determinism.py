from pathlib import Path

import pytest

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


@pytest.mark.parametrize(
    ("trace_pack_id", "seed"),
    (
        ("parity_single_wave_v0", 11_001),
        ("parity_action_rejection_v0", 33_003),
        ("parity_prayer_toggle_timing_v0", 11_001),
        ("parity_terminal_tick_cap_v0", 11_001),
    ),
)
def test_v2_fast_mechanics_trace_determinism(
    tmp_path: Path,
    trace_pack_id: str,
    seed: int,
):
    require_live_runtime()

    first_output = tmp_path / f"{trace_pack_id}_first.json"
    second_output = tmp_path / f"{trace_pack_id}_second.json"
    for output in (first_output, second_output):
        result = run_script(
            "collect_mechanics_parity_trace.py",
            "--mode",
            "v2_fast",
            "--trace-pack",
            trace_pack_id,
            "--seed",
            str(seed),
            "--output",
            str(output),
            timeout_seconds=300.0,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Mechanics trace collection failed for {trace_pack_id}.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

    first = load_json(first_output)
    second = load_json(second_output)

    assert first["runtime_path"] == "v2_fast"
    assert second["runtime_path"] == "v2_fast"
    assert first["record_count"] == second["record_count"]
    assert first["semantic_digest"] == second["semantic_digest"]
    assert first["records"] == second["records"]
