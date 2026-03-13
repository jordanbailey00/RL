from __future__ import annotations

from pathlib import Path

import pytest

from fight_caves_rl.contracts.parity_trace_schema import MECHANICS_PARITY_TRACE_FIELD_NAMES
from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_mechanics_parity_trace_collection_smoke(tmp_path: Path):
    require_live_runtime()

    oracle_output = tmp_path / "oracle_mechanics_trace.json"
    fast_output = tmp_path / "fast_mechanics_trace.json"

    for mode, output in (("oracle", oracle_output), ("v2_fast", fast_output)):
        result = run_script(
            "collect_mechanics_parity_trace.py",
            "--mode",
            mode,
            "--trace-pack",
            "wait_only_16_v0",
            "--output",
            str(output),
            timeout_seconds=300.0,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"Mechanics parity trace collection failed for mode={mode}.\n"
                f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
            )

    oracle = load_json(oracle_output)
    fast = load_json(fast_output)

    assert oracle["runtime_path"] == "oracle"
    assert fast["runtime_path"] == "v2_fast"
    assert oracle["schema_id"] == fast["schema_id"] == "fight_caves_mechanics_parity_trace_v1"
    assert oracle["schema_version"] == fast["schema_version"] == 1
    assert oracle["trace_pack"] == fast["trace_pack"] == "wait_only_16_v0"
    assert oracle["seed"] == fast["seed"]
    assert oracle["record_count"] == fast["record_count"] == 17
    assert oracle["field_names"] == fast["field_names"] == list(MECHANICS_PARITY_TRACE_FIELD_NAMES)
    assert sorted(oracle["records"][0].keys()) == sorted(MECHANICS_PARITY_TRACE_FIELD_NAMES)
    assert sorted(fast["records"][0].keys()) == sorted(MECHANICS_PARITY_TRACE_FIELD_NAMES)
