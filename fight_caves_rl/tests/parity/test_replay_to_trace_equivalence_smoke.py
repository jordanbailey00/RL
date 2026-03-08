import json
import subprocess
import sys
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths
from fight_caves_rl.replay.trace_packs import resolve_trace_pack, serialize_action


def test_replay_to_trace_equivalence_smoke(tmp_path: Path):
    _require_live_runtime()

    trace_pack = resolve_trace_pack("parity_single_wave_v0")
    payload = _collect_trajectory(
        tmp_path / "parity_single_wave_wrapper.json",
        mode="wrapper",
        trace_pack=trace_pack.identity.contract_id,
    )

    assert len(payload["steps"]) == len(trace_pack.steps)
    assert payload["summary"]["final_relative_tick"] == len(trace_pack.steps)
    for recorded_step, expected_step in zip(payload["steps"], trace_pack.steps, strict=True):
        assert recorded_step["action"] == serialize_action(expected_step.action)


def _collect_trajectory(output: Path, *, mode: str, trace_pack: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "collect_trajectory_trace.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            mode,
            "--trace-pack",
            trace_pack,
            "--output",
            str(output),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"Trajectory collection failed for replay/trace smoke.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
