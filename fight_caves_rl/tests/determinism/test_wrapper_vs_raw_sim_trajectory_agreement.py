import json
import subprocess
import sys
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths


def test_wrapper_vs_raw_sim_trajectory_agreement(tmp_path: Path):
    _require_live_runtime()

    wrapper_trace = _collect_trajectory(
        tmp_path / "wrapper_trajectory.json",
        mode="wrapper",
        trace_pack="parity_single_wave_v0",
    )
    raw_trace = _collect_trajectory(
        tmp_path / "raw_trajectory.json",
        mode="raw",
        trace_pack="parity_single_wave_v0",
    )

    assert wrapper_trace["semantic_episode_state"] == raw_trace["semantic_episode_state"]
    assert wrapper_trace["semantic_initial_observation"] == raw_trace["semantic_initial_observation"]
    assert wrapper_trace["summary"] == raw_trace["summary"]
    assert len(wrapper_trace["steps"]) == len(raw_trace["steps"])
    for wrapper_step, raw_step in zip(wrapper_trace["steps"], raw_trace["steps"], strict=True):
        assert wrapper_step["action"] == raw_step["action"]
        assert wrapper_step["semantic_observation"] == raw_step["semantic_observation"]
        assert wrapper_step["action_result"] == raw_step["action_result"]
        assert wrapper_step["semantic_visible_targets"] == raw_step["semantic_visible_targets"]
        assert wrapper_step["terminal_reason"] == raw_step["terminal_reason"]


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
            f"Trajectory collection failed for mode={mode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
