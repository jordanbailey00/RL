import json
import subprocess
import sys
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths


def test_wrapper_step_matches_raw_sim_trace(tmp_path: Path):
    _require_live_runtime()

    wrapper_trace = _collect_trace(tmp_path / "wrapper_trace.json", mode="wrapper", seed=987654321, action=0)
    raw_trace = _collect_trace(tmp_path / "raw_trace.json", mode="raw", seed=987654321, action=0)

    assert wrapper_trace["seed"] == raw_trace["seed"] == 987654321
    assert wrapper_trace["action"] == raw_trace["action"] == 0
    assert wrapper_trace["episode_state"] == raw_trace["episode_state"]
    assert wrapper_trace["observation"] == raw_trace["observation"]
    assert wrapper_trace["next_observation"] == raw_trace["next_observation"]
    assert wrapper_trace["action_result"] == raw_trace["action_result"]
    assert wrapper_trace["visible_targets"] == raw_trace["visible_targets"]
    assert wrapper_trace["reward"] == raw_trace["reward"] == 0.0
    assert wrapper_trace["terminated"] is raw_trace["terminated"] is False
    assert wrapper_trace["truncated"] is raw_trace["truncated"] is False
    assert wrapper_trace["terminal_reason"] is raw_trace["terminal_reason"] is None
    assert wrapper_trace["terminal_reason_inferred"] is raw_trace["terminal_reason_inferred"] is False
    assert wrapper_trace["episode_steps"] == raw_trace["episode_steps"] == 1


def _collect_trace(output: Path, mode: str, seed: int, action: int) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "collect_step_trace.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            mode,
            "--seed",
            str(seed),
            "--action",
            str(action),
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
            f"Trace collection failed for mode={mode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
