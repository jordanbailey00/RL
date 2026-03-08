import json
import subprocess
import sys
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths

pytestmark = pytest.mark.usefixtures("disable_subprocess_capture")


def test_batch_bridge_matches_reference_lockstep_trace(tmp_path: Path):
    _require_live_runtime()

    reference = _collect_trace(tmp_path / "reference_batch_trace.json", mode="reference")
    batch = _collect_trace(tmp_path / "batch_trace.json", mode="batch")

    assert reference["bridge_protocol"] == batch["bridge_protocol"]
    assert reference["semantic_episode_states"] == batch["semantic_episode_states"]
    assert reference["steps"] == batch["steps"]
    assert reference["semantic_digest"] == batch["semantic_digest"]


def _collect_trace(output: Path, *, mode: str) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "collect_batch_trace.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            mode,
            "--env-count",
            "3",
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
            f"Batch trace collection failed for mode={mode}.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
