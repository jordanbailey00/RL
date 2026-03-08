import json
import subprocess
import sys
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths


def test_fixed_seed_reset_reproducibility(tmp_path: Path):
    _require_live_runtime()

    payload = _collect_reset_repro(tmp_path / "reset_repro.json", seed=11_001)

    assert payload["first_episode_state"] == payload["second_episode_state"]
    assert payload["first_observation"] == payload["second_observation"]


def _collect_reset_repro(output: Path, *, seed: int) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "collect_reset_repro.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--seed",
            str(seed),
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
            f"Reset reproducibility collection failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
