import json
import subprocess
import sys
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths
from fight_caves_rl.envs.schema import FIGHT_CAVE_EPISODE_START_CONTRACT


def test_wrapper_reset_matches_sim_contract(tmp_path: Path):
    _require_live_runtime()

    payload = _collect_step_trace(tmp_path / "wrapper_reset_trace.json", seed=123456789)
    observation = payload["observation"]
    info = {
        "episode_state": payload["episode_state"],
    }

    assert observation["episode_seed"] == 123456789
    assert observation["player"]["run_energy_percent"] == 100
    assert observation["player"]["running"] is True
    assert observation["player"]["consumables"]["shark_count"] == FIGHT_CAVE_EPISODE_START_CONTRACT.default_sharks
    assert observation["player"]["consumables"]["prayer_potion_dose_count"] == (
        FIGHT_CAVE_EPISODE_START_CONTRACT.default_prayer_potions * 4
    )
    assert info["episode_state"]["seed"] == 123456789


def _collect_step_trace(output: Path, *, seed: int) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "collect_step_trace.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--mode",
            "wrapper",
            "--seed",
            str(seed),
            "--action",
            "0",
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
            f"Step trace collection failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
