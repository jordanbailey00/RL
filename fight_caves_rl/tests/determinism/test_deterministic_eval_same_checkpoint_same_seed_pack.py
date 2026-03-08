import json
import subprocess
import sys
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths


def test_deterministic_eval_same_checkpoint_same_seed_pack(tmp_path: Path):
    _require_live_runtime()

    checkpoint = tmp_path / "scripted_checkpoint.json"
    checkpoint.write_text(
        json.dumps(
            {
                "checkpoint_schema": "scripted_policy_checkpoint_v0",
                "policy_id": "wait_only_v0",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    first = _collect_eval_summary(
        tmp_path / "eval_first.json",
        checkpoint=checkpoint,
        seed_pack="bootstrap_smoke",
        max_steps=32,
    )
    second = _collect_eval_summary(
        tmp_path / "eval_second.json",
        checkpoint=checkpoint,
        seed_pack="bootstrap_smoke",
        max_steps=32,
    )

    assert first == second


def _collect_eval_summary(
    output: Path,
    *,
    checkpoint: Path,
    seed_pack: str,
    max_steps: int,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "collect_seedpack_eval.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--seed-pack",
            seed_pack,
            "--max-steps",
            str(max_steps),
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
            f"Seed-pack eval collection failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return json.loads(output.read_text(encoding="utf-8"))


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
