from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths
from fight_caves_rl.tests.smoke._helpers import load_json, offline_wandb_env, run_script


def test_replay_eval_equivalence(tmp_path: Path):
    _require_live_runtime()

    train_summary_path = tmp_path / "train_summary.json"
    train_result = run_script(
        "train.py",
        "--config",
        "configs/train/smoke_ppo_v0.yaml",
        "--total-timesteps",
        "4",
        "--data-dir",
        str(tmp_path / "train_artifacts"),
        "--output",
        str(train_summary_path),
        env=offline_wandb_env(tmp_path, tags="determinism,replay-train"),
    )
    if train_result.returncode != 0:
        raise AssertionError(
            f"Replay equivalence train setup failed.\nstdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}"
        )
    checkpoint = Path(str(load_json(train_summary_path)["checkpoint_path"]))

    first = _collect_replay_eval(
        tmp_path / "replay_eval_first.json",
        checkpoint=checkpoint,
        seed_pack="bootstrap_smoke",
        max_steps=32,
    )
    second = _collect_replay_eval(
        tmp_path / "replay_eval_second.json",
        checkpoint=checkpoint,
        seed_pack="bootstrap_smoke",
        max_steps=32,
    )

    assert _canonicalize_summary(first["summary"]) == _canonicalize_summary(second["summary"])
    assert first["replay_pack"] == second["replay_pack"]
    assert first["replay_index"] == second["replay_index"]


def _collect_replay_eval(
    output: Path,
    *,
    checkpoint: Path,
    seed_pack: str,
    max_steps: int,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "scripts" / "replay_eval.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(repo_root / "configs" / "eval" / "replay_eval_v0.yaml"),
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
            f"Replay eval collection failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    summary = json.loads(output.read_text(encoding="utf-8"))
    return {
        "summary": summary,
        "replay_pack": json.loads(Path(str(summary["replay_pack_path"])).read_text(encoding="utf-8")),
        "replay_index": json.loads(Path(str(summary["replay_index_path"])).read_text(encoding="utf-8")),
    }


def _canonicalize_summary(payload: dict[str, object]) -> dict[str, object]:
    ignored = {
        "artifacts",
        "eval_summary_path",
        "replay_index_path",
        "replay_pack_path",
        "run_manifest_path",
        "wandb_run_id",
    }
    return {
        key: value
        for key, value in payload.items()
        if key not in ignored
    }


def _require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))
