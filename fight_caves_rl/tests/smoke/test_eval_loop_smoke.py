from __future__ import annotations

from pathlib import Path

from fight_caves_rl.tests.smoke._helpers import load_json, require_live_runtime, run_script


def test_eval_loop_smoke(tmp_path: Path):
    require_live_runtime()

    summary_path = tmp_path / "train_summary.json"
    eval_path = tmp_path / "eval_summary.json"
    train_result = run_script(
        "train.py",
        "--config",
        "configs/train/smoke_ppo_v0.yaml",
        "--total-timesteps",
        "4",
        "--data-dir",
        str(tmp_path / "train_artifacts"),
        "--output",
        str(summary_path),
    )
    if train_result.returncode != 0:
        raise AssertionError(
            f"Train smoke failed.\nstdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}"
        )

    checkpoint_path = str(load_json(summary_path)["checkpoint_path"])
    eval_result = run_script(
        "eval.py",
        "--checkpoint",
        checkpoint_path,
        "--config",
        "configs/eval/replay_eval_v0.yaml",
        "--max-steps",
        "16",
        "--output",
        str(eval_path),
    )
    if eval_result.returncode != 0:
        raise AssertionError(
            f"Eval smoke failed.\nstdout:\n{eval_result.stdout}\nstderr:\n{eval_result.stderr}"
        )

    summary = load_json(eval_path)
    assert summary["config_id"] == "replay_eval_v0"
    assert summary["seed_pack"] == "bootstrap_smoke"
    assert len(summary["per_seed"]) == 2
    assert summary["summary_digest"]
