from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_train_ceiling_benchmark_smoke(tmp_path: Path):
    output = tmp_path / "train_ceiling.json"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/benchmark_train_ceiling.py",
            "--config",
            "configs/benchmark/train_1024env_v0.yaml",
            "--env-counts",
            "1",
            "--total-timesteps",
            "32",
            "--output",
            str(output),
        ],
        cwd="/home/jordan/code/RL",
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert str(payload["metric_contract_id"]) == "train_ceiling_diagnostic_v1"
    measurements = payload["measurements"]
    assert len(measurements) == 1
    assert int(measurements[0]["env_count"]) == 1
    assert str(measurements[0]["metric_scope"]) == "diagnostic_shipped_sync_path_v1"
    assert float(measurements[0]["diagnostic_env_steps_per_second"]) > 0.0
    assert float(measurements[0]["runner_stage_seconds"]["evaluate_seconds"]) >= 0.0
    assert "train_advantage" in measurements[0]["trainer_bucket_totals"]
