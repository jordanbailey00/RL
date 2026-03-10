from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from fight_caves_rl.benchmarks.phase2_prototype_packet import (
    PHASE2_PROTOTYPE_ENV_COUNTS,
    build_phase2_prototype_packet_report,
)
from fight_caves_rl.utils.paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh the Phase 2 prototype benchmark decision packet."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = repo_root()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    production_reports: dict[int, Path] = {}
    for env_count in PHASE2_PROTOTYPE_ENV_COUNTS:
        output_path = output_dir / f"prototype_sync_v1_{env_count}env_disabled.json"
        production_reports[env_count] = output_path
        _run_command(
            [
                sys.executable,
                str(root / "scripts" / "benchmark_train.py"),
                "--config",
                str(root / "configs" / "benchmark" / "train_1024env_v0.yaml"),
                "--runner-mode",
                "prototype_sync_v1",
                "--env-count",
                str(env_count),
                "--total-timesteps",
                "4096",
                "--logging-modes",
                "disabled",
                "--output",
                str(output_path),
            ],
            cwd=root,
        )

    learner_ceiling_report = output_dir / "train_ceiling_16_64.json"
    _run_command(
        [
            sys.executable,
            str(root / "scripts" / "benchmark_train_ceiling.py"),
            "--config",
            str(root / "configs" / "benchmark" / "train_1024env_v0.yaml"),
            "--env-counts",
            "16,64",
            "--total-timesteps",
            "4096",
            "--output",
            str(learner_ceiling_report),
        ],
        cwd=root,
    )

    packet = build_phase2_prototype_packet_report(
        output_dir=output_dir,
        production_reports=production_reports,
        learner_ceiling_report=learner_ceiling_report,
    )
    (output_dir / "phase2_prototype_packet.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_command(command: list[str], *, cwd: Path) -> None:
    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Phase 2 prototype packet command failed.\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


if __name__ == "__main__":
    main()
