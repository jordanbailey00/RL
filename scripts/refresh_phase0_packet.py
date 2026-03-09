from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from fight_caves_rl.benchmarks.phase0_packet import (
    PHASE0_BRIDGE_ENV_COUNTS,
    PHASE0_TRAIN_ENV_COUNTS,
    PHASE0_VECENV_ENV_COUNTS,
    build_phase0_packet_report,
)
from fight_caves_rl.utils.config import load_bootstrap_config
from fight_caves_rl.utils.paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the Phase 0 benchmark packet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-sim", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    bootstrap = load_bootstrap_config()
    sim_repo = bootstrap.sim_repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_report_path = sim_repo / "docs" / "performance_benchmark.json"
    if not args.skip_sim:
        _run_command(
            ["./gradlew", "--no-daemon", ":game:headlessPerformanceReport"],
            cwd=sim_repo,
        )
    if not sim_report_path.is_file():
        raise RuntimeError(f"Expected sim report json at {sim_report_path}.")

    bridge_reports: dict[int, Path] = {}
    for env_count in PHASE0_BRIDGE_ENV_COUNTS:
        output_path = output_dir / f"bridge_{env_count}env.json"
        bridge_reports[env_count] = output_path
        config_name = "bridge_1env_v0.yaml" if env_count == 1 else "bridge_64env_v0.yaml"
        _run_command(
            [
                sys.executable,
                str(root / "scripts" / "benchmark_bridge.py"),
                "--config",
                str(root / "configs" / "benchmark" / config_name),
                "--env-count",
                str(env_count),
                "--output",
                str(output_path),
            ],
            cwd=root,
        )

    vecenv_reports: dict[int, Path] = {}
    for env_count in PHASE0_VECENV_ENV_COUNTS:
        output_path = output_dir / f"vecenv_{env_count}env.json"
        vecenv_reports[env_count] = output_path
        _run_command(
            [
                sys.executable,
                str(root / "scripts" / "benchmark_env.py"),
                "--config",
                str(root / "configs" / "benchmark" / "vecenv_256env_v0.yaml"),
                "--env-count",
                str(env_count),
                "--output",
                str(output_path),
            ],
            cwd=root,
        )

    train_reports: dict[int, Path] = {}
    if not args.skip_train:
        for env_count in PHASE0_TRAIN_ENV_COUNTS:
            output_path = output_dir / f"train_{env_count}env.json"
            train_reports[env_count] = output_path
            _run_command(
                [
                    sys.executable,
                    str(root / "scripts" / "benchmark_train.py"),
                    "--config",
                    str(root / "configs" / "benchmark" / "train_1024env_v0.yaml"),
                    "--env-count",
                    str(env_count),
                    "--total-timesteps",
                    "1024",
                    "--logging-modes",
                    "disabled",
                    "--output",
                    str(output_path),
                ],
                cwd=root,
            )

    packet = build_phase0_packet_report(
        output_dir=output_dir,
        sim_report_path=sim_report_path,
        bridge_reports=bridge_reports,
        vecenv_reports=vecenv_reports,
        train_reports=train_reports,
    )
    (output_dir / "phase0_packet.json").write_text(
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
            "Phase 0 packet command failed.\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


if __name__ == "__main__":
    main()
