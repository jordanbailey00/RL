from __future__ import annotations

import argparse
import cProfile
import json
from pathlib import Path
import sys

import numpy as np

from fight_caves_rl.benchmarks.phase1_packet import (
    PHASE1_BRIDGE_ENV_COUNTS,
    PHASE1_VECENV_ENV_COUNTS,
    build_phase1_packet_report,
    write_profile_top_table,
)
from fight_caves_rl.puffer.factory import load_smoke_train_config, make_vecenv
from fight_caves_rl.utils.paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the Phase 1 benchmark decision packet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase0-baseline-dir", type=Path, default=None)
    args = parser.parse_args()

    root = repo_root()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bridge_reports: dict[int, Path] = {}
    for env_count in PHASE1_BRIDGE_ENV_COUNTS:
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
    for env_count in PHASE1_VECENV_ENV_COUNTS:
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

    python_profile_path = output_dir / "python_vec16_steady.prof"
    _run_python_profile(python_profile_path)
    python_profile_top_path = write_profile_top_table(
        python_profile_path,
        output_dir / "python_vec16_top.txt",
    )

    packet = build_phase1_packet_report(
        output_dir=output_dir,
        phase0_baseline_dir=None if args.phase0_baseline_dir is None else args.phase0_baseline_dir.resolve(),
        bridge_reports=bridge_reports,
        vecenv_reports=vecenv_reports,
        python_profile_path=python_profile_path,
        python_profile_top_path=python_profile_top_path,
    )
    (output_dir / "phase1_packet.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _run_python_profile(output_path: Path) -> None:
    config = load_smoke_train_config(Path("configs/benchmark/vecenv_256env_v0.yaml"))
    config["num_envs"] = 16
    vecenv = make_vecenv(config, backend="embedded")
    seed = int(config["train"]["seed"])
    actions = np.zeros((16, len(vecenv.single_action_space.nvec)), dtype=np.int32)
    try:
        vecenv.async_reset(seed)
        vecenv.recv()
        for _ in range(8):
            vecenv.send(actions)
            vecenv.recv()
        profiler = cProfile.Profile()
        profiler.enable()
        for _ in range(64):
            vecenv.send(actions)
            vecenv.recv()
        profiler.disable()
        profiler.dump_stats(str(output_path))
    finally:
        vecenv.close()


def _run_command(command: list[str], *, cwd: Path) -> None:
    import subprocess

    result = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Phase 1 packet command failed.\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


if __name__ == "__main__":
    main()
