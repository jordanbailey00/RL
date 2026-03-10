from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import yaml

from fight_caves_rl.benchmarks.phase2_packet import (
    PHASE2_TRAIN_ENV_COUNTS,
    PHASE2_TRANSPORT_ENV_COUNTS,
    build_phase2_packet_report,
)
from fight_caves_rl.envs.shared_memory_transport import (
    PIPE_PICKLE_TRANSPORT_MODE,
    SHARED_MEMORY_TRANSPORT_MODE,
)
from fight_caves_rl.puffer.factory import load_smoke_train_config
from fight_caves_rl.utils.paths import repo_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the Phase 2 transport benchmark decision packet.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    root = repo_root()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    transport_reports: dict[int, Path] = {}
    for env_count in PHASE2_TRANSPORT_ENV_COUNTS:
        output_path = output_dir / f"transport_{env_count}env.json"
        transport_reports[env_count] = output_path
        _run_command(
            [
                sys.executable,
                str(root / "scripts" / "benchmark_subprocess_transport.py"),
                "--config",
                str(root / "configs" / "benchmark" / "vecenv_256env_v0.yaml"),
                "--env-count",
                str(env_count),
                "--rounds",
                "64",
                "--output",
                str(output_path),
            ],
            cwd=root,
        )

    train_reports: dict[tuple[str, int], Path] = {}
    for transport_mode in (PIPE_PICKLE_TRANSPORT_MODE, SHARED_MEMORY_TRANSPORT_MODE):
        for env_count in PHASE2_TRAIN_ENV_COUNTS:
            output_path = output_dir / f"train_{transport_mode}_{env_count}env.json"
            train_reports[(transport_mode, env_count)] = output_path
            config_path = _write_temp_train_config(
                config_id=f"phase2_{transport_mode}_{env_count}env_v0",
                env_count=env_count,
                transport_mode=transport_mode,
            )
            try:
                _run_command(
                    [
                        sys.executable,
                        str(root / "scripts" / "benchmark_train.py"),
                        "--config",
                        str(config_path),
                        "--env-count",
                        str(env_count),
                        "--total-timesteps",
                        "512",
                        "--logging-modes",
                        "disabled",
                        "--output",
                        str(output_path),
                    ],
                    cwd=root,
                )
            finally:
                config_path.unlink(missing_ok=True)

    packet = build_phase2_packet_report(
        output_dir=output_dir,
        transport_reports=transport_reports,
        train_reports=train_reports,
    )
    (output_dir / "phase2_packet.json").write_text(
        json.dumps(packet.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_temp_train_config(
    *,
    config_id: str,
    env_count: int,
    transport_mode: str,
) -> Path:
    root = repo_root()
    config = load_smoke_train_config(root / "configs" / "benchmark" / "train_1024env_v0.yaml")
    config["config_id"] = str(config_id)
    config["num_envs"] = int(env_count)
    config.setdefault("env", {})["subprocess_transport_mode"] = str(transport_mode)
    config["env"]["account_name_prefix"] = f"rl_phase2_{transport_mode}_{env_count}"
    config["train"]["checkpoint_interval"] = 1_000_000
    config.setdefault("logging", {})["dashboard"] = False

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
        return Path(handle.name)


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
            "Phase 2 packet command failed.\n"
            f"command: {' '.join(command)}\n"
            f"cwd: {cwd}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )


if __name__ == "__main__":
    main()
