from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
from uuid import uuid4

import yaml

from fight_caves_rl.defaults import (
    DEFAULT_DEMO_BACKEND,
    DEFAULT_ENV_BENCHMARK_CONFIG_PATH,
    DEFAULT_TRAIN_BENCHMARK_CONFIG_PATH,
    DEFAULT_TRAIN_CONFIG_PATH,
    DEFAULT_TRAIN_ENV_BACKEND,
    DEFAULT_VECENV_SMOKE_CONFIG_PATH,
    DEMO_BACKEND_FIGHT_CAVES_DEMO_LITE,
    DEMO_BACKEND_ORACLE_V1,
    DEMO_BACKEND_RSPS_HEADED,
    backend_selection_registry,
)
from fight_caves_rl.utils.paths import repo_root


@dataclass(frozen=True)
class CommandReport:
    label: str
    command: tuple[str, ...]
    returncode: int
    duration_seconds: float
    stdout_path: str
    stderr_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AcceptanceCheck:
    name: str
    passed: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PR13 MVP acceptance gate.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    root = repo_root()
    acceptance_root = _build_output_dir(args.output_dir)
    logs_dir = acceptance_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.report or (acceptance_root / "acceptance_report.json")
    suite_env = os.environ.copy()
    runtime_env = _build_offline_env(acceptance_root, base_env=suite_env)

    command_reports: list[CommandReport] = []
    checks: list[AcceptanceCheck] = []
    report: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "gate_id": "mvp_acceptance_v0",
        "repo_root": str(root),
        "output_dir": str(acceptance_root),
        "commands": [],
        "checks": [],
        "all_passed": False,
    }

    try:
        suite_commands = (
            ("unit", [sys.executable, "-m", "pytest", "fight_caves_rl/tests/unit", "-q"]),
            ("train", [sys.executable, "-m", "pytest", "fight_caves_rl/tests/train", "-q"]),
            (
                "integration",
                [sys.executable, "-m", "pytest", "fight_caves_rl/tests/integration", "-q"],
            ),
            (
                "determinism",
                [sys.executable, "-m", "pytest", "fight_caves_rl/tests/determinism", "-q"],
            ),
            ("parity", [sys.executable, "-m", "pytest", "fight_caves_rl/tests/parity", "-q"]),
            ("smoke", [sys.executable, "-m", "pytest", "fight_caves_rl/tests/smoke", "-q"]),
            (
                "performance",
                [sys.executable, "-m", "pytest", "fight_caves_rl/tests/performance", "-q"],
            ),
        )
        for label, command in suite_commands:
            command_reports.append(
                _run_command(
                    label=f"pytest_{label}",
                    command=command,
                    env=suite_env,
                    logs_dir=logs_dir,
                )
            )

        train_summary_path = acceptance_root / "train_summary.json"
        command_reports.append(
            _run_command(
                label="train_smoke",
                command=[
                    sys.executable,
                    str(root / "scripts" / "train.py"),
                    "--config",
                    str(root / DEFAULT_TRAIN_CONFIG_PATH),
                    "--total-timesteps",
                    "16",
                    "--data-dir",
                    str(acceptance_root / "train_data"),
                    "--output",
                    str(train_summary_path),
                ],
                env=runtime_env,
                logs_dir=logs_dir,
                timeout_seconds=300.0,
            )
        )
        train_summary = _load_json(train_summary_path)
        checkpoint_path = Path(str(train_summary["checkpoint_path"]))
        checks.append(
            AcceptanceCheck(
                name="train_artifacts",
                passed=_artifact_categories(train_summary)
                >= {"checkpoint", "checkpoint_metadata", "run_manifest"},
                details={
                    "categories": sorted(_artifact_categories(train_summary)),
                    "checkpoint_path": str(checkpoint_path),
                    "run_manifest_path": str(train_summary["run_manifest_path"]),
                },
            )
        )
        checks.append(
            AcceptanceCheck(
                name="default_backend_selection",
                passed=(
                    DEFAULT_TRAIN_ENV_BACKEND == "v2_fast"
                    and DEFAULT_DEMO_BACKEND == DEMO_BACKEND_RSPS_HEADED
                    and (root / DEFAULT_TRAIN_CONFIG_PATH).is_file()
                    and (root / DEFAULT_VECENV_SMOKE_CONFIG_PATH).is_file()
                    and (root / DEFAULT_ENV_BENCHMARK_CONFIG_PATH).is_file()
                    and (root / DEFAULT_TRAIN_BENCHMARK_CONFIG_PATH).is_file()
                    and Path(backend_selection_registry()[DEMO_BACKEND_RSPS_HEADED].entrypoint).is_file()
                    and Path(backend_selection_registry()[DEMO_BACKEND_ORACLE_V1].entrypoint).is_file()
                    and (root.parent / "fight-caves-demo-lite" / "README.md").is_file()
                    and yaml.safe_load(
                        (root / "configs" / "train" / "smoke_ppo_v0.yaml").read_text(encoding="utf-8")
                    )["env"]["env_backend"]
                    == "v1_bridge"
                    and yaml.safe_load(
                        (root / "configs" / "train" / "train_baseline_v0.yaml").read_text(encoding="utf-8")
                    )["env"]["env_backend"]
                    == "v1_bridge"
                    and yaml.safe_load(
                        (root / "configs" / "benchmark" / "vecenv_256env_v0.yaml").read_text(encoding="utf-8")
                    )["env"]["env_backend"]
                    == "v1_bridge"
                    and yaml.safe_load(
                        (root / "configs" / "benchmark" / "train_1024env_v0.yaml").read_text(encoding="utf-8")
                    )["env"]["env_backend"]
                    == "v1_bridge"
                ),
                details={
                    "train_default_config": str(root / DEFAULT_TRAIN_CONFIG_PATH),
                    "train_default_env_backend": DEFAULT_TRAIN_ENV_BACKEND,
                    "demo_default_backend": DEFAULT_DEMO_BACKEND,
                    "preserved_train_fallbacks": {
                        "smoke_ppo_v0": "v1_bridge",
                        "train_baseline_v0": "v1_bridge",
                    },
                    "preserved_benchmark_fallbacks": {
                        "vecenv_256env_v0": "v1_bridge",
                        "train_1024env_v0": "v1_bridge",
                    },
                    "fallback_backends": [
                        DEMO_BACKEND_FIGHT_CAVES_DEMO_LITE,
                        DEMO_BACKEND_ORACLE_V1,
                    ],
                },
            )
        )

        first_eval_path = acceptance_root / "replay_eval_first.json"
        second_eval_path = acceptance_root / "replay_eval_second.json"
        for label, output_path in (
            ("replay_eval_first", first_eval_path),
            ("replay_eval_second", second_eval_path),
        ):
            command_reports.append(
                _run_command(
                    label=label,
                    command=[
                        sys.executable,
                        str(root / "scripts" / "replay_eval.py"),
                        "--checkpoint",
                        str(checkpoint_path),
                        "--config",
                        str(root / "configs" / "eval" / "replay_eval_v0.yaml"),
                        "--max-steps",
                        "32",
                    "--output",
                    str(output_path),
                ],
                    env=runtime_env,
                    logs_dir=logs_dir,
                    timeout_seconds=300.0,
                )
            )
        first_eval = _load_json(first_eval_path)
        second_eval = _load_json(second_eval_path)
        first_replay_pack = _load_json(Path(str(first_eval["replay_pack_path"])))
        second_replay_pack = _load_json(Path(str(second_eval["replay_pack_path"])))
        first_replay_index = _load_json(Path(str(first_eval["replay_index_path"])))
        second_replay_index = _load_json(Path(str(second_eval["replay_index_path"])))
        checks.append(
            AcceptanceCheck(
                name="deterministic_replay_eval",
                passed=(
                    _canonicalize_eval_summary(first_eval)
                    == _canonicalize_eval_summary(second_eval)
                    and first_replay_pack == second_replay_pack
                    and first_replay_index == second_replay_index
                ),
                details={
                    "first_summary_path": str(first_eval_path),
                    "second_summary_path": str(second_eval_path),
                    "summary_digest": first_eval["summary_digest"],
                    "seed_pack": first_eval["seed_pack"],
                },
            )
        )
        checks.append(
            AcceptanceCheck(
                name="eval_artifacts",
                passed=_artifact_categories(first_eval)
                >= {"eval_summary", "replay_pack", "replay_index", "run_manifest"},
                details={
                    "categories": sorted(_artifact_categories(first_eval)),
                    "eval_summary_path": str(first_eval["eval_summary_path"]),
                    "replay_pack_path": str(first_eval["replay_pack_path"]),
                    "replay_index_path": str(first_eval["replay_index_path"]),
                    "run_manifest_path": str(first_eval["run_manifest_path"]),
                },
            )
        )

        parity_report_path = acceptance_root / "parity_report.json"
        command_reports.append(
            _run_command(
                label="parity_canary",
                command=[
                    sys.executable,
                    str(root / "scripts" / "run_parity_canary.py"),
                    "--config",
                    str(root / "configs" / "eval" / "parity_canary_v0.yaml"),
                    "--output",
                    str(parity_report_path),
                ],
                env=runtime_env,
                logs_dir=logs_dir,
                timeout_seconds=300.0,
            )
        )
        parity_report = _load_json(parity_report_path)
        checks.append(
            AcceptanceCheck(
                name="parity_canaries",
                passed=bool(parity_report["all_passed"]),
                details={
                    "scenario_count": len(parity_report["scenarios"]),
                    "mechanics_scenario_count": sum(
                        1
                        for scenario in parity_report["scenarios"]
                        if "oracle_matches_v2_fast" in scenario
                    ),
                    "mechanics_failures": [
                        scenario["scenario_id"]
                        for scenario in parity_report["scenarios"]
                        if scenario.get("oracle_matches_v2_fast") is False
                    ],
                    "parity_report_path": str(parity_report_path),
                },
            )
        )

        bridge_bench_path = acceptance_root / "bridge_benchmark.json"
        env_bench_path = acceptance_root / "env_benchmark.json"
        train_bench_path = acceptance_root / "train_benchmark.json"
        benchmark_commands = (
            (
                "benchmark_bridge",
                [
                    sys.executable,
                    str(root / "scripts" / "benchmark_bridge.py"),
                    "--config",
                    str(root / "configs" / "benchmark" / "bridge_1env_v0.yaml"),
                    "--rounds",
                    "16",
                    "--output",
                    str(bridge_bench_path),
                ],
            ),
            (
                "benchmark_env",
                [
                    sys.executable,
                    str(root / "scripts" / "benchmark_env.py"),
                    "--config",
                    str(root / DEFAULT_ENV_BENCHMARK_CONFIG_PATH),
                    "--env-count",
                    "8",
                    "--rounds",
                    "16",
                    "--output",
                    str(env_bench_path),
                ],
            ),
            (
                "benchmark_train",
                [
                    sys.executable,
                    str(root / "scripts" / "benchmark_train.py"),
                    "--config",
                    str(root / DEFAULT_TRAIN_BENCHMARK_CONFIG_PATH),
                    "--env-count",
                    "2",
                    "--total-timesteps",
                    "8",
                    "--logging-modes",
                    "disabled,standard",
                    "--output",
                    str(train_bench_path),
                ],
            ),
        )
        for label, command in benchmark_commands:
            command_reports.append(
                _run_command(
                    label=label,
                    command=command,
                    env=runtime_env,
                    logs_dir=logs_dir,
                    timeout_seconds=300.0,
                )
            )

        bridge_bench = _load_json(bridge_bench_path)
        env_bench = _load_json(env_bench_path)
        train_bench = _load_json(train_bench_path)
        checks.append(
            AcceptanceCheck(
                name="benchmark_outputs",
                passed=(
                    float(bridge_bench["batch"]["env_steps_per_second"]) > 0.0
                    and float(env_bench["measurement"]["env_steps_per_second"]) > 0.0
                    and all(
                        float(entry["env_steps_per_second"]) > 0.0
                        for entry in train_bench["measurements"]
                    )
                ),
                details={
                    "bridge_output_path": str(bridge_bench_path),
                    "env_output_path": str(env_bench_path),
                    "train_output_path": str(train_bench_path),
                },
            )
        )
        checks.append(
            AcceptanceCheck(
                name="headed_phase8_proof_available",
                passed=all(
                    path.is_file()
                    for path in (
                        root.parent / "RSPS" / "docs" / "fight_caves_demo_backend_control_validation.md",
                        root.parent / "RSPS" / "docs" / "fight_caves_demo_headed_replay_validation.md",
                        root.parent / "RSPS" / "docs" / "fight_caves_demo_live_checkpoint_validation.md",
                    )
                ),
                details={
                    "backend_control_validation": str(
                        root.parent / "RSPS" / "docs" / "fight_caves_demo_backend_control_validation.md"
                    ),
                    "headed_replay_validation": str(
                        root.parent / "RSPS" / "docs" / "fight_caves_demo_headed_replay_validation.md"
                    ),
                    "live_checkpoint_validation": str(
                        root.parent / "RSPS" / "docs" / "fight_caves_demo_live_checkpoint_validation.md"
                    ),
                },
            )
        )

        report["commands"] = [entry.to_dict() for entry in command_reports]
        report["checks"] = [entry.to_dict() for entry in checks]
        report["train_summary_path"] = str(train_summary_path)
        report["replay_eval_first_path"] = str(first_eval_path)
        report["replay_eval_second_path"] = str(second_eval_path)
        report["parity_report_path"] = str(parity_report_path)
        report["bridge_benchmark_path"] = str(bridge_bench_path)
        report["env_benchmark_path"] = str(env_bench_path)
        report["train_benchmark_path"] = str(train_bench_path)
        report["all_passed"] = all(entry.passed for entry in checks)
        _write_report(report_path, report)
        if not report["all_passed"]:
            raise SystemExit(1)
    except Exception as exc:
        report["commands"] = [entry.to_dict() for entry in command_reports]
        report["checks"] = [entry.to_dict() for entry in checks]
        report["all_passed"] = False
        report["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_report(report_path, report)
        raise


def _build_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.expanduser().resolve()
    run_id = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}"
    return (repo_root() / "artifacts" / "acceptance" / "mvp_acceptance_v0" / run_id).resolve()


def _build_offline_env(output_dir: Path, *, base_env: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ if base_env is None else base_env)
    env.update(
        {
            "WANDB_PROJECT": "fight-caves-rl-acceptance",
            "WANDB_MODE": "offline",
            "WANDB_GROUP": "acceptance",
            "WANDB_TAGS": "acceptance,mvp",
            "WANDB_DIR": str(output_dir / "wandb"),
            "WANDB_DATA_DIR": str(output_dir / "wandb-data"),
            "WANDB_CACHE_DIR": str(output_dir / "wandb-cache"),
        }
    )
    return env


def _run_command(
    *,
    label: str,
    command: list[str],
    env: dict[str, str],
    logs_dir: Path,
    timeout_seconds: float = 600.0,
) -> CommandReport:
    stdout_path = logs_dir / f"{label}.stdout.log"
    stderr_path = logs_dir / f"{label}.stderr.log"
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=str(repo_root()),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_seconds,
    )
    duration_seconds = time.perf_counter() - started
    stdout_path.write_text(result.stdout, encoding="utf-8")
    stderr_path.write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"{label} failed with exit code {result.returncode}. "
            f"See {stdout_path} and {stderr_path}."
        )
    return CommandReport(
        label=label,
        command=tuple(str(part) for part in command),
        returncode=int(result.returncode),
        duration_seconds=float(duration_seconds),
        stdout_path=str(stdout_path),
        stderr_path=str(stderr_path),
    )


def _artifact_categories(payload: dict[str, Any]) -> set[str]:
    return {
        str(record["category"])
        for record in payload.get("artifacts", [])
    }


def _canonicalize_eval_summary(payload: dict[str, Any]) -> dict[str, Any]:
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
