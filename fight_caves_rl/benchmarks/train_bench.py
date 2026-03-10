from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import tempfile
from time import perf_counter, perf_counter_ns
from typing import Any, Sequence

import yaml

from fight_caves_rl.benchmarks.common import BenchmarkContext, build_benchmark_context
from fight_caves_rl.policies.mlp import MultiDiscreteMLPPolicy
from fight_caves_rl.puffer.factory import (
    build_puffer_train_config,
    load_smoke_train_config,
    make_vecenv,
)
from fight_caves_rl.puffer.production_trainer import PrototypeProductionTrainer
from fight_caves_rl.puffer.trainer import ConfigurablePuffeRL
from fight_caves_rl.utils.paths import repo_root

SMOKE_SUBPROCESS_RUNNER_MODE = "smoke_subprocess_v1"
CORE_INPROCESS_RUNNER_MODE = "core_inprocess_v1"
PROTOTYPE_SYNC_RUNNER_MODE = "prototype_sync_v1"


@dataclass(frozen=True)
class TrainBenchmarkMeasurement:
    metric_scope: str
    runner_mode: str
    logging_mode: str
    wandb_mode: str
    env_count: int
    total_timesteps: int
    global_step: int
    elapsed_nanos: int
    wall_clock_elapsed_nanos: int
    env_steps_per_second: float
    production_env_steps_per_second: float
    wall_clock_env_steps_per_second: float
    evaluate_seconds: float
    train_seconds: float
    final_evaluate_seconds: float
    runner_stage_seconds: dict[str, float]
    trainer_bucket_totals: dict[str, dict[str, float | int]]
    log_records: int
    artifact_count: int
    aggressive_log_bursts: int
    checkpoint_path: str
    run_manifest_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainBenchmarkReport:
    created_at: str
    metric_contract_id: str
    config_id: str
    env_count: int
    bridge_protocol_id: str
    bridge_protocol_version: int
    measurements: tuple[TrainBenchmarkMeasurement, ...]
    sps_ratio_vs_disabled: dict[str, float]
    context: BenchmarkContext

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["measurements"] = [measurement.to_dict() for measurement in self.measurements]
        payload["context"] = self.context.to_dict()
        return payload


class _NullLogger:
    def __init__(self) -> None:
        self.run_id = "null"
        self.records: list[object] = []
        self.artifact_records: list[object] = []
        self.effective_tags: tuple[str, ...] = ()

    def log(self, *args: Any, **kwargs: Any) -> None:
        return None

    def close(self, *args: Any, **kwargs: Any) -> None:
        return None

    def build_artifact_record(self, *args: Any, **kwargs: Any) -> object:
        raise RuntimeError("Core benchmark runner does not build artifacts.")

    def update_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_artifact(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


def run_train_benchmark(
    config_path: str | Path,
    *,
    total_timesteps_override: int | None = None,
    env_count_override: int | None = None,
    logging_modes_override: Sequence[str] | None = None,
    runner_mode: str = SMOKE_SUBPROCESS_RUNNER_MODE,
) -> TrainBenchmarkReport:
    config = load_smoke_train_config(config_path)
    if env_count_override is not None:
        config["num_envs"] = int(env_count_override)

    config.setdefault("logging", {})["dashboard"] = False
    benchmark_config = dict(config.get("benchmark", {}))
    total_timesteps = int(
        total_timesteps_override
        if total_timesteps_override is not None
        else benchmark_config.get("total_timesteps", config["train"]["total_timesteps"])
    )
    logging_modes = tuple(
        logging_modes_override
        if logging_modes_override is not None
        else benchmark_config.get("logging_modes", ("disabled", "standard", "aggressive"))
    )
    aggressive_log_bursts = int(benchmark_config.get("aggressive_log_bursts", 4))
    timeout_seconds = int(benchmark_config.get("timeout_seconds", 180))
    context = build_benchmark_context(
        env_count=int(config["num_envs"]),
        logging_mode="recorded_per_measurement",
        replay_mode="disabled",
        dashboard_mode="disabled",
        reward_config_id=str(config["reward_config"]),
        curriculum_config_id=str(config["curriculum_config"]),
    )
    measurements = tuple(
        _run_train_measurement(
            base_config=config,
            total_timesteps=total_timesteps,
            logging_mode=str(mode),
            aggressive_log_bursts=aggressive_log_bursts,
            timeout_seconds=timeout_seconds,
            runner_mode=runner_mode,
        )
        for mode in logging_modes
    )
    disabled = next(
        (measurement for measurement in measurements if measurement.logging_mode == "disabled"),
        None,
    )
    ratios: dict[str, float] = {}
    for measurement in measurements:
        if disabled is None or disabled.production_env_steps_per_second <= 0.0:
            ratios[measurement.logging_mode] = 0.0
            continue
        ratios[measurement.logging_mode] = (
            measurement.production_env_steps_per_second / disabled.production_env_steps_per_second
        )
    return TrainBenchmarkReport(
        created_at=datetime.now(UTC).isoformat(),
        metric_contract_id="train_benchmark_production_v1",
        config_id=str(config["config_id"]),
        env_count=int(config["num_envs"]),
        bridge_protocol_id=context.bridge_protocol_id,
        bridge_protocol_version=context.bridge_protocol_version,
        measurements=measurements,
        sps_ratio_vs_disabled=ratios,
        context=context,
    )


def _run_train_measurement(
    *,
    base_config: dict[str, Any],
    total_timesteps: int,
    logging_mode: str,
    aggressive_log_bursts: int,
    timeout_seconds: int,
    runner_mode: str,
) -> TrainBenchmarkMeasurement:
    if runner_mode == PROTOTYPE_SYNC_RUNNER_MODE:
        return _run_train_measurement_prototype(
            base_config=base_config,
            total_timesteps=total_timesteps,
            logging_mode=logging_mode,
            aggressive_log_bursts=aggressive_log_bursts,
        )
    if runner_mode == CORE_INPROCESS_RUNNER_MODE:
        return _run_train_measurement_core(
            base_config=base_config,
            total_timesteps=total_timesteps,
            logging_mode=logging_mode,
            aggressive_log_bursts=aggressive_log_bursts,
        )
    if runner_mode != SMOKE_SUBPROCESS_RUNNER_MODE:
        raise ValueError(f"Unsupported train benchmark runner mode: {runner_mode!r}")
    return _run_train_measurement_smoke(
        base_config=base_config,
        total_timesteps=total_timesteps,
        logging_mode=logging_mode,
        aggressive_log_bursts=aggressive_log_bursts,
        timeout_seconds=timeout_seconds,
    )


def _run_train_measurement_smoke(
    *,
    base_config: dict[str, Any],
    total_timesteps: int,
    logging_mode: str,
    aggressive_log_bursts: int,
    timeout_seconds: int,
) -> TrainBenchmarkMeasurement:
    config = json.loads(json.dumps(base_config))
    config["train"]["total_timesteps"] = int(total_timesteps)
    effective_batch = max(1, min(int(total_timesteps), int(config["train"]["batch_size"])))
    config["train"]["batch_size"] = effective_batch
    config["train"]["minibatch_size"] = max(
        1,
        min(int(config["train"]["minibatch_size"]), effective_batch),
    )
    config["train"]["max_minibatch_size"] = max(
        int(config["train"]["minibatch_size"]),
        min(int(config["train"]["max_minibatch_size"]), effective_batch),
    )
    config["train"]["bptt_horizon"] = max(
        1,
        min(int(config["train"]["bptt_horizon"]), effective_batch),
    )
    root = repo_root()
    benchmark_root = root / "artifacts" / "benchmarks" / "train" / str(config["config_id"])
    mode_output_dir = benchmark_root / logging_mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = mode_output_dir / "train_summary.json"

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)
        config_path = Path(handle.name)

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled" if logging_mode == "disabled" else env.get("WANDB_MODE", "offline")
    if logging_mode == "aggressive":
        env["FC_RL_BENCHMARK_EXTRA_LOG_BURSTS"] = str(int(aggressive_log_bursts))
    else:
        env.pop("FC_RL_BENCHMARK_EXTRA_LOG_BURSTS", None)

    try:
        started = perf_counter_ns()
        try:
            import subprocess
            import sys

            result = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts" / "train.py"),
                    "--config",
                    str(config_path),
                    "--total-timesteps",
                    str(total_timesteps),
                    "--data-dir",
                    str(mode_output_dir / "run_data"),
                    "--output",
                    str(output_path),
                ],
                cwd=str(root),
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=float(timeout_seconds),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Train benchmark subprocess timed out after {int(timeout_seconds)}s."
            ) from exc
        elapsed_nanos = perf_counter_ns() - started
        if result.returncode != 0:
            raise RuntimeError(
                "Train benchmark subprocess failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    finally:
        config_path.unlink(missing_ok=True)

    global_step = int(payload["global_step"])
    production_elapsed_nanos = int(elapsed_nanos)
    production_env_steps_per_second = (
        0.0
        if production_elapsed_nanos <= 0
        else global_step * 1_000_000_000.0 / float(production_elapsed_nanos)
    )
    return TrainBenchmarkMeasurement(
        metric_scope="production_fast_path_v1",
        runner_mode=SMOKE_SUBPROCESS_RUNNER_MODE,
        logging_mode=str(logging_mode),
        wandb_mode=str(env["WANDB_MODE"]),
        env_count=int(config["num_envs"]),
        total_timesteps=int(total_timesteps),
        global_step=global_step,
        elapsed_nanos=production_elapsed_nanos,
        wall_clock_elapsed_nanos=int(elapsed_nanos),
        env_steps_per_second=float(production_env_steps_per_second),
        production_env_steps_per_second=float(production_env_steps_per_second),
        wall_clock_env_steps_per_second=float(production_env_steps_per_second),
        evaluate_seconds=0.0,
        train_seconds=0.0,
        final_evaluate_seconds=0.0,
        runner_stage_seconds={},
        trainer_bucket_totals={},
        log_records=int(payload["log_records"]),
        artifact_count=len(list(payload["artifacts"])),
        aggressive_log_bursts=int(aggressive_log_bursts if logging_mode == "aggressive" else 0),
        checkpoint_path=str(payload["checkpoint_path"]),
        run_manifest_path=str(payload["run_manifest_path"]),
    )


def _run_train_measurement_prototype(
    *,
    base_config: dict[str, Any],
    total_timesteps: int,
    logging_mode: str,
    aggressive_log_bursts: int,
) -> TrainBenchmarkMeasurement:
    if logging_mode != "disabled":
        raise ValueError("prototype_sync_v1 currently supports only disabled logging mode.")

    config = json.loads(json.dumps(base_config))
    config["train"]["total_timesteps"] = int(total_timesteps)
    effective_batch = max(1, min(int(total_timesteps), int(config["train"]["batch_size"])))
    config["train"]["batch_size"] = effective_batch
    config["train"]["minibatch_size"] = max(
        1,
        min(int(config["train"]["minibatch_size"]), effective_batch),
    )
    config["train"]["max_minibatch_size"] = max(
        int(config["train"]["minibatch_size"]),
        min(int(config["train"]["max_minibatch_size"]), effective_batch),
    )
    config["train"]["bptt_horizon"] = max(
        1,
        min(int(config["train"]["bptt_horizon"]), effective_batch),
    )
    config["train"]["checkpoint_interval"] = 1_000_000
    config.setdefault("logging", {})["dashboard"] = False
    root = repo_root()
    benchmark_root = root / "artifacts" / "benchmarks" / "train" / str(config["config_id"])
    data_dir = benchmark_root / "disabled" / "run_data_prototype"
    data_dir.mkdir(parents=True, exist_ok=True)

    stage_started = perf_counter()
    vecenv = make_vecenv(config, backend="subprocess")
    vecenv_build_seconds = perf_counter() - stage_started

    stage_started = perf_counter()
    policy = MultiDiscreteMLPPolicy.from_spaces(
        vecenv.single_observation_space,
        vecenv.single_action_space,
        hidden_size=int(config["policy"]["hidden_size"]),
    )
    policy_build_seconds = perf_counter() - stage_started
    train_config = build_puffer_train_config(
        config,
        data_dir=data_dir,
        total_timesteps=int(total_timesteps),
    )
    stage_started = perf_counter()
    trainer = PrototypeProductionTrainer(
        train_config,
        vecenv,
        policy,
    )
    trainer_init_seconds = perf_counter() - stage_started

    evaluate_seconds = 0.0
    train_seconds = 0.0
    final_evaluate_seconds = 0.0
    close_seconds = 0.0
    started = perf_counter_ns()
    try:
        while trainer.global_step < int(train_config["total_timesteps"]):
            step_started = perf_counter()
            trainer.collect_rollout()
            evaluate_seconds += perf_counter() - step_started

            step_started = perf_counter()
            trainer.train_update()
            train_seconds += perf_counter() - step_started

        global_step = int(trainer.global_step)
    finally:
        stage_started = perf_counter()
        trainer.close()
        close_seconds += perf_counter() - stage_started

    elapsed_nanos = perf_counter_ns() - started
    production_env_steps_per_second = (
        0.0
        if elapsed_nanos <= 0
        else global_step * 1_000_000_000.0 / float(elapsed_nanos)
    )
    return TrainBenchmarkMeasurement(
        metric_scope="production_fast_path_v1",
        runner_mode=PROTOTYPE_SYNC_RUNNER_MODE,
        logging_mode=str(logging_mode),
        wandb_mode="disabled",
        env_count=int(config["num_envs"]),
        total_timesteps=int(total_timesteps),
        global_step=global_step,
        elapsed_nanos=int(elapsed_nanos),
        wall_clock_elapsed_nanos=int(elapsed_nanos),
        env_steps_per_second=float(production_env_steps_per_second),
        production_env_steps_per_second=float(production_env_steps_per_second),
        wall_clock_env_steps_per_second=float(production_env_steps_per_second),
        evaluate_seconds=float(evaluate_seconds),
        train_seconds=float(train_seconds),
        final_evaluate_seconds=float(final_evaluate_seconds),
        runner_stage_seconds={
            "vecenv_build_seconds": float(vecenv_build_seconds),
            "policy_build_seconds": float(policy_build_seconds),
            "trainer_init_seconds": float(trainer_init_seconds),
            "evaluate_seconds": float(evaluate_seconds),
            "train_seconds": float(train_seconds),
            "final_evaluate_seconds": float(final_evaluate_seconds),
            "close_seconds": float(close_seconds),
        },
        trainer_bucket_totals=trainer.instrumentation_snapshot(),
        log_records=0,
        artifact_count=0,
        aggressive_log_bursts=int(aggressive_log_bursts if logging_mode == "aggressive" else 0),
        checkpoint_path="",
        run_manifest_path="",
    )


def _run_train_measurement_core(
    *,
    base_config: dict[str, Any],
    total_timesteps: int,
    logging_mode: str,
    aggressive_log_bursts: int,
) -> TrainBenchmarkMeasurement:
    if logging_mode != "disabled":
        raise ValueError(
            "core_inprocess_v1 currently supports only disabled logging mode."
        )

    config = json.loads(json.dumps(base_config))
    config["train"]["total_timesteps"] = int(total_timesteps)
    effective_batch = max(1, min(int(total_timesteps), int(config["train"]["batch_size"])))
    config["train"]["batch_size"] = effective_batch
    config["train"]["minibatch_size"] = max(
        1,
        min(int(config["train"]["minibatch_size"]), effective_batch),
    )
    config["train"]["max_minibatch_size"] = max(
        int(config["train"]["minibatch_size"]),
        min(int(config["train"]["max_minibatch_size"]), effective_batch),
    )
    config["train"]["bptt_horizon"] = max(
        1,
        min(int(config["train"]["bptt_horizon"]), effective_batch),
    )
    config["train"]["checkpoint_interval"] = 1_000_000
    config.setdefault("logging", {})["dashboard"] = False
    root = repo_root()
    benchmark_root = root / "artifacts" / "benchmarks" / "train" / str(config["config_id"])
    data_dir = benchmark_root / "disabled" / "run_data_core"
    data_dir.mkdir(parents=True, exist_ok=True)

    stage_started = perf_counter()
    vecenv = make_vecenv(config, backend="subprocess")
    vecenv_build_seconds = perf_counter() - stage_started

    stage_started = perf_counter()
    policy = MultiDiscreteMLPPolicy.from_spaces(
        vecenv.single_observation_space,
        vecenv.single_action_space,
        hidden_size=int(config["policy"]["hidden_size"]),
    )
    policy_build_seconds = perf_counter() - stage_started
    train_config = build_puffer_train_config(
        config,
        data_dir=data_dir,
        total_timesteps=int(total_timesteps),
    )
    stage_started = perf_counter()
    trainer = ConfigurablePuffeRL(
        train_config,
        vecenv,
        policy,
        _NullLogger(),
        dashboard_enabled=False,
        checkpointing_enabled=False,
        profiling_enabled=False,
        utilization_enabled=False,
        logging_enabled=False,
        instrumentation_enabled=True,
    )
    trainer_init_seconds = perf_counter() - stage_started

    evaluate_seconds = 0.0
    train_seconds = 0.0
    final_evaluate_seconds = 0.0
    close_seconds = 0.0
    started = perf_counter_ns()
    try:
        while trainer.global_step < int(train_config["total_timesteps"]):
            step_started = perf_counter()
            trainer.evaluate()
            evaluate_seconds += perf_counter() - step_started

            step_started = perf_counter()
            trainer.train()
            train_seconds += perf_counter() - step_started

        step_started = perf_counter()
        trainer.evaluate()
        final_evaluate_seconds += perf_counter() - step_started
        global_step = int(trainer.global_step)
    finally:
        stage_started = perf_counter()
        trainer.close()
        close_seconds += perf_counter() - stage_started

    elapsed_nanos = perf_counter_ns() - started
    production_elapsed_nanos = max(
        0,
        int(elapsed_nanos - round(float(final_evaluate_seconds) * 1_000_000_000.0)),
    )
    production_env_steps_per_second = (
        0.0
        if production_elapsed_nanos <= 0
        else global_step * 1_000_000_000.0 / float(production_elapsed_nanos)
    )
    wall_clock_env_steps_per_second = (
        0.0
        if elapsed_nanos <= 0
        else global_step * 1_000_000_000.0 / float(elapsed_nanos)
    )
    return TrainBenchmarkMeasurement(
        metric_scope="production_fast_path_v1",
        runner_mode=CORE_INPROCESS_RUNNER_MODE,
        logging_mode=str(logging_mode),
        wandb_mode="disabled",
        env_count=int(config["num_envs"]),
        total_timesteps=int(total_timesteps),
        global_step=global_step,
        elapsed_nanos=int(production_elapsed_nanos),
        wall_clock_elapsed_nanos=int(elapsed_nanos),
        env_steps_per_second=float(production_env_steps_per_second),
        production_env_steps_per_second=float(production_env_steps_per_second),
        wall_clock_env_steps_per_second=float(wall_clock_env_steps_per_second),
        evaluate_seconds=float(evaluate_seconds),
        train_seconds=float(train_seconds),
        final_evaluate_seconds=float(final_evaluate_seconds),
        runner_stage_seconds={
            "vecenv_build_seconds": float(vecenv_build_seconds),
            "policy_build_seconds": float(policy_build_seconds),
            "trainer_init_seconds": float(trainer_init_seconds),
            "evaluate_seconds": float(evaluate_seconds),
            "train_seconds": float(train_seconds),
            "final_evaluate_seconds": float(final_evaluate_seconds),
            "close_seconds": float(close_seconds),
        },
        trainer_bucket_totals=trainer.instrumentation_snapshot(),
        log_records=0,
        artifact_count=0,
        aggressive_log_bursts=int(aggressive_log_bursts if logging_mode == "aggressive" else 0),
        checkpoint_path="",
        run_manifest_path="",
    )


def parse_logging_modes(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_runner_mode(value: str | None) -> str:
    if not value:
        return SMOKE_SUBPROCESS_RUNNER_MODE
    return str(value)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run PR11 training benchmark measurements.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/train_1024env_v0.yaml"),
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--env-count", type=int, default=None)
    parser.add_argument("--logging-modes", type=str, default=None)
    parser.add_argument("--runner-mode", type=str, default=SMOKE_SUBPROCESS_RUNNER_MODE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_train_benchmark(
        args.config,
        total_timesteps_override=args.total_timesteps,
        env_count_override=args.env_count,
        logging_modes_override=parse_logging_modes(args.logging_modes),
        runner_mode=parse_runner_mode(args.runner_mode),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
