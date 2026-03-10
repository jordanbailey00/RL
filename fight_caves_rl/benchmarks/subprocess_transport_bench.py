from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter_ns

import numpy as np

from fight_caves_rl.benchmarks.common import BenchmarkContext, build_benchmark_context
from fight_caves_rl.envs.shared_memory_transport import (
    PIPE_PICKLE_TRANSPORT_MODE,
    SHARED_MEMORY_TRANSPORT_MODE,
)
from fight_caves_rl.puffer.factory import load_smoke_train_config, make_vecenv
from fight_caves_rl.utils.paths import repo_root


@dataclass(frozen=True)
class SubprocessTransportMeasurement:
    transport_mode: str
    env_count: int
    rounds: int
    total_env_steps: int
    elapsed_nanos: int
    env_steps_per_second: float
    tick_rounds_per_second: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SubprocessTransportBenchmarkReport:
    created_at: str
    config_id: str
    env_count: int
    measurements: tuple[SubprocessTransportMeasurement, ...]
    speedup_vs_pipe_pickle: dict[str, float]
    context: BenchmarkContext

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["measurements"] = [measurement.to_dict() for measurement in self.measurements]
        payload["context"] = self.context.to_dict()
        return payload


def run_subprocess_transport_benchmark(
    config_path: str | Path,
    *,
    rounds_override: int | None = None,
    env_count_override: int | None = None,
    transport_modes_override: tuple[str, ...] | None = None,
) -> SubprocessTransportBenchmarkReport:
    config = load_smoke_train_config(config_path)
    if env_count_override is not None:
        config["num_envs"] = int(env_count_override)

    benchmark_config = dict(config.get("benchmark", {}))
    rounds = int(
        rounds_override
        if rounds_override is not None
        else benchmark_config.get("transport_rounds", benchmark_config.get("rounds", 128))
    )
    if rounds <= 0:
        raise ValueError(f"rounds must be > 0, got {rounds}.")
    transport_modes = tuple(
        transport_modes_override
        if transport_modes_override is not None
        else benchmark_config.get(
            "transport_modes",
            (PIPE_PICKLE_TRANSPORT_MODE, SHARED_MEMORY_TRANSPORT_MODE),
        )
    )

    context = build_benchmark_context(
        env_count=int(config["num_envs"]),
        logging_mode="benchmark_standard",
        replay_mode="disabled",
        dashboard_mode="disabled",
        reward_config_id=str(config["reward_config"]),
        curriculum_config_id=str(config["curriculum_config"]),
    )
    measurements = tuple(
        _collect_measurement_subprocess(
            config_path=config_path,
            rounds=rounds,
            env_count=int(config["num_envs"]),
            transport_mode=str(transport_mode),
        )
        for transport_mode in transport_modes
    )
    pipe_pickle = next(
        (
            measurement
            for measurement in measurements
            if measurement.transport_mode == PIPE_PICKLE_TRANSPORT_MODE
        ),
        None,
    )
    ratios: dict[str, float] = {}
    for measurement in measurements:
        if pipe_pickle is None or pipe_pickle.env_steps_per_second <= 0.0:
            ratios[measurement.transport_mode] = 0.0
            continue
        ratios[measurement.transport_mode] = (
            measurement.env_steps_per_second / pipe_pickle.env_steps_per_second
        )
    return SubprocessTransportBenchmarkReport(
        created_at=datetime.now(UTC).isoformat(),
        config_id=str(config["config_id"]),
        env_count=int(config["num_envs"]),
        measurements=measurements,
        speedup_vs_pipe_pickle=ratios,
        context=context,
    )


def _run_transport_measurement(
    config_path: str | Path,
    *,
    rounds: int,
    env_count: int,
    transport_mode: str,
) -> SubprocessTransportMeasurement:
    config = load_smoke_train_config(config_path)
    config["num_envs"] = int(env_count)
    config.setdefault("logging", {})["dashboard"] = False
    config.setdefault("env", {})["subprocess_transport_mode"] = str(transport_mode)
    benchmark_config = dict(config.get("benchmark", {}))
    warmup_rounds = int(
        benchmark_config.get("transport_warmup_rounds", benchmark_config.get("warmup_rounds", 16))
    )

    vecenv = make_vecenv(config, backend="subprocess")
    try:
        seed = int(config["train"]["seed"])
        zero_action = np.zeros(
            (int(config["num_envs"]), len(vecenv.single_action_space.nvec)),
            dtype=np.int32,
        )
        vecenv.async_reset(seed)
        vecenv.recv()
        for _ in range(warmup_rounds):
            vecenv.send(zero_action)
            vecenv.recv()

        started = perf_counter_ns()
        for _ in range(rounds):
            vecenv.send(zero_action)
            vecenv.recv()
        elapsed_nanos = perf_counter_ns() - started
    finally:
        vecenv.close()

    total_env_steps = int(env_count) * int(rounds)
    return SubprocessTransportMeasurement(
        transport_mode=str(transport_mode),
        env_count=int(env_count),
        rounds=int(rounds),
        total_env_steps=int(total_env_steps),
        elapsed_nanos=int(elapsed_nanos),
        env_steps_per_second=(
            0.0
            if elapsed_nanos <= 0
            else total_env_steps * 1_000_000_000.0 / float(elapsed_nanos)
        ),
        tick_rounds_per_second=(
            0.0 if elapsed_nanos <= 0 else rounds * 1_000_000_000.0 / float(elapsed_nanos)
        ),
    )


def _collect_measurement_subprocess(
    *,
    config_path: str | Path,
    rounds: int,
    env_count: int,
    transport_mode: str,
) -> SubprocessTransportMeasurement:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        output_path = Path(handle.name)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fight_caves_rl.benchmarks.subprocess_transport_bench",
                "--config",
                str(config_path),
                "--rounds",
                str(rounds),
                "--env-count",
                str(env_count),
                "--transport-mode",
                str(transport_mode),
                "--mode",
                "measure",
                "--output",
                str(output_path),
            ],
            cwd=str(repo_root()),
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Subprocess transport benchmark failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return SubprocessTransportMeasurement(**payload)
    finally:
        output_path.unlink(missing_ok=True)


def parse_transport_modes(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run subprocess transport benchmark measurements.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/vecenv_256env_v0.yaml"),
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--env-count", type=int, default=None)
    parser.add_argument(
        "--transport-mode",
        type=str,
        default=PIPE_PICKLE_TRANSPORT_MODE,
    )
    parser.add_argument("--transport-modes", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=("combined", "measure"),
        default="combined",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.mode == "measure":
        config = load_smoke_train_config(args.config)
        env_count = int(args.env_count if args.env_count is not None else config["num_envs"])
        benchmark_config = dict(config.get("benchmark", {}))
        rounds = int(
            args.rounds
            if args.rounds is not None
            else benchmark_config.get("transport_rounds", benchmark_config.get("rounds", 128))
        )
        payload = _run_transport_measurement(
            args.config,
            rounds=rounds,
            env_count=env_count,
            transport_mode=str(args.transport_mode),
        ).to_dict()
    else:
        payload = run_subprocess_transport_benchmark(
            args.config,
            rounds_override=args.rounds,
            env_count_override=args.env_count,
            transport_modes_override=parse_transport_modes(args.transport_modes),
        ).to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
