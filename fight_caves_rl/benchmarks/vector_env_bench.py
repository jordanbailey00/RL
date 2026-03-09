from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from fight_caves_rl.puffer.factory import load_smoke_train_config, make_vecenv


@dataclass(frozen=True)
class VecEnvBenchmarkMeasurement:
    env_count: int
    rounds: int
    total_env_steps: int
    elapsed_nanos: int
    env_steps_per_second: float
    tick_rounds_per_second: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class VecEnvBenchmarkReport:
    created_at: str
    config_id: str
    env_count: int
    bridge_protocol_id: str
    bridge_protocol_version: int
    measurement: VecEnvBenchmarkMeasurement

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["measurement"] = self.measurement.to_dict()
        return payload


def run_vecenv_benchmark(
    config_path: str | Path,
    *,
    rounds_override: int | None = None,
    env_count_override: int | None = None,
) -> VecEnvBenchmarkReport:
    config = load_smoke_train_config(config_path)
    if env_count_override is not None:
        config["num_envs"] = int(env_count_override)

    benchmark_config = dict(config.get("benchmark", {}))
    rounds = int(rounds_override if rounds_override is not None else benchmark_config.get("rounds", 128))
    warmup_rounds = int(benchmark_config.get("warmup_rounds", 0))
    if rounds <= 0:
        raise ValueError(f"rounds must be > 0, got {rounds}.")

    vecenv = make_vecenv(config, backend="embedded")
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
        total_env_steps = int(config["num_envs"]) * rounds
        measurement = VecEnvBenchmarkMeasurement(
            env_count=int(config["num_envs"]),
            rounds=rounds,
            total_env_steps=total_env_steps,
            elapsed_nanos=elapsed_nanos,
            env_steps_per_second=(
                0.0
                if elapsed_nanos <= 0
                else total_env_steps * 1_000_000_000.0 / float(elapsed_nanos)
            ),
            tick_rounds_per_second=(
                0.0 if elapsed_nanos <= 0 else rounds * 1_000_000_000.0 / float(elapsed_nanos)
            ),
        )
        return VecEnvBenchmarkReport(
            created_at=datetime.now(UTC).isoformat(),
            config_id=str(config["config_id"]),
            env_count=int(config["num_envs"]),
            bridge_protocol_id=str(vecenv.client.protocol.bridge_protocol_id),
            bridge_protocol_version=int(vecenv.client.protocol.bridge_protocol_version),
            measurement=measurement,
        )
    finally:
        vecenv.close()
