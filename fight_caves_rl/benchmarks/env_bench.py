from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter_ns
from typing import Any

import numpy as np

from fight_caves_rl.benchmarks.common import BenchmarkContext, build_benchmark_context
from fight_caves_rl.puffer.factory import (
    build_policy_episode_env,
    load_smoke_train_config,
    make_vecenv,
)
from fight_caves_rl.utils.paths import repo_root


@dataclass(frozen=True)
class EnvBenchmarkMeasurement:
    label: str
    env_count: int
    rounds: int
    total_env_steps: int
    elapsed_nanos: int
    env_steps_per_second: float
    tick_rounds_per_second: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EnvBenchmarkReport:
    created_at: str
    config_id: str
    env_count: int
    wrapper_env_count: int
    bridge_protocol_id: str
    bridge_protocol_version: int
    wrapper: EnvBenchmarkMeasurement
    measurement: EnvBenchmarkMeasurement
    speedup_vs_wrapper: float
    context: BenchmarkContext

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["wrapper"] = self.wrapper.to_dict()
        payload["measurement"] = self.measurement.to_dict()
        payload["context"] = self.context.to_dict()
        return payload


def run_env_benchmark(
    config_path: str | Path,
    *,
    rounds_override: int | None = None,
    env_count_override: int | None = None,
    wrapper_env_count_override: int | None = None,
) -> EnvBenchmarkReport:
    config = load_smoke_train_config(config_path)
    if env_count_override is not None:
        config["num_envs"] = int(env_count_override)

    benchmark_config = dict(config.get("benchmark", {}))
    rounds = int(rounds_override if rounds_override is not None else benchmark_config.get("rounds", 128))
    if rounds <= 0:
        raise ValueError(f"rounds must be > 0, got {rounds}.")
    wrapper_env_count = int(
        wrapper_env_count_override
        if wrapper_env_count_override is not None
        else benchmark_config.get("wrapper_env_count", 1)
    )
    if wrapper_env_count <= 0:
        raise ValueError(f"wrapper_env_count must be > 0, got {wrapper_env_count}.")

    reward_config_id = str(config["reward_config"])
    curriculum_config_id = str(config["curriculum_config"])
    dashboard_enabled = bool(dict(config.get("logging", {})).get("dashboard", False))
    context = build_benchmark_context(
        env_count=int(config["num_envs"]),
        logging_mode="benchmark_standard",
        replay_mode="disabled",
        dashboard_mode="enabled" if dashboard_enabled else "disabled",
        reward_config_id=reward_config_id,
        curriculum_config_id=curriculum_config_id,
    )
    wrapper = _collect_measurement_subprocess(
        config_path=config_path,
        mode="wrapper",
        rounds=rounds,
        env_count=int(config["num_envs"]),
        wrapper_env_count=wrapper_env_count,
    )
    vecenv = _collect_measurement_subprocess(
        config_path=config_path,
        mode="vecenv",
        rounds=rounds,
        env_count=int(config["num_envs"]),
        wrapper_env_count=wrapper_env_count,
    )
    speedup = (
        0.0
        if wrapper.env_steps_per_second <= 0.0
        else vecenv.env_steps_per_second / wrapper.env_steps_per_second
    )
    return EnvBenchmarkReport(
        created_at=datetime.now(UTC).isoformat(),
        config_id=str(config["config_id"]),
        env_count=int(config["num_envs"]),
        wrapper_env_count=int(wrapper_env_count),
        bridge_protocol_id=context.bridge_protocol_id,
        bridge_protocol_version=context.bridge_protocol_version,
        wrapper=wrapper,
        measurement=vecenv,
        speedup_vs_wrapper=float(speedup),
        context=context,
    )


def _run_wrapper_measurement(
    config: dict[str, Any],
    *,
    rounds: int,
    env_count: int,
) -> EnvBenchmarkMeasurement:
    env_settings = dict(config.get("env", {}))
    envs = [
        build_policy_episode_env(
            {
                **env_settings,
                "account_name_prefix": f"{env_settings.get('account_name_prefix', 'rl_env_bench')}_wrapper_{slot_index}",
            },
            reward_config_id=str(config["reward_config"]),
            curriculum_config_id=str(config["curriculum_config"]),
        )
        for slot_index in range(env_count)
    ]
    try:
        zero_action = np.zeros(len(envs[0].action_space.nvec), dtype=np.int32)
        seed_base = int(config["train"]["seed"])
        for slot_index, env in enumerate(envs):
            env.reset(seed=seed_base + slot_index)

        started = perf_counter_ns()
        for _ in range(rounds):
            for slot_index, env in enumerate(envs):
                _obs, _reward, terminated, truncated, _info = env.step(zero_action)
                if terminated or truncated:
                    env.reset(seed=seed_base + slot_index)
        elapsed_nanos = perf_counter_ns() - started
        total_env_steps = env_count * rounds
        return _build_measurement(
            label="wrapper_sequential",
            env_count=env_count,
            rounds=rounds,
            total_env_steps=total_env_steps,
            elapsed_nanos=elapsed_nanos,
        )
    finally:
        for env in envs:
            env.close()


def _run_vecenv_measurement(
    config: dict[str, Any],
    *,
    rounds: int,
    warmup_rounds: int,
) -> EnvBenchmarkMeasurement:
    vecenv = make_vecenv(config)
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
        return _build_measurement(
            label="vecenv_lockstep",
            env_count=int(config["num_envs"]),
            rounds=rounds,
            total_env_steps=total_env_steps,
            elapsed_nanos=elapsed_nanos,
        )
    finally:
        vecenv.close()


def _build_measurement(
    *,
    label: str,
    env_count: int,
    rounds: int,
    total_env_steps: int,
    elapsed_nanos: int,
) -> EnvBenchmarkMeasurement:
    return EnvBenchmarkMeasurement(
        label=label,
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
    mode: str,
    rounds: int,
    env_count: int,
    wrapper_env_count: int,
) -> EnvBenchmarkMeasurement:
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        output_path = Path(handle.name)
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "fight_caves_rl.benchmarks.env_bench",
                "--config",
                str(config_path),
                "--rounds",
                str(rounds),
                "--env-count",
                str(env_count),
                "--wrapper-env-count",
                str(wrapper_env_count),
                "--mode",
                mode,
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
                "Env benchmark subprocess failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        return EnvBenchmarkMeasurement(**payload)
    finally:
        output_path.unlink(missing_ok=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run PR11 env benchmark measurements.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/vecenv_256env_v0.yaml"),
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--env-count", type=int, default=None)
    parser.add_argument("--wrapper-env-count", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("combined", "wrapper", "vecenv"),
        default="combined",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = load_smoke_train_config(args.config)
    if args.env_count is not None:
        config["num_envs"] = int(args.env_count)
    benchmark_config = dict(config.get("benchmark", {}))
    rounds = int(args.rounds if args.rounds is not None else benchmark_config.get("rounds", 128))
    wrapper_env_count = int(
        args.wrapper_env_count
        if args.wrapper_env_count is not None
        else benchmark_config.get("wrapper_env_count", 1)
    )

    if args.mode == "wrapper":
        payload = _run_wrapper_measurement(
            config,
            rounds=rounds,
            env_count=wrapper_env_count,
        ).to_dict()
    elif args.mode == "vecenv":
        payload = _run_vecenv_measurement(
            config,
            rounds=rounds,
            warmup_rounds=int(benchmark_config.get("warmup_rounds", 0)),
        ).to_dict()
    else:
        payload = run_env_benchmark(
            args.config,
            rounds_override=args.rounds,
            env_count_override=args.env_count,
            wrapper_env_count_override=args.wrapper_env_count,
        ).to_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
