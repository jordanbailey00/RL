from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np
import pufferlib
import pufferlib.vector
import yaml

from fight_caves_rl.benchmarks.common import BenchmarkContext, build_benchmark_context
from fight_caves_rl.envs.puffer_encoding import (
    build_policy_action_space,
    build_policy_observation_space,
)
from fight_caves_rl.policies.mlp import MultiDiscreteMLPPolicy
from fight_caves_rl.puffer.factory import build_puffer_train_config, load_smoke_train_config
from fight_caves_rl.puffer.trainer import ConfigurablePuffeRL


@dataclass(frozen=True)
class TrainCeilingMeasurement:
    metric_scope: str
    env_count: int
    total_timesteps: int
    global_step: int
    elapsed_seconds: float
    env_steps_per_second: float
    diagnostic_env_steps_per_second: float
    evaluate_seconds: float
    train_seconds: float
    final_evaluate_seconds: float
    runner_stage_seconds: dict[str, float]
    trainer_bucket_totals: dict[str, dict[str, float | int]]
    evaluate_iterations: int
    train_iterations: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrainCeilingReport:
    created_at: str
    metric_contract_id: str
    config_id: str
    measurements: tuple[TrainCeilingMeasurement, ...]
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
        raise RuntimeError("Train ceiling benchmark does not build artifacts.")

    def update_config(self, *args: Any, **kwargs: Any) -> None:
        return None

    def log_artifact(self, *args: Any, **kwargs: Any) -> None:
        return None

    def finish(self) -> None:
        return None


class _FakeVecEnv:
    reset = pufferlib.vector.reset
    step = pufferlib.vector.step

    def __init__(self, env_count: int) -> None:
        self.driver_env = self
        self.agents_per_batch = int(env_count)
        self.num_agents = self.agents_per_batch
        self.single_observation_space = build_policy_observation_space()
        self.single_action_space = build_policy_action_space()
        self.action_space = pufferlib.spaces.joint_space(
            self.single_action_space,
            self.agents_per_batch,
        )
        self.observation_space = pufferlib.spaces.joint_space(
            self.single_observation_space,
            self.agents_per_batch,
        )
        self.emulated = {
            "observation_dtype": self.single_observation_space.dtype,
            "emulated_observation_dtype": self.single_observation_space.dtype,
        }
        self.agent_ids = np.arange(self.num_agents)
        self.flag = pufferlib.vector.RESET
        self.initialized = False
        observation_dim = int(self.single_observation_space.shape[0])
        self._observations = np.zeros((self.num_agents, observation_dim), dtype=np.float32)
        self._rewards = np.zeros((self.num_agents,), dtype=np.float32)
        self._terminals = np.zeros((self.num_agents,), dtype=np.bool_)
        self._truncations = np.zeros((self.num_agents,), dtype=np.bool_)
        self._teacher_actions = np.zeros((self.num_agents,), dtype=np.int32)
        self._agent_ids = np.arange(self.num_agents, dtype=np.int64)
        self._masks = np.ones((self.num_agents,), dtype=np.bool_)
        self._infos = [{} for _ in range(self.num_agents)]

    @property
    def num_envs(self) -> int:
        return self.agents_per_batch

    def async_reset(self, seed: int | None = None) -> None:
        self.flag = pufferlib.vector.RECV

    def send(self, actions: np.ndarray) -> None:
        if not actions.flags.contiguous:
            actions = np.ascontiguousarray(actions)
        pufferlib.vector.send_precheck(self, actions)

    def recv(self) -> tuple[np.ndarray, ...]:
        pufferlib.vector.recv_precheck(self)
        return (
            self._observations,
            self._rewards,
            self._terminals,
            self._truncations,
            self._teacher_actions,
            self._infos,
            self._agent_ids,
            self._masks,
        )

    def notify(self) -> None:
        return None

    def close(self) -> None:
        return None


def run_train_ceiling_benchmark(
    config_path: str | Path,
    *,
    env_counts_override: Sequence[int] | None = None,
    total_timesteps_override: int | None = None,
) -> TrainCeilingReport:
    base_config = load_smoke_train_config(config_path)
    benchmark_config = dict(base_config.get("benchmark", {}))
    total_timesteps = int(
        total_timesteps_override
        if total_timesteps_override is not None
        else benchmark_config.get("total_timesteps", base_config["train"]["total_timesteps"])
    )
    env_counts = tuple(
        int(entry)
        for entry in (
            env_counts_override
            if env_counts_override is not None
            else benchmark_config.get("ceiling_env_counts", (4, 16, 64))
        )
    )
    if not env_counts:
        raise ValueError("Train ceiling benchmark requires at least one env count.")

    reward_config_id = str(base_config["reward_config"])
    curriculum_config_id = str(base_config["curriculum_config"])
    context = build_benchmark_context(
        env_count=max(env_counts),
        logging_mode="disabled",
        replay_mode="disabled",
        dashboard_mode="disabled",
        reward_config_id=reward_config_id,
        curriculum_config_id=curriculum_config_id,
    )
    measurements = tuple(
        _run_train_ceiling_measurement(
            base_config=base_config,
            env_count=env_count,
            total_timesteps=total_timesteps,
        )
        for env_count in env_counts
    )
    return TrainCeilingReport(
        created_at=datetime.now(UTC).isoformat(),
        metric_contract_id="train_ceiling_diagnostic_v1",
        config_id=str(base_config["config_id"]),
        measurements=measurements,
        context=context,
    )


def _run_train_ceiling_measurement(
    *,
    base_config: dict[str, Any],
    env_count: int,
    total_timesteps: int,
) -> TrainCeilingMeasurement:
    config = json.loads(json.dumps(base_config))
    config["num_envs"] = int(env_count)
    config.setdefault("logging", {})["dashboard"] = False
    _clamp_train_config_for_benchmark(config, total_timesteps=total_timesteps)

    stage_started = perf_counter()
    vecenv = _FakeVecEnv(int(env_count))
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
        data_dir=Path("/tmp/fc_train_ceiling_bench"),
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
    evaluate_iterations = 0
    train_iterations = 0
    started = perf_counter()

    while trainer.global_step < int(train_config["total_timesteps"]):
        step_started = perf_counter()
        trainer.evaluate()
        evaluate_seconds += perf_counter() - step_started
        evaluate_iterations += 1

        step_started = perf_counter()
        trainer.train()
        train_seconds += perf_counter() - step_started
        train_iterations += 1

    final_started = perf_counter()
    trainer.evaluate()
    final_evaluate_seconds = perf_counter() - final_started
    elapsed_seconds = perf_counter() - started
    global_step = int(trainer.global_step)
    stage_started = perf_counter()
    trainer.close()
    close_seconds = perf_counter() - stage_started

    return TrainCeilingMeasurement(
        metric_scope="diagnostic_shipped_sync_path_v1",
        env_count=int(env_count),
        total_timesteps=int(total_timesteps),
        global_step=global_step,
        elapsed_seconds=float(elapsed_seconds),
        env_steps_per_second=(
            0.0 if elapsed_seconds <= 0.0 else float(global_step) / float(elapsed_seconds)
        ),
        diagnostic_env_steps_per_second=(
            0.0 if elapsed_seconds <= 0.0 else float(global_step) / float(elapsed_seconds)
        ),
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
        evaluate_iterations=int(evaluate_iterations),
        train_iterations=int(train_iterations),
    )


def _clamp_train_config_for_benchmark(
    config: dict[str, Any],
    *,
    total_timesteps: int,
) -> None:
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


def parse_env_counts(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    values = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    return values if values else None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run a fake-env learner ceiling benchmark.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark/train_1024env_v0.yaml"),
    )
    parser.add_argument("--env-counts", type=str, default=None)
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run_train_ceiling_benchmark(
        args.config,
        env_counts_override=parse_env_counts(args.env_counts),
        total_timesteps_override=args.total_timesteps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
