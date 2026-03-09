from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import yaml

from fight_caves_rl.benchmarks.common import BenchmarkContext, build_benchmark_context
from fight_caves_rl.bridge.batch_client import BatchClientConfig, HeadlessBatchClient
from fight_caves_rl.bridge.buffers import build_step_buffers
from fight_caves_rl.bridge.contracts import HeadlessBootstrapConfig


@dataclass(frozen=True)
class BridgeBenchmarkConfig:
    config_id: str
    env_count: int
    rounds: int
    warmup_rounds: int
    action_id: int
    tick_cap: int
    start_world: bool
    include_future_leakage: bool
    start_wave: int
    seed_base: int


@dataclass(frozen=True)
class BridgeBenchmarkMeasurement:
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
class BridgeBenchmarkReport:
    created_at: str
    config_id: str
    env_count: int
    bridge_protocol_id: str
    bridge_protocol_version: int
    reference: BridgeBenchmarkMeasurement
    batch: BridgeBenchmarkMeasurement
    speedup_vs_reference: float
    context: BenchmarkContext

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["reference"] = self.reference.to_dict()
        payload["batch"] = self.batch.to_dict()
        payload["context"] = self.context.to_dict()
        return payload


def load_bridge_benchmark_config(path: str | Path) -> BridgeBenchmarkConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return BridgeBenchmarkConfig(
        config_id=str(payload["config_id"]),
        env_count=int(payload["env_count"]),
        rounds=int(payload["rounds"]),
        warmup_rounds=int(payload.get("warmup_rounds", 0)),
        action_id=int(payload.get("action_id", 0)),
        tick_cap=int(payload.get("tick_cap", 20_000)),
        start_world=bool(payload.get("start_world", False)),
        include_future_leakage=bool(payload.get("include_future_leakage", False)),
        start_wave=int(payload.get("start_wave", 1)),
        seed_base=int(payload.get("seed_base", 50_000)),
    )


def run_bridge_benchmark(
    config_path: str | Path,
    *,
    rounds_override: int | None = None,
    env_count_override: int | None = None,
) -> BridgeBenchmarkReport:
    config = load_bridge_benchmark_config(config_path)
    if rounds_override is not None or env_count_override is not None:
        config = BridgeBenchmarkConfig(
            **{
                **asdict(config),
                "rounds": int(rounds_override if rounds_override is not None else config.rounds),
                "env_count": int(
                    env_count_override if env_count_override is not None else config.env_count
                ),
            }
        )

    client = HeadlessBatchClient.create(
        BatchClientConfig(
            env_count=config.env_count,
            account_name_prefix=f"bridge_bench_{config.config_id}",
            start_wave=config.start_wave,
            tick_cap=config.tick_cap,
            include_future_leakage=config.include_future_leakage,
            bootstrap=HeadlessBootstrapConfig(start_world=config.start_world),
        )
    )
    try:
        context = build_benchmark_context(
            env_count=int(config.env_count),
            logging_mode="benchmark_minimized",
            replay_mode="disabled",
            dashboard_mode="disabled",
            reward_config_id="reward_sparse_v0",
            curriculum_config_id="curriculum_disabled_v0",
        )
        seeds = [config.seed_base + slot_index for slot_index in range(config.env_count)]
        _warmup(client, config, seeds)
        reference = _run_reference_measurement(client, config, seeds)
        batch = _run_batch_measurement(client, config, seeds)
        speedup = (
            0.0
            if reference.env_steps_per_second <= 0.0
            else batch.env_steps_per_second / reference.env_steps_per_second
        )
        return BridgeBenchmarkReport(
            created_at=datetime.now(UTC).isoformat(),
            config_id=config.config_id,
            env_count=config.env_count,
            bridge_protocol_id=client.protocol.bridge_protocol_id,
            bridge_protocol_version=client.protocol.bridge_protocol_version,
            reference=reference,
            batch=batch,
            speedup_vs_reference=float(speedup),
            context=context,
        )
    finally:
        client.close()


def _warmup(client: HeadlessBatchClient, config: BridgeBenchmarkConfig, seeds: list[int]) -> None:
    if config.warmup_rounds <= 0:
        return
    actions = [config.action_id] * config.env_count
    client.reset_batch(seeds=seeds)
    for _ in range(config.warmup_rounds):
        client.step_batch(actions)


def _run_reference_measurement(
    client: HeadlessBatchClient,
    config: BridgeBenchmarkConfig,
    seeds: list[int],
) -> BridgeBenchmarkMeasurement:
    if config.env_count == 1:
        return _run_single_slot_reference(client, config, seeds[0])
    return _run_multislot_reference(client, config, seeds)


def _run_batch_measurement(
    client: HeadlessBatchClient,
    config: BridgeBenchmarkConfig,
    seeds: list[int],
) -> BridgeBenchmarkMeasurement:
    if config.env_count == 1:
        return _run_single_slot_trace_batch(client, config, seeds[0])
    return _run_multislot_batch(client, config, seeds)


def _run_single_slot_reference(
    client: HeadlessBatchClient,
    config: BridgeBenchmarkConfig,
    seed: int,
) -> BridgeBenchmarkMeasurement:
    response = client.reset_batch(seeds=[seed], options=[{"start_wave": config.start_wave}])
    del response
    slot = client._slot(0)
    elapsed_nanos = 0
    for _ in range(config.rounds):
        started = perf_counter_ns()
        snapshot = client.client.step_once(
            slot.player,
            config.action_id,
            include_future_leakage=config.include_future_leakage,
        )
        elapsed_nanos += perf_counter_ns() - started
        slot.episode_steps += 1
        slot.last_observation = snapshot.observation
    return _build_measurement(
        label="reference_step_once",
        env_count=1,
        rounds=config.rounds,
        elapsed_nanos=elapsed_nanos,
    )


def _run_single_slot_trace_batch(
    client: HeadlessBatchClient,
    config: BridgeBenchmarkConfig,
    seed: int,
) -> BridgeBenchmarkMeasurement:
    client.reset_batch(seeds=[seed], options=[{"start_wave": config.start_wave}])
    report = client.run_action_trace(
        0,
        [config.action_id] * config.rounds,
        ticks_after=1,
        observe_every=0,
    )
    return _build_measurement(
        label="batch_trace",
        env_count=1,
        rounds=config.rounds,
        elapsed_nanos=int(report["elapsed_nanos"]),
    )


def _run_multislot_reference(
    client: HeadlessBatchClient,
    config: BridgeBenchmarkConfig,
    seeds: list[int],
) -> BridgeBenchmarkMeasurement:
    client.reset_batch(
        seeds=seeds,
        options=[{"start_wave": config.start_wave} for _ in range(config.env_count)],
    )
    actions = [config.action_id] * config.env_count
    elapsed_nanos = 0
    for _ in range(config.rounds):
        response = client.step_reference(actions)
        build_step_buffers(response.results)
        elapsed_nanos += response.elapsed_nanos
    return _build_measurement(
        label="reference_lockstep",
        env_count=config.env_count,
        rounds=config.rounds,
        elapsed_nanos=elapsed_nanos,
    )


def _run_multislot_batch(
    client: HeadlessBatchClient,
    config: BridgeBenchmarkConfig,
    seeds: list[int],
) -> BridgeBenchmarkMeasurement:
    client.reset_batch(
        seeds=seeds,
        options=[{"start_wave": config.start_wave} for _ in range(config.env_count)],
    )
    actions = [config.action_id] * config.env_count
    elapsed_nanos = 0
    for _ in range(config.rounds):
        response = client.step_batch(actions)
        build_step_buffers(response.results)
        elapsed_nanos += response.elapsed_nanos
    return _build_measurement(
        label="batch_lockstep",
        env_count=config.env_count,
        rounds=config.rounds,
        elapsed_nanos=elapsed_nanos,
    )


def _build_measurement(
    *,
    label: str,
    env_count: int,
    rounds: int,
    elapsed_nanos: int,
) -> BridgeBenchmarkMeasurement:
    total_env_steps = int(env_count) * int(rounds)
    env_steps_per_second = (
        0.0
        if elapsed_nanos <= 0 or total_env_steps <= 0
        else total_env_steps * 1_000_000_000.0 / float(elapsed_nanos)
    )
    tick_rounds_per_second = (
        0.0
        if elapsed_nanos <= 0 or rounds <= 0
        else rounds * 1_000_000_000.0 / float(elapsed_nanos)
    )
    return BridgeBenchmarkMeasurement(
        label=label,
        env_count=int(env_count),
        rounds=int(rounds),
        total_env_steps=total_env_steps,
        elapsed_nanos=int(elapsed_nanos),
        env_steps_per_second=float(env_steps_per_second),
        tick_rounds_per_second=float(tick_rounds_per_second),
    )
