from pathlib import Path

import pufferlib.pytorch
import pytest
import torch
import yaml

from fight_caves_rl.benchmarks.common import BenchmarkContext, BenchmarkHardwareProfile
from fight_caves_rl.benchmarks.train_bench import PROTOTYPE_SYNC_RUNNER_MODE, run_train_benchmark
from fight_caves_rl.benchmarks.train_ceiling_bench import _FakeVecEnv
from fight_caves_rl.policies.mlp import MultiDiscreteMLPPolicy
from fight_caves_rl.puffer.factory import build_puffer_train_config, load_smoke_train_config
from fight_caves_rl.puffer.production_trainer import (
    PrototypeProductionTrainer,
    _sample_multidiscrete_logits,
)


def _prototype_train_config(tmp_path: Path) -> dict[str, object]:
    config = load_smoke_train_config()
    config["num_envs"] = 2
    config["train"]["total_timesteps"] = 8
    config["train"]["batch_size"] = 8
    config["train"]["minibatch_size"] = 8
    config["train"]["max_minibatch_size"] = 8
    config["train"]["bptt_horizon"] = 4
    config["train"]["checkpoint_interval"] = 1_000_000
    config.setdefault("logging", {})["dashboard"] = False
    return build_puffer_train_config(
        config,
        data_dir=tmp_path,
        total_timesteps=8,
    )


def test_prototype_production_trainer_runs_single_epoch(tmp_path: Path):
    train_config = _prototype_train_config(tmp_path)
    vecenv = _FakeVecEnv(2)
    policy = MultiDiscreteMLPPolicy.from_spaces(
        vecenv.single_observation_space,
        vecenv.single_action_space,
        hidden_size=128,
    )
    trainer = PrototypeProductionTrainer(train_config, vecenv, policy)

    while trainer.global_step < int(train_config["total_timesteps"]):
        trainer.collect_rollout()
        trainer.train_update()

    trainer.close()
    snapshot = trainer.instrumentation_snapshot()

    assert trainer.global_step == int(train_config["total_timesteps"])
    assert "rollout_policy_forward" in snapshot
    assert "update_policy_forward" in snapshot
    assert "trainer_close" in snapshot


def test_prototype_production_trainer_rejects_rnn(tmp_path: Path):
    train_config = _prototype_train_config(tmp_path)
    train_config["use_rnn"] = True
    vecenv = _FakeVecEnv(2)
    policy = MultiDiscreteMLPPolicy.from_spaces(
        vecenv.single_observation_space,
        vecenv.single_action_space,
        hidden_size=128,
    )

    with pytest.raises(ValueError, match="does not support recurrent policies"):
        PrototypeProductionTrainer(train_config, vecenv, policy)


def test_multidiscrete_sampler_matches_pufferlib_for_action_conditioned_path():
    logits = (
        torch.randn(4, 7),
        torch.randn(4, 5),
        torch.randn(4, 3),
    )
    action = torch.tensor(
        [
            [1, 2, 0],
            [0, 4, 2],
            [3, 1, 1],
            [2, 0, 2],
        ],
        dtype=torch.int32,
    )

    _, expected_logprob, expected_entropy = pufferlib.pytorch.sample_logits(
        logits,
        action=action,
    )
    _, logprob, entropy = _sample_multidiscrete_logits(logits, action=action)

    assert torch.allclose(logprob, expected_logprob)
    assert torch.allclose(entropy, expected_entropy)


def test_multidiscrete_sampler_returns_self_consistent_sample():
    torch.manual_seed(123)
    logits = (
        torch.randn(4, 7),
        torch.randn(4, 5),
        torch.randn(4, 3),
    )

    action, logprob, entropy = _sample_multidiscrete_logits(logits)
    _, recomputed_logprob, recomputed_entropy = _sample_multidiscrete_logits(
        logits,
        action=action,
    )

    assert action.shape == (4, 3)
    assert action.dtype == torch.int32
    assert torch.allclose(logprob, recomputed_logprob)
    assert torch.allclose(entropy, recomputed_entropy)


def test_run_train_benchmark_supports_prototype_runner(tmp_path: Path, monkeypatch):
    config = load_smoke_train_config()
    config["config_id"] = "prototype_unit_v0"
    config["num_envs"] = 2
    config["train"]["total_timesteps"] = 8
    config["train"]["batch_size"] = 8
    config["train"]["minibatch_size"] = 8
    config["train"]["max_minibatch_size"] = 8
    config["train"]["bptt_horizon"] = 4
    config.setdefault("logging", {})["dashboard"] = False
    config_path = tmp_path / "prototype_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")

    monkeypatch.setattr(
        "fight_caves_rl.benchmarks.train_bench.make_vecenv",
        lambda cfg, backend="subprocess", instrumentation_enabled=False: _FakeVecEnv(
            int(cfg["num_envs"])
        ),
    )
    monkeypatch.setattr(
        "fight_caves_rl.benchmarks.train_bench.build_benchmark_context",
        lambda **kwargs: BenchmarkContext(
            benchmark_profile_id="official_profile_v0",
            benchmark_profile_version=1,
            benchmark_mode="benchmark",
            bridge_mode="subprocess_isolated_jvm",
            reward_config_id=str(kwargs["reward_config_id"]),
            curriculum_config_id=str(kwargs["curriculum_config_id"]),
            replay_mode=str(kwargs["replay_mode"]),
            logging_mode=str(kwargs["logging_mode"]),
            dashboard_mode=str(kwargs["dashboard_mode"]),
            env_count=int(kwargs["env_count"]),
            rl_repo="/tmp/rl",
            rl_commit_sha="deadbeef",
            sim_repo="/tmp/sim",
            sim_commit_sha="deadbeef",
            rsps_repo="/tmp/rsps",
            rsps_commit_sha="deadbeef",
            sim_artifact_task="headlessDistZip",
            sim_artifact_path="/tmp/headless.zip",
            bridge_protocol_id="bridge_contract_v0",
            bridge_protocol_version=1,
            observation_schema_id="obs_schema_v0",
            observation_schema_version=1,
            action_schema_id="action_schema_v0",
            action_schema_version=1,
            episode_start_contract_id="episode_start_v0",
            episode_start_contract_version=1,
            pufferlib_distribution="pufferlib-core",
            pufferlib_version="3.0.17",
            hardware_profile=BenchmarkHardwareProfile(
                platform="linux-test",
                system="Linux",
                release="test",
                version="test",
                machine="x86_64",
                processor="x86_64",
                cpu_count=1,
                python_implementation="CPython",
                python_version="3.11.0",
                host_class="linux_native",
                is_wsl=False,
                performance_source_of_truth=True,
                java_runtime_version=None,
                java_vm_name=None,
            ),
        ),
    )

    report = run_train_benchmark(
        config_path,
        total_timesteps_override=8,
        env_count_override=2,
        logging_modes_override=("disabled",),
        runner_mode=PROTOTYPE_SYNC_RUNNER_MODE,
    )

    measurement = report.measurements[0]
    assert report.metric_contract_id == "train_benchmark_production_v1"
    assert measurement.runner_mode == PROTOTYPE_SYNC_RUNNER_MODE
    assert measurement.logging_mode == "disabled"
    assert measurement.final_evaluate_seconds == 0.0
    assert measurement.production_env_steps_per_second > 0.0
    assert "rollout_policy_forward" in measurement.trainer_bucket_totals
    assert "update_policy_forward" in measurement.trainer_bucket_totals
    assert measurement.env_hot_path_bucket_totals == {}
    assert measurement.memory_profile["combined_peak_rss_kib"] >= 0
