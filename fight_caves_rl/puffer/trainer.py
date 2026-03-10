from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys
import time
from time import perf_counter
from typing import Any

import numpy as np
import pufferlib
import pufferlib.pufferl
import torch
from fight_caves_rl.envs.shared_memory_transport import PIPE_PICKLE_TRANSPORT_MODE
from fight_caves_rl.logging.wandb_client import WandbRunLogger
from fight_caves_rl.manifests.run_manifest import (
    build_train_run_manifest,
    write_run_manifest,
)
from fight_caves_rl.policies.checkpointing import (
    build_checkpoint_metadata,
    write_checkpoint_metadata,
)
from fight_caves_rl.policies.mlp import MultiDiscreteMLPPolicy
from fight_caves_rl.puffer.factory import (
    build_puffer_train_config,
    build_train_output_dir,
    load_smoke_train_config,
    make_vecenv,
)
from fight_caves_rl.replay.eval_runner import run_replay_eval
from fight_caves_rl.utils.config import load_bootstrap_config


TrainerInstrumentationSnapshot = dict[str, dict[str, float | int]]


class _TrainerInstrumentation:
    def __init__(self) -> None:
        self._seconds: defaultdict[str, float] = defaultdict(float)
        self._calls: defaultdict[str, int] = defaultdict(int)

    def record(self, bucket: str, seconds: float) -> None:
        self._seconds[str(bucket)] += max(0.0, float(seconds))
        self._calls[str(bucket)] += 1

    def snapshot(self) -> TrainerInstrumentationSnapshot:
        return {
            name: {
                "seconds": float(self._seconds[name]),
                "calls": int(self._calls[name]),
            }
            for name in sorted(self._seconds)
        }


class ConfigurablePuffeRL(pufferlib.pufferl.PuffeRL):
    """Thin PuffeRL wrapper that keeps dashboard rendering opt-in and TTY-bound."""

    def __init__(
        self,
        *args: Any,
        dashboard_enabled: bool = True,
        checkpointing_enabled: bool = True,
        profiling_enabled: bool = True,
        utilization_enabled: bool = True,
        logging_enabled: bool = True,
        instrumentation_enabled: bool = False,
        **kwargs: Any,
    ) -> None:
        self._dashboard_enabled = bool(dashboard_enabled)
        self._checkpointing_enabled = bool(checkpointing_enabled)
        self._profiling_enabled = bool(profiling_enabled)
        self._utilization_enabled = bool(utilization_enabled)
        self._logging_enabled = bool(logging_enabled)
        self._instrumentation_enabled = bool(instrumentation_enabled)
        self._instrumentation = (
            _TrainerInstrumentation() if self._instrumentation_enabled else None
        )
        super().__init__(*args, **kwargs)
        if not self._profiling_enabled:
            self.profile = _NullProfile()
        if not self._utilization_enabled:
            self.utilization.stop()
            self.utilization = _NullUtilization()
        if not self._logging_enabled:
            # Suppress periodic mean_and_log work in benchmark-only runs.
            self.last_log_time = float("inf")

    def print_dashboard(self, *args: Any, **kwargs: Any) -> None:
        if not self._dashboard_enabled:
            return
        super().print_dashboard(*args, **kwargs)

    def mean_and_log(self) -> dict[str, Any] | None:
        started = perf_counter()
        if not self._logging_enabled:
            self._record_instrumentation("trainer_mean_and_log_skipped", perf_counter() - started)
            return {}
        result = super().mean_and_log()
        self._record_instrumentation("trainer_mean_and_log", perf_counter() - started)
        return result

    def instrumentation_snapshot(self) -> TrainerInstrumentationSnapshot:
        instrumentation = getattr(self, "_instrumentation", None)
        if instrumentation is None:
            return {}
        return instrumentation.snapshot()

    def _record_instrumentation(self, bucket: str, seconds: float) -> None:
        instrumentation = getattr(self, "_instrumentation", None)
        if instrumentation is None:
            return
        instrumentation.record(bucket, seconds)

    @pufferlib.pufferl.record
    def evaluate(self):
        if not self._instrumentation_enabled:
            return super().evaluate()

        profile = self.profile
        epoch = self.epoch
        profile("eval", epoch)
        profile("eval_misc", epoch, nest=True)
        evaluate_started = perf_counter()

        config = self.config
        device = config["device"]

        if config["use_rnn"]:
            reset_started = perf_counter()
            for k in self.lstm_h:
                self.lstm_h[k].zero_()
                self.lstm_c[k].zero_()
            self._record_instrumentation("eval_rnn_reset", perf_counter() - reset_started)

        self.full_rows = 0
        while self.full_rows < self.segments:
            recv_started = perf_counter()
            profile("env", epoch)
            o, r, d, t, ta, info, env_id, mask = self.vecenv.recv()
            self._record_instrumentation("eval_env_recv", perf_counter() - recv_started)

            misc_started = perf_counter()
            profile("eval_misc", epoch)
            env_id = slice(env_id[0], env_id[-1] + 1)
            done_mask = d + t
            self.global_step += int(mask.sum())
            self._record_instrumentation("eval_misc", perf_counter() - misc_started)

            copy_started = perf_counter()
            profile("eval_copy", epoch)
            o = torch.as_tensor(o)
            o_device = o.to(device)
            r = torch.as_tensor(r).to(device)
            d = torch.as_tensor(d).to(device)
            self._record_instrumentation("eval_tensor_copy", perf_counter() - copy_started)

            forward_started = perf_counter()
            profile("eval_forward", epoch)
            with torch.no_grad(), self.amp_context:
                state = dict(
                    reward=r,
                    done=d,
                    env_id=env_id,
                    mask=mask,
                )

                if config["use_rnn"]:
                    state["lstm_h"] = self.lstm_h[env_id.start]
                    state["lstm_c"] = self.lstm_c[env_id.start]

                logits, value = self.policy.forward_eval(o_device, state)
                action, logprob, _ = pufferlib.pytorch.sample_logits(logits)
                r = torch.clamp(r, -1, 1)
            self._record_instrumentation("eval_policy_forward", perf_counter() - forward_started)

            write_started = perf_counter()
            profile("eval_copy", epoch)
            with torch.no_grad():
                if config["use_rnn"]:
                    self.lstm_h[env_id.start] = state["lstm_h"]
                    self.lstm_c[env_id.start] = state["lstm_c"]

                l = self.ep_lengths[env_id.start].item()
                batch_rows = slice(
                    self.ep_indices[env_id.start].item(),
                    1 + self.ep_indices[env_id.stop - 1].item(),
                )

                if config["cpu_offload"]:
                    self.observations[batch_rows, l] = o
                else:
                    self.observations[batch_rows, l] = o_device

                self.actions[batch_rows, l] = action
                self.logprobs[batch_rows, l] = logprob
                self.rewards[batch_rows, l] = r
                self.terminals[batch_rows, l] = d.float()
                self.values[batch_rows, l] = value.flatten()

                self.ep_lengths[env_id] += 1
                if l + 1 >= config["bptt_horizon"]:
                    num_full = env_id.stop - env_id.start
                    self.ep_indices[env_id] = self.free_idx + torch.arange(
                        num_full,
                        device=config["device"],
                    ).int()
                    self.ep_lengths[env_id] = 0
                    self.free_idx += num_full
                    self.full_rows += num_full
            self._record_instrumentation("eval_rollout_write", perf_counter() - write_started)

            action_started = perf_counter()
            with torch.no_grad():
                action = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action = np.clip(
                        action,
                        self.vecenv.action_space.low,
                        self.vecenv.action_space.high,
                    )
            self._record_instrumentation("eval_action_to_numpy", perf_counter() - action_started)

            stats_started = perf_counter()
            profile("eval_misc", epoch)
            for i in info:
                for k, v in pufferlib.unroll_nested_dict(i):
                    if isinstance(v, np.ndarray):
                        v = v.tolist()
                    elif isinstance(v, (list, tuple)):
                        self.stats[k].extend(v)
                    else:
                        self.stats[k].append(v)
            self._record_instrumentation("eval_info_stats", perf_counter() - stats_started)

            send_started = perf_counter()
            profile("env", epoch)
            self.vecenv.send(action)
            self._record_instrumentation("eval_env_send", perf_counter() - send_started)

            done_started = perf_counter()
            _ = done_mask
            self._record_instrumentation("eval_done_mask", perf_counter() - done_started)

        reset_started = perf_counter()
        profile("eval_misc", epoch)
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        self._record_instrumentation("eval_reset_state", perf_counter() - reset_started)
        self._record_instrumentation("eval_total", perf_counter() - evaluate_started)
        profile.end()
        return self.stats

    @pufferlib.pufferl.record
    def train(self):
        if self._logging_enabled:
            return super().train()

        profile = self.profile
        epoch = self.epoch
        profile("train", epoch)
        config = self.config
        device = config["device"]
        train_started = perf_counter()

        b0 = config["prio_beta0"]
        a = config["prio_alpha"]
        clip_coef = config["clip_coef"]
        vf_clip = config["vf_clip_coef"]
        anneal_beta = b0 + (1 - b0) * a * self.epoch / self.total_epochs
        self.ratio[:] = 1

        for mb in range(self.total_minibatches):
            profile("train_misc", epoch, nest=True)
            self.amp_context.__enter__()

            shape = self.values.shape
            bucket_started = perf_counter()
            advantages = torch.zeros(shape, device=device)
            advantages = pufferlib.pufferl.compute_puff_advantage(
                self.values,
                self.rewards,
                self.terminals,
                self.ratio,
                advantages,
                config["gamma"],
                config["gae_lambda"],
                config["vtrace_rho_clip"],
                config["vtrace_c_clip"],
            )
            self._record_instrumentation("train_advantage", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            profile("train_copy", epoch)
            adv = advantages.abs().sum(axis=1)
            prio_weights = torch.nan_to_num(adv**a, 0, 0, 0)
            prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]
            self._record_instrumentation("train_minibatch_prepare", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            profile("train_forward", epoch)
            if not config["use_rnn"]:
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)

            state = dict(
                action=mb_actions,
                lstm_h=None,
                lstm_c=None,
            )

            logits, newvalue = self.policy(mb_obs, state)
            _, newlogprob, entropy = pufferlib.pytorch.sample_logits(logits, action=mb_actions)
            self._record_instrumentation("train_policy_forward", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            profile("train_misc", epoch)
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            adv = mb_prio * (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            pg_loss1 = -adv * ratio
            pg_loss2 = -adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()

            loss = pg_loss + config["vf_coef"] * v_loss - config["ent_coef"] * entropy_loss
            self.amp_context.__enter__()

            self.values[idx] = newvalue.detach().float()
            self._record_instrumentation("train_loss_compute", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            profile("learn", epoch)
            loss.backward()
            self._record_instrumentation("train_backward", perf_counter() - bucket_started)

            if (mb + 1) % self.accumulate_minibatches == 0:
                bucket_started = perf_counter()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), config["max_grad_norm"])
                self.optimizer.step()
                self.optimizer.zero_grad()
                self._record_instrumentation("train_optimizer_step", perf_counter() - bucket_started)

        bucket_started = perf_counter()
        profile("train_misc", epoch)
        if config["anneal_lr"]:
            self.scheduler.step()
        self._record_instrumentation("train_scheduler", perf_counter() - bucket_started)

        bucket_started = perf_counter()
        self.losses = {}
        self.epoch += 1
        done_training = self.global_step >= config["total_timesteps"]
        if done_training:
            self.stats = defaultdict(list)
            self.last_log_time = time.time()
            self.last_log_step = self.global_step
            profile.clear()
        self._record_instrumentation("train_done_cleanup", perf_counter() - bucket_started)

        if self.epoch % config["checkpoint_interval"] == 0 or done_training:
            checkpoint_started = perf_counter()
            self.save_checkpoint()
            self._record_instrumentation("train_checkpoint", perf_counter() - checkpoint_started)
            self.msg = f"Checkpoint saved at update {self.epoch}"

        self._record_instrumentation("train_total", perf_counter() - train_started)
        return None

    def save_checkpoint(self) -> str:
        started = perf_counter()
        if self._checkpointing_enabled:
            path = super().save_checkpoint()
            self._record_instrumentation("trainer_save_checkpoint", perf_counter() - started)
            return path
        path = os.path.join(self.config["data_dir"], f"{self.logger.run_id}.pt")
        self._record_instrumentation("trainer_save_checkpoint_skipped", perf_counter() - started)
        return path

    def close(self) -> str:
        started = perf_counter()
        if self._checkpointing_enabled:
            path = super().close()
            self._record_instrumentation("trainer_close", perf_counter() - started)
            return path
        self.vecenv.close()
        self.utilization.stop()
        path = os.path.join(self.config["data_dir"], f"{self.logger.run_id}.pt")
        self._record_instrumentation("trainer_close", perf_counter() - started)
        return path


class _NullProfile:
    def __iter__(self):
        return iter(())

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        return None

    def end(self) -> None:
        return None

    def clear(self) -> None:
        return None


class _NullUtilization:
    def stop(self) -> None:
        return None


@dataclass(frozen=True)
class TrainRunResult:
    config_id: str
    transport_mode: str
    checkpoint_path: str
    checkpoint_metadata_path: str
    global_step: int
    log_records: int
    puffer_logs: list[dict[str, float]]
    wandb_run_id: str
    run_manifest_path: str
    artifacts: list[dict[str, object]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def should_enable_dashboard(
    config: dict[str, Any],
    *,
    stdout_isatty: bool | None = None,
    stderr_isatty: bool | None = None,
) -> bool:
    requested = bool(dict(config.get("logging", {})).get("dashboard", False))
    if not requested:
        return False
    if stdout_isatty is None:
        stdout_isatty = sys.stdout.isatty()
    if stderr_isatty is None:
        stderr_isatty = sys.stderr.isatty()
    return bool(stdout_isatty and stderr_isatty)


def trace_stage(stage: str) -> None:
    trace_dir = os.environ.get("FC_RL_TRACE_DIR")
    if not trace_dir:
        return
    path = Path(trace_dir).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    trace_path = path / f"train-{os.getpid()}.trace"
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{stage}\n")


def run_smoke_training(
    *,
    config_path: str | Path | None = None,
    total_timesteps: int | None = None,
    data_dir: str | Path | None = None,
) -> TrainRunResult:
    trace_stage("run_smoke_training:start")
    bootstrap_config = load_bootstrap_config()
    trace_stage("run_smoke_training:bootstrap_config_loaded")
    config = load_smoke_train_config(config_path)
    trace_stage("run_smoke_training:config_loaded")
    transport_mode = str(
        dict(config.get("env", {})).get("subprocess_transport_mode", PIPE_PICKLE_TRANSPORT_MODE)
    )
    output_dir = build_train_output_dir(str(config["config_id"]), data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_stage("run_smoke_training:before_make_vecenv")
    vecenv = make_vecenv(config, backend="subprocess")
    trace_stage("run_smoke_training:vecenv_ready")
    policy = MultiDiscreteMLPPolicy.from_spaces(
        vecenv.single_observation_space,
        vecenv.single_action_space,
        hidden_size=int(config["policy"]["hidden_size"]),
    )
    trace_stage("run_smoke_training:policy_ready")
    puffer_train_config = build_puffer_train_config(
        config,
        data_dir=output_dir,
        total_timesteps=total_timesteps,
    )
    dashboard_enabled = should_enable_dashboard(config)
    trace_stage(f"run_smoke_training:dashboard_enabled={int(dashboard_enabled)}")
    logger = WandbRunLogger(
        config=bootstrap_config,
        run_kind="train",
        config_id=str(config["config_id"]),
        tags=(str(config["config_id"]), "smoke-train"),
    )
    trace_stage("run_smoke_training:logger_ready")
    trainer = ConfigurablePuffeRL(
        puffer_train_config,
        vecenv,
        policy,
        logger,
        dashboard_enabled=dashboard_enabled,
    )
    trace_stage("run_smoke_training:trainer_ready")

    try:
        while trainer.global_step < puffer_train_config["total_timesteps"]:
            trace_stage(f"run_smoke_training:loop_eval:{trainer.global_step}")
            trainer.evaluate()
            trace_stage(f"run_smoke_training:loop_train:{trainer.global_step}")
            trainer.train()

        trace_stage("run_smoke_training:final_eval")
        trainer.evaluate()
        trace_stage("run_smoke_training:mean_and_log")
        trainer.mean_and_log()
        trace_stage("run_smoke_training:close")
        checkpoint_path = Path(trainer.close())
        trace_stage("run_smoke_training:closed")
        trainer.logger.close(str(checkpoint_path))
    finally:
        if hasattr(trainer, "vecenv"):
            try:
                trace_stage("run_smoke_training:vecenv_close")
                trainer.vecenv.close()
            except Exception:
                pass

    trace_stage("run_smoke_training:metadata")
    metadata = build_checkpoint_metadata(
        train_config_id=str(config["config_id"]),
        policy_id=str(config["policy"]["id"]),
        reward_config_id=str(config["reward_config"]),
        curriculum_config_id=str(config["curriculum_config"]),
    )
    metadata_path = write_checkpoint_metadata(checkpoint_path, metadata)
    run_artifact_dir = output_dir / "runs" / str(logger.run_id)
    run_artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_artifact_dir / "run_manifest.json"
    checkpoint_record = logger.build_artifact_record(
        category="checkpoint",
        path=checkpoint_path,
    )
    checkpoint_metadata_record = logger.build_artifact_record(
        category="checkpoint_metadata",
        path=metadata_path,
    )
    manifest_record = logger.build_artifact_record(
        category="run_manifest",
        path=manifest_path,
    )
    manifest = build_train_run_manifest(
        bootstrap_config=bootstrap_config,
        config_id=str(config["config_id"]),
        run_id=str(logger.run_id),
        run_output_dir=run_artifact_dir,
        reward_config_id=str(config["reward_config"]),
        curriculum_config_id=str(config["curriculum_config"]),
        policy_id=str(config["policy"]["id"]),
        env_count=int(config["num_envs"]),
        bridge_mode="subprocess_isolated_jvm",
        dashboard_enabled=dashboard_enabled,
        wandb_tags=logger.effective_tags,
        checkpoint_metadata=metadata,
        checkpoint_path=checkpoint_path,
        checkpoint_metadata_path=metadata_path,
        artifacts=(checkpoint_record, checkpoint_metadata_record, manifest_record),
    )
    write_run_manifest(manifest_path, manifest)
    trace_stage("run_smoke_training:manifest_written")
    logger.update_config(manifest.to_dict())
    logger.close(str(checkpoint_path))
    logger.log_artifact(
        checkpoint_record,
        metadata={
            "run_kind": "train",
            "config_id": str(config["config_id"]),
            "checkpoint_format_id": metadata.checkpoint_format_id,
            "checkpoint_format_version": metadata.checkpoint_format_version,
        },
    )
    logger.log_artifact(
        checkpoint_metadata_record,
        metadata={
            "run_kind": "train",
            "config_id": str(config["config_id"]),
            "artifact_category": "checkpoint_metadata",
        },
    )
    logger.log_artifact(
        manifest_record,
        metadata={
            "run_kind": "train",
            "config_id": str(config["config_id"]),
            "artifact_category": "run_manifest",
        },
    )
    trace_stage("run_smoke_training:artifacts_logged")
    logger.finish()
    trace_stage("run_smoke_training:logger_finished")
    puffer_logs = [record.payload for record in logger.records]
    trace_stage("run_smoke_training:return")
    return TrainRunResult(
        config_id=str(config["config_id"]),
        transport_mode=transport_mode,
        checkpoint_path=str(checkpoint_path),
        checkpoint_metadata_path=str(metadata_path),
        global_step=int(trainer.global_step),
        log_records=len(logger.records),
        puffer_logs=puffer_logs,
        wandb_run_id=str(logger.run_id),
        run_manifest_path=str(manifest_path),
        artifacts=[record.to_dict() for record in logger.artifact_records],
    )


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    return run_replay_eval(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        max_steps=max_steps,
    )
