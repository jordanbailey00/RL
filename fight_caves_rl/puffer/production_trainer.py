from __future__ import annotations

from collections import defaultdict
from contextlib import nullcontext
import time
from time import perf_counter
from typing import Any
import warnings

import numpy as np
import pufferlib
import pufferlib.pufferl
import torch


ProductionTrainerInstrumentationSnapshot = dict[str, dict[str, float | int]]


class _ProductionTrainerInstrumentation:
    def __init__(self) -> None:
        self._seconds: defaultdict[str, float] = defaultdict(float)
        self._calls: defaultdict[str, int] = defaultdict(int)

    def record(self, bucket: str, seconds: float) -> None:
        self._seconds[str(bucket)] += max(0.0, float(seconds))
        self._calls[str(bucket)] += 1

    def snapshot(self) -> ProductionTrainerInstrumentationSnapshot:
        return {
            name: {
                "seconds": float(self._seconds[name]),
                "calls": int(self._calls[name]),
            }
            for name in sorted(self._seconds)
        }


class PrototypeProductionTrainer:
    """Project-owned synchronous trainer path for the Phase 2 prototype batch."""

    def __init__(
        self,
        config: dict[str, Any],
        vecenv: Any,
        policy: torch.nn.Module,
    ) -> None:
        if bool(config.get("use_rnn", False)):
            raise ValueError("prototype_sync_v1 does not support recurrent policies.")

        torch.backends.cudnn.deterministic = bool(config["torch_deterministic"])
        torch.backends.cudnn.benchmark = True

        seed = int(config["seed"])
        vecenv.async_reset(seed)

        obs_space = vecenv.single_observation_space
        action_space = vecenv.single_action_space
        total_agents = int(vecenv.num_agents)
        self.total_agents = total_agents

        if config["batch_size"] == "auto" and config["bptt_horizon"] == "auto":
            raise pufferlib.APIUsageError("Must specify batch_size or bptt_horizon")
        if config["batch_size"] == "auto":
            config["batch_size"] = total_agents * int(config["bptt_horizon"])
        elif config["bptt_horizon"] == "auto":
            config["bptt_horizon"] = int(config["batch_size"]) // total_agents

        batch_size = int(config["batch_size"])
        horizon = int(config["bptt_horizon"])
        segments = batch_size // horizon
        self.segments = segments
        if total_agents > segments:
            raise pufferlib.APIUsageError(
                f"Total agents {total_agents} must be <= segments {segments}"
            )

        device = config["device"]
        self._device = device
        self._cpu_offload = bool(config["cpu_offload"])
        obs_device = "cpu" if self._cpu_offload else device
        self.observations = torch.zeros(
            segments,
            horizon,
            *obs_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[obs_space.dtype],
            pin_memory=device == "cuda" and self._cpu_offload,
            device=obs_device,
        )
        self.actions = torch.zeros(
            segments,
            horizon,
            *action_space.shape,
            dtype=pufferlib.pytorch.numpy_to_torch_dtype_dict[action_space.dtype],
            device=device,
        )
        self.values = torch.zeros(segments, horizon, device=device)
        self.logprobs = torch.zeros(segments, horizon, device=device)
        self.rewards = torch.zeros(segments, horizon, device=device)
        self.terminals = torch.zeros(segments, horizon, device=device)
        self.truncations = torch.zeros(segments, horizon, device=device)
        self.ratio = torch.ones(segments, horizon, device=device)
        self.ep_lengths = torch.zeros(total_agents, device=device, dtype=torch.int32)
        self.ep_indices = torch.arange(total_agents, device=device, dtype=torch.int32)
        self.free_idx = total_agents

        minibatch_size = min(int(config["minibatch_size"]), int(config["max_minibatch_size"]))
        if batch_size < int(config["minibatch_size"]):
            raise pufferlib.APIUsageError(
                f"batch_size {batch_size} must be >= minibatch_size {config['minibatch_size']}"
            )
        self.minibatch_size = minibatch_size
        self.accumulate_minibatches = max(
            1, int(config["minibatch_size"]) // int(config["max_minibatch_size"])
        )
        self.total_minibatches = int(int(config["update_epochs"]) * batch_size / minibatch_size)
        self.minibatch_segments = minibatch_size // horizon
        if self.minibatch_segments * horizon != minibatch_size:
            raise pufferlib.APIUsageError(
                f"minibatch_size {minibatch_size} must be divisible by bptt_horizon {horizon}"
            )

        self.uncompiled_policy = policy
        self.policy = policy
        if bool(config["compile"]):
            self.policy = torch.compile(
                policy,
                mode=str(config["compile_mode"]),
                fullgraph=bool(config["compile_fullgraph"]),
            )

        self.optimizer = _build_optimizer(self.policy, config)
        epochs = max(1, int(config["total_timesteps"]) // batch_size)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        self.total_epochs = epochs

        precision = str(config["precision"])
        device_type = device if isinstance(device, str) else device.type
        if precision not in ("float32", "bfloat16"):
            raise pufferlib.APIUsageError(
                f"Invalid precision: {precision}: use float32 or bfloat16"
            )
        if device_type == "cpu" and precision == "float32":
            self.amp_context = nullcontext()
        else:
            self.amp_context = torch.amp.autocast(
                device_type=device_type,
                dtype=getattr(torch, precision),
            )

        self.config = config
        self.vecenv = vecenv
        self.epoch = 0
        self.global_step = 0
        self._instrumentation = _ProductionTrainerInstrumentation()

    def instrumentation_snapshot(self) -> ProductionTrainerInstrumentationSnapshot:
        return self._instrumentation.snapshot()

    def collect_rollout(self) -> None:
        config = self.config
        device = self._device
        total_started = perf_counter()
        full_rows = 0
        while full_rows < self.segments:
            bucket_started = perf_counter()
            observations, rewards, terminals, truncations, _, _, env_id, mask = self.vecenv.recv()
            self._record("rollout_env_recv", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            env_slice = _env_id_to_slice(env_id)
            self.global_step += int(np.asarray(mask).sum())
            self._record("rollout_env_index", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            observation_tensor = torch.as_tensor(observations)
            observation_device = observation_tensor.to(device)
            reward_tensor = torch.as_tensor(rewards).to(device)
            terminal_tensor = torch.as_tensor(terminals).to(device)
            truncation_tensor = torch.as_tensor(truncations).to(device)
            self._record("rollout_tensor_copy", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            with torch.no_grad(), self.amp_context:
                state = {
                    "reward": reward_tensor,
                    "done": terminal_tensor,
                    "env_id": env_slice,
                    "mask": mask,
                }
                logits, value = self.policy.forward_eval(observation_device, state)
                action, logprob, _ = _sample_policy_outputs(logits)
                reward_tensor = torch.clamp(reward_tensor, -1, 1)
            self._record("rollout_policy_forward", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            with torch.no_grad():
                rollout_index = int(self.ep_lengths[env_slice.start].item())
                batch_rows = slice(
                    int(self.ep_indices[env_slice.start].item()),
                    1 + int(self.ep_indices[env_slice.stop - 1].item()),
                )

                if self._cpu_offload:
                    self.observations[batch_rows, rollout_index] = observation_tensor
                else:
                    self.observations[batch_rows, rollout_index] = observation_device
                self.actions[batch_rows, rollout_index] = action
                self.logprobs[batch_rows, rollout_index] = logprob
                self.rewards[batch_rows, rollout_index] = reward_tensor
                self.terminals[batch_rows, rollout_index] = terminal_tensor.float()
                self.truncations[batch_rows, rollout_index] = truncation_tensor.float()
                self.values[batch_rows, rollout_index] = value.flatten()

                self.ep_lengths[env_slice] += 1
                if rollout_index + 1 >= int(config["bptt_horizon"]):
                    num_full = env_slice.stop - env_slice.start
                    self.ep_indices[env_slice] = self.free_idx + torch.arange(
                        num_full,
                        device=device,
                        dtype=torch.int32,
                    )
                    self.ep_lengths[env_slice] = 0
                    self.free_idx += num_full
                    full_rows += num_full
            self._record("rollout_buffer_write", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            with torch.no_grad():
                action_numpy = action.cpu().numpy()
                if isinstance(logits, torch.distributions.Normal):
                    action_numpy = np.clip(
                        action_numpy,
                        self.vecenv.action_space.low,
                        self.vecenv.action_space.high,
                    )
            self._record("rollout_action_to_numpy", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            self.vecenv.send(action_numpy)
            self._record("rollout_env_send", perf_counter() - bucket_started)

        bucket_started = perf_counter()
        self.free_idx = self.total_agents
        self.ep_indices = torch.arange(self.total_agents, device=device, dtype=torch.int32)
        self.ep_lengths.zero_()
        self._record("rollout_reset_state", perf_counter() - bucket_started)
        self._record("rollout_total", perf_counter() - total_started)

    def train_update(self) -> None:
        config = self.config
        device = self._device
        total_started = perf_counter()

        b0 = float(config["prio_beta0"])
        alpha = float(config["prio_alpha"])
        clip_coef = float(config["clip_coef"])
        vf_clip = float(config["vf_clip_coef"])
        anneal_beta = b0 + (1.0 - b0) * alpha * self.epoch / self.total_epochs

        self.ratio[:] = 1
        self.optimizer.zero_grad(set_to_none=True)

        bucket_started = perf_counter()
        advantages = torch.zeros(self.values.shape, device=device)
        advantages = pufferlib.pufferl.compute_puff_advantage(
            self.values,
            self.rewards,
            self.terminals,
            self.ratio,
            advantages,
            float(config["gamma"]),
            float(config["gae_lambda"]),
            float(config["vtrace_rho_clip"]),
            float(config["vtrace_c_clip"]),
        )
        self._record("update_advantage", perf_counter() - bucket_started)

        bucket_started = perf_counter()
        advantage_sum = advantages.abs().sum(axis=1)
        prio_weights = torch.nan_to_num(advantage_sum**alpha, 0, 0, 0)
        prio_probs = (prio_weights + 1e-6) / (prio_weights.sum() + 1e-6)
        self._record("update_priority_prepare", perf_counter() - bucket_started)

        for minibatch_index in range(self.total_minibatches):
            bucket_started = perf_counter()
            idx = torch.multinomial(prio_probs, self.minibatch_segments)
            mb_prio = (self.segments * prio_probs[idx, None]) ** -anneal_beta
            mb_obs = self.observations[idx]
            mb_actions = self.actions[idx]
            mb_logprobs = self.logprobs[idx]
            mb_values = self.values[idx]
            mb_returns = advantages[idx] + mb_values
            mb_advantages = advantages[idx]
            self._record("update_minibatch_prepare", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            if not bool(config["use_rnn"]):
                mb_obs = mb_obs.reshape(-1, *self.vecenv.single_observation_space.shape)
            state = {
                "action": mb_actions,
                "lstm_h": None,
                "lstm_c": None,
            }
            logits, newvalue = self.policy(mb_obs, state)
            _, newlogprob, entropy = _sample_policy_outputs(logits, action=mb_actions)
            self._record("update_policy_forward", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            newlogprob = newlogprob.reshape(mb_logprobs.shape)
            logratio = newlogprob - mb_logprobs
            ratio = logratio.exp()
            self.ratio[idx] = ratio.detach()

            normalized_advantages = mb_prio * (mb_advantages - mb_advantages.mean()) / (
                mb_advantages.std() + 1e-8
            )

            pg_loss1 = -normalized_advantages * ratio
            pg_loss2 = -normalized_advantages * torch.clamp(
                ratio, 1.0 - clip_coef, 1.0 + clip_coef
            )
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()

            newvalue = newvalue.view(mb_returns.shape)
            v_clipped = mb_values + torch.clamp(newvalue - mb_values, -vf_clip, vf_clip)
            v_loss_unclipped = (newvalue - mb_returns) ** 2
            v_loss_clipped = (v_clipped - mb_returns) ** 2
            v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

            entropy_loss = entropy.mean()
            loss = (
                pg_loss
                + float(config["vf_coef"]) * v_loss
                - float(config["ent_coef"]) * entropy_loss
            )

            self.values[idx] = newvalue.detach().float()
            self._record("update_loss_compute", perf_counter() - bucket_started)

            bucket_started = perf_counter()
            loss.backward()
            self._record("update_backward", perf_counter() - bucket_started)

            if (minibatch_index + 1) % self.accumulate_minibatches == 0:
                bucket_started = perf_counter()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), float(config["max_grad_norm"])
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._record("update_optimizer_step", perf_counter() - bucket_started)

        if self.total_minibatches % self.accumulate_minibatches != 0:
            bucket_started = perf_counter()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), float(config["max_grad_norm"])
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._record("update_optimizer_step", perf_counter() - bucket_started)

        if bool(config["anneal_lr"]):
            bucket_started = perf_counter()
            self.scheduler.step()
            self._record("update_scheduler", perf_counter() - bucket_started)

        self.epoch += 1
        self._record("update_total", perf_counter() - total_started)

    def close(self) -> None:
        started = perf_counter()
        self.vecenv.close()
        self._record("trainer_close", perf_counter() - started)

    def _record(self, bucket: str, seconds: float) -> None:
        self._instrumentation.record(bucket, seconds)


def _build_optimizer(policy: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(config["optimizer"])
    if optimizer_name == "adam":
        return torch.optim.Adam(
            policy.parameters(),
            lr=float(config["learning_rate"]),
            betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
            eps=float(config["adam_eps"]),
        )
    if optimizer_name == "muon":
        from heavyball import ForeachMuon

        warnings.filterwarnings(action="ignore", category=UserWarning, module=r"heavyball.*")
        import heavyball.utils

        heavyball.utils.compile_mode = (
            str(config["compile_mode"]) if bool(config["compile"]) else None
        )
        return ForeachMuon(
            policy.parameters(),
            lr=float(config["learning_rate"]),
            betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
            eps=float(config["adam_eps"]),
        )
    raise ValueError(f"Unknown optimizer: {optimizer_name}")


def _env_id_to_slice(env_id: np.ndarray | list[int] | tuple[int, ...]) -> slice:
    if len(env_id) == 0:
        raise ValueError("Prototype trainer received an empty env_id batch.")
    env_id_array = np.asarray(env_id)
    return slice(int(env_id_array[0]), int(env_id_array[-1]) + 1)


def _sample_policy_outputs(
    logits: Any,
    action: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(logits, torch.Tensor) or isinstance(logits, torch.distributions.Normal):
        return pufferlib.pytorch.sample_logits(logits, action=action)
    return _sample_multidiscrete_logits(logits, action=action)


def _sample_multidiscrete_logits(
    logits: tuple[torch.Tensor, ...],
    action: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not logits:
        raise ValueError("Prototype trainer received empty multi-discrete logits.")

    batch_size = int(logits[0].shape[0])
    action_view = None if action is None else action.view(batch_size, -1)
    sampled_actions: list[torch.Tensor] = []
    log_probs: list[torch.Tensor] = []
    entropies: list[torch.Tensor] = []

    for head_index, head_logits in enumerate(logits):
        head_log_probs = torch.log_softmax(head_logits, dim=-1)
        head_probs = head_log_probs.exp()
        if action_view is None:
            sample_weights = torch.nan_to_num(head_probs, 1e-8, 1e-8, 1e-8)
            head_action = torch.multinomial(
                sample_weights,
                1,
                replacement=True,
            ).squeeze(-1)
        else:
            head_action = action_view[:, head_index].long()
        sampled_actions.append(head_action.to(dtype=torch.int32))
        log_probs.append(
            head_log_probs.gather(-1, head_action.unsqueeze(-1)).squeeze(-1)
        )
        entropies.append(-(head_probs * head_log_probs).sum(dim=-1))

    actions = torch.stack(sampled_actions, dim=1)
    logprob = torch.stack(log_probs, dim=1).sum(dim=1)
    entropy = torch.stack(entropies, dim=1).sum(dim=1)
    return actions, logprob, entropy
