from __future__ import annotations

from dataclasses import dataclass, field
from multiprocessing.connection import Connection
import multiprocessing
import traceback
from typing import Any, Mapping

import numpy as np
import pufferlib
import pufferlib.vector

from fight_caves_rl.bridge.contracts import HeadlessBootstrapConfig
from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.curriculum.registry import build_curriculum
from fight_caves_rl.envs.puffer_encoding import (
    build_policy_action_space,
    build_policy_observation_space,
)
from fight_caves_rl.envs.shared_memory_transport import (
    PIPE_PICKLE_TRANSPORT_MODE,
    SHARED_MEMORY_TRANSPORT_MODE,
    SUBPROCESS_TRANSPORT_MODES,
    SharedMemoryTransportParent,
    SharedMemoryTransportWorker,
)
from fight_caves_rl.envs.vector_env import HeadlessBatchVecEnv, HeadlessBatchVecEnvConfig
from fight_caves_rl.rewards.registry import resolve_reward_fn


@dataclass(frozen=True)
class SubprocessHeadlessBatchVecEnvConfig:
    env_count: int
    reward_config_id: str
    curriculum_config_id: str
    transport_mode: str = PIPE_PICKLE_TRANSPORT_MODE
    account_name_prefix: str = "rl_vecenv"
    start_wave: int = 1
    ammo: int = 1000
    prayer_potions: int = 8
    sharks: int = 20
    tick_cap: int = 20_000
    include_future_leakage: bool = False
    bootstrap: HeadlessBootstrapConfig = field(default_factory=HeadlessBootstrapConfig)


@dataclass(frozen=True)
class _WorkerError:
    error_type: str
    message: str
    traceback_text: str


class SubprocessHeadlessBatchVecEnv:
    reset = pufferlib.vector.reset
    step = pufferlib.vector.step

    def __init__(self, config: SubprocessHeadlessBatchVecEnvConfig) -> None:
        if int(config.env_count) <= 0:
            raise ValueError(f"env_count must be > 0, got {config.env_count}.")
        self.config = config
        self.driver_env = self
        self.agents_per_batch = int(config.env_count)
        self.num_agents = self.agents_per_batch
        self.single_observation_space = build_policy_observation_space()
        self.single_action_space = build_policy_action_space()
        self.transport_mode = str(config.transport_mode)
        if self.transport_mode not in SUBPROCESS_TRANSPORT_MODES:
            raise ValueError(
                "Unsupported subprocess transport mode: "
                f"{self.transport_mode!r}. Expected one of {SUBPROCESS_TRANSPORT_MODES!r}."
            )
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
        self.initialized = False
        self.flag = pufferlib.vector.RESET
        self.infos: list[dict[str, Any]] = []
        self._closed = False
        self._ctx = multiprocessing.get_context("spawn")
        self._transport_parent: SharedMemoryTransportParent | None = None
        if self.transport_mode == SHARED_MEMORY_TRANSPORT_MODE:
            self._transport_parent = SharedMemoryTransportParent(
                env_count=self.agents_per_batch,
                action_dim=len(self.single_action_space.nvec),
                observation_dim=int(self.single_observation_space.shape[0]),
            )
        parent_conn, child_conn = self._ctx.Pipe()
        self._conn = parent_conn
        self._process = self._ctx.Process(
            target=_subprocess_vecenv_worker,
            args=(child_conn, _config_to_payload(config, transport_parent=self._transport_parent)),
            daemon=True,
        )
        self._process.start()
        child_conn.close()

    @property
    def num_envs(self) -> int:
        return self.agents_per_batch

    def async_reset(self, seed: int | None = None) -> None:
        self.flag = pufferlib.vector.RECV
        self._send_command("reset", {"seed": None if seed is None else int(seed)})

    def send(self, actions: np.ndarray) -> None:
        if not actions.flags.contiguous:
            actions = np.ascontiguousarray(actions)
        checked = pufferlib.vector.send_precheck(self, actions)
        if self._transport_parent is not None:
            self._transport_parent.write_actions(np.asarray(checked))
            self._send_command("step", {})
            return
        self._send_command("step", {"actions": np.array(checked, copy=True)})

    def recv(self):
        pufferlib.vector.recv_precheck(self)
        payload = self._recv_payload()
        if payload.get("transport_mode") == SHARED_MEMORY_TRANSPORT_MODE:
            if self._transport_parent is None:
                raise BridgeError(
                    "Shared-memory subprocess vecenv response arrived without a parent transport."
                )
            payload = self._transport_parent.materialize_transition(payload)
        return (
            payload["observations"],
            payload["rewards"],
            payload["terminals"],
            payload["truncations"],
            payload["teacher_actions"],
            payload["infos"],
            payload["agent_ids"],
            payload["masks"],
        )

    def notify(self) -> None:
        return None

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._process.is_alive():
                self._send_command("close", None)
                if self._conn.poll(2.0):
                    self._recv_payload()
        except Exception:
            pass
        finally:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=2.0)
            self._conn.close()
            if self._transport_parent is not None:
                self._transport_parent.close(unlink=True)
            self._closed = True

    def _send_command(self, command: str, payload: Mapping[str, Any] | None) -> None:
        if not self._process.is_alive():
            raise BridgeError(
                "Subprocess vecenv worker is not running. The previous command likely crashed the worker."
            )
        self._conn.send((command, dict(payload or {})))

    def _recv_payload(self) -> dict[str, Any]:
        try:
            status, payload = self._conn.recv()
        except EOFError as exc:
            raise BridgeError(
                "Subprocess vecenv worker exited unexpectedly while waiting for a response."
            ) from exc
        if status == "ok":
            return payload
        error = _WorkerError(**payload)
        raise BridgeError(
            "Subprocess vecenv worker failed with "
            f"{error.error_type}: {error.message}\n{error.traceback_text}"
        )


def _subprocess_vecenv_worker(conn: Connection, payload: dict[str, Any]) -> None:
    vecenv: HeadlessBatchVecEnv | None = None
    transport_worker: SharedMemoryTransportWorker | None = None
    try:
        vecenv = _build_worker_vecenv(payload)
        if payload["transport_mode"] == SHARED_MEMORY_TRANSPORT_MODE:
            transport_worker = SharedMemoryTransportWorker.attach(payload["transport"])
        while True:
            command, data = conn.recv()
            if command == "reset":
                vecenv.async_reset(seed=data.get("seed"))
                transition = vecenv.recv()
                conn.send(("ok", _serialize_transition(transition, transport_worker=transport_worker)))
                continue
            if command == "step":
                if transport_worker is None:
                    actions = np.asarray(data["actions"])
                else:
                    actions = transport_worker.read_actions()
                vecenv.send(actions)
                transition = vecenv.recv()
                conn.send(("ok", _serialize_transition(transition, transport_worker=transport_worker)))
                continue
            if command == "close":
                vecenv.close()
                conn.send(("ok", {"closed": True}))
                return
            raise ValueError(f"Unsupported subprocess vecenv command: {command!r}")
    except BaseException as exc:
        try:
            conn.send(
                (
                    "error",
                    {
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                        "traceback_text": traceback.format_exc(),
                    },
                )
            )
        except Exception:
            pass
        raise
    finally:
        if transport_worker is not None:
            try:
                transport_worker.close()
            except Exception:
                pass
        if vecenv is not None:
            try:
                vecenv.close()
            except Exception:
                pass
        conn.close()


def _build_worker_vecenv(payload: dict[str, Any]) -> HeadlessBatchVecEnv:
    curriculum = build_curriculum(str(payload["curriculum_config_id"]))
    return HeadlessBatchVecEnv(
        HeadlessBatchVecEnvConfig(
            env_count=int(payload["env_count"]),
            account_name_prefix=str(payload["account_name_prefix"]),
            start_wave=int(payload["start_wave"]),
            ammo=int(payload["ammo"]),
            prayer_potions=int(payload["prayer_potions"]),
            sharks=int(payload["sharks"]),
            tick_cap=int(payload["tick_cap"]),
            include_future_leakage=bool(payload["include_future_leakage"]),
            bootstrap=HeadlessBootstrapConfig(
                load_content_scripts=bool(payload["bootstrap"]["load_content_scripts"]),
                start_world=bool(payload["bootstrap"]["start_world"]),
                install_shutdown_hook=bool(payload["bootstrap"]["install_shutdown_hook"]),
                settings_overrides=dict(payload["bootstrap"]["settings_overrides"]),
            ),
            reset_options_provider=curriculum.reset_overrides,
        ),
        reward_fn=resolve_reward_fn(str(payload["reward_config_id"])),
    )


def _serialize_transition(
    transition: tuple[Any, ...],
    *,
    transport_worker: SharedMemoryTransportWorker | None,
) -> dict[str, Any]:
    if transport_worker is not None:
        return transport_worker.publish_transition(transition)
    observations, rewards, terminals, truncations, teacher_actions, infos, agent_ids, masks = transition
    return {
        "observations": np.array(observations, copy=True),
        "rewards": np.array(rewards, copy=True),
        "terminals": np.array(terminals, copy=True),
        "truncations": np.array(truncations, copy=True),
        "teacher_actions": np.array(teacher_actions, copy=True),
        "infos": list(infos),
        "agent_ids": np.array(agent_ids, copy=True),
        "masks": np.array(masks, copy=True),
    }


def _config_to_payload(
    config: SubprocessHeadlessBatchVecEnvConfig,
    *,
    transport_parent: SharedMemoryTransportParent | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "env_count": int(config.env_count),
        "reward_config_id": str(config.reward_config_id),
        "curriculum_config_id": str(config.curriculum_config_id),
        "transport_mode": str(config.transport_mode),
        "account_name_prefix": str(config.account_name_prefix),
        "start_wave": int(config.start_wave),
        "ammo": int(config.ammo),
        "prayer_potions": int(config.prayer_potions),
        "sharks": int(config.sharks),
        "tick_cap": int(config.tick_cap),
        "include_future_leakage": bool(config.include_future_leakage),
        "bootstrap": {
            "load_content_scripts": bool(config.bootstrap.load_content_scripts),
            "start_world": bool(config.bootstrap.start_world),
            "install_shutdown_hook": bool(config.bootstrap.install_shutdown_hook),
            "settings_overrides": dict(config.bootstrap.settings_overrides),
        },
    }
    if str(config.transport_mode) == SHARED_MEMORY_TRANSPORT_MODE:
        if transport_parent is None:
            raise ValueError(
                "Shared-memory subprocess transport requires a parent-side shared-memory transport."
            )
        payload["transport"] = transport_parent.spec().to_payload()
    return payload
