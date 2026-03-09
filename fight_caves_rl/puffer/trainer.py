from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path
import sys
from typing import Any

import pufferlib.pufferl
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


class ConfigurablePuffeRL(pufferlib.pufferl.PuffeRL):
    """Thin PuffeRL wrapper that keeps dashboard rendering opt-in and TTY-bound."""

    def __init__(self, *args: Any, dashboard_enabled: bool = True, **kwargs: Any) -> None:
        self._dashboard_enabled = bool(dashboard_enabled)
        super().__init__(*args, **kwargs)

    def print_dashboard(self, *args: Any, **kwargs: Any) -> None:
        if not self._dashboard_enabled:
            return
        super().print_dashboard(*args, **kwargs)


@dataclass(frozen=True)
class TrainRunResult:
    config_id: str
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
