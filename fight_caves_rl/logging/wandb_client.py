from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
import time
from typing import Any, Iterable, Mapping
from uuid import uuid4

from fight_caves_rl.logging.artifact_naming import ArtifactRecord, build_artifact_record
from fight_caves_rl.logging.metrics import coerce_numeric_metrics, namespace_metrics
from fight_caves_rl.utils.config import BootstrapConfig


@dataclass(frozen=True)
class LoggedMetricRecord:
    step: int
    payload: dict[str, float]


@dataclass
class WandbRunLogger:
    config: BootstrapConfig
    run_kind: str
    config_id: str
    run_id: str | None = None
    tags: tuple[str, ...] = ()
    records: list[LoggedMetricRecord] = field(default_factory=list)
    artifact_records: list[ArtifactRecord] = field(default_factory=list)
    closed_model_path: str | None = None
    _wandb: Any | None = field(init=False, default=None, repr=False)
    _run: Any | None = field(init=False, default=None, repr=False)
    _finished: bool = field(init=False, default=False, repr=False)
    effective_tags: tuple[str, ...] = field(init=False, default=(), repr=False)

    def __post_init__(self) -> None:
        if self.run_id is None:
            unique_suffix = uuid4().hex[:12]
            timestamp = int(time.time())
            self.run_id = f"{self.config.wandb_run_prefix}-{self.run_kind}-{timestamp}-{unique_suffix}"
        self.effective_tags = tuple(dict.fromkeys((*self.config.wandb_tags, *self.tags)))
        self._prepare_local_dirs()
        if self.config.wandb_mode != "disabled":
            self._initialize_wandb()

    def _prepare_local_dirs(self) -> None:
        self.config.wandb_dir.mkdir(parents=True, exist_ok=True)
        self.config.wandb_data_dir.mkdir(parents=True, exist_ok=True)
        self.config.wandb_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WANDB_DIR", str(self.config.wandb_dir))
        os.environ.setdefault("WANDB_DATA_DIR", str(self.config.wandb_data_dir))
        os.environ.setdefault("WANDB_CACHE_DIR", str(self.config.wandb_cache_dir))

    def _initialize_wandb(self) -> None:
        import wandb

        self._wandb = wandb
        self._run = wandb.init(
            id=self.run_id,
            project=self.config.wandb_project,
            entity=self.config.wandb_entity,
            group=self.config.wandb_group,
            job_type=self.run_kind,
            name=f"{self.run_kind}-{self.config_id}-{str(self.run_id).split('-')[-1]}",
            allow_val_change=True,
            save_code=False,
            resume=self.config.wandb_resume,
            config={
                "config_id": self.config_id,
                "run_kind": self.run_kind,
            },
            tags=list(self.effective_tags),
            mode=self.config.wandb_mode,
            dir=str(self.config.wandb_dir),
            settings=wandb.Settings(
                console="off",
                quiet=True,
                silent=True,
                x_disable_stats=True,
                x_disable_machine_info=True,
            ),
        )

    def update_config(self, payload: Mapping[str, Any]) -> None:
        if self._run is not None:
            self._run.config.update(dict(payload), allow_val_change=True)

    def log(self, logs: Mapping[str, Any], step: int) -> None:
        payload = namespace_metrics("train", logs)
        self.records.append(LoggedMetricRecord(step=int(step), payload=payload))
        if self._run is not None and payload:
            self._run.log(payload, step=int(step))
            self._emit_benchmark_log_bursts(payload, step=int(step))

    def log_metrics(
        self,
        payload: Mapping[str, Any],
        *,
        step: int,
        namespace: str | None = None,
    ) -> dict[str, float]:
        normalized = (
            namespace_metrics(namespace, payload)
            if namespace is not None
            else coerce_numeric_metrics(payload)
        )
        self.records.append(LoggedMetricRecord(step=int(step), payload=normalized))
        if self._run is not None and normalized:
            self._run.log(normalized, step=int(step))
        return normalized

    def build_artifact_record(self, *, category: str, path: str | Path) -> ArtifactRecord:
        return build_artifact_record(
            run_kind=self.run_kind,
            config_id=self.config_id,
            run_id=str(self.run_id),
            category=category,
            path=path,
        )

    def log_artifact(
        self,
        record: ArtifactRecord,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.artifact_records.append(record)
        if self._run is None:
            return

        artifact = self._wandb.Artifact(
            record.name,
            type=record.artifact_type,
            metadata=dict(metadata or {}),
        )
        artifact.add_file(record.path, name=Path(record.path).name)
        self._run.log_artifact(artifact)

    def close(self, model_path: str | None = None) -> None:
        if model_path is not None:
            self.closed_model_path = model_path

    def finish(self) -> None:
        if self._finished:
            return
        if self._wandb is not None:
            self._wandb.finish()
        self._finished = True

    def _emit_benchmark_log_bursts(self, payload: Mapping[str, Any], *, step: int) -> None:
        extra_bursts = int(os.environ.get("FC_RL_BENCHMARK_EXTRA_LOG_BURSTS", "0") or "0")
        if self._run is None or extra_bursts <= 0:
            return
        numeric_payload = coerce_numeric_metrics(payload)
        if not numeric_payload:
            return
        for burst_index in range(extra_bursts):
            burst_payload = {
                f"benchmark_logging/repeat_{burst_index}/{key}": value
                for key, value in numeric_payload.items()
            }
            self.records.append(LoggedMetricRecord(step=int(step), payload=burst_payload))
            self._run.log(burst_payload, step=int(step))

    def metrics_to_dicts(self) -> list[dict[str, Any]]:
        return [asdict(record) for record in self.records]
