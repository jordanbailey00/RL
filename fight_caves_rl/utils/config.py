from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Mapping

from fight_caves_rl.manifests.versions import (
    PUFFERLIB_BASELINE_DISTRIBUTION,
    PUFFERLIB_BASELINE_VERSION,
)
from fight_caves_rl.utils.paths import repo_root, workspace_root


@dataclass(frozen=True)
class BootstrapConfig:
    rl_repo: Path
    sim_repo: Path
    rsps_repo: Path
    python_baseline: str
    pufferlib_distribution: str
    pufferlib_version: str
    wandb_project: str
    wandb_entity: str | None
    wandb_group: str | None
    wandb_mode: str
    wandb_resume: str
    wandb_run_prefix: str
    wandb_tags: tuple[str, ...]
    wandb_dir: Path
    wandb_data_dir: Path
    wandb_cache_dir: Path


def _optional_env(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_csv_env(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def load_bootstrap_config(env: Mapping[str, str] | None = None) -> BootstrapConfig:
    env_map = env or environ
    rl_root = repo_root()
    workspace = workspace_root()
    artifacts_root = rl_root / "artifacts"
    return BootstrapConfig(
        rl_repo=Path(env_map.get("RL_REPO", str(rl_root))),
        sim_repo=Path(env_map.get("FIGHT_CAVES_RL_REPO", str(workspace / "fight-caves-RL"))),
        rsps_repo=Path(env_map.get("RSPS_REPO", str(workspace / "RSPS"))),
        python_baseline=env_map.get("PYTHON_BASELINE", "3.11"),
        pufferlib_distribution=env_map.get(
            "PUFFERLIB_DISTRIBUTION",
            PUFFERLIB_BASELINE_DISTRIBUTION,
        ),
        pufferlib_version=env_map.get("PUFFERLIB_VERSION", PUFFERLIB_BASELINE_VERSION),
        wandb_project=env_map.get("WANDB_PROJECT", "fight-caves-rl"),
        wandb_entity=_optional_env(env_map.get("WANDB_ENTITY")),
        wandb_group=_optional_env(env_map.get("WANDB_GROUP", "smoke")),
        wandb_mode=env_map.get("WANDB_MODE", "offline"),
        wandb_resume=env_map.get("WANDB_RESUME", "allow"),
        wandb_run_prefix=env_map.get("WANDB_RUN_PREFIX", "fc-rl"),
        wandb_tags=_parse_csv_env(env_map.get("WANDB_TAGS", "fight-caves,rl")),
        wandb_dir=Path(env_map.get("WANDB_DIR", str(artifacts_root / "wandb"))),
        wandb_data_dir=Path(env_map.get("WANDB_DATA_DIR", str(artifacts_root / "wandb-data"))),
        wandb_cache_dir=Path(
            env_map.get("WANDB_CACHE_DIR", str(artifacts_root / "wandb-cache"))
        ),
    )
