from dataclasses import dataclass
from os import environ
from pathlib import Path
from typing import Mapping

from fight_caves_rl.utils.paths import repo_root, workspace_root


@dataclass(frozen=True)
class BootstrapConfig:
    rl_repo: Path
    sim_repo: Path
    rsps_repo: Path
    python_baseline: str
    pufferlib_version: str
    wandb_project: str
    wandb_mode: str


def load_bootstrap_config(env: Mapping[str, str] | None = None) -> BootstrapConfig:
    env_map = env or environ
    rl_root = repo_root()
    workspace = workspace_root()
    return BootstrapConfig(
        rl_repo=Path(env_map.get("RL_REPO", str(rl_root))),
        sim_repo=Path(env_map.get("FIGHT_CAVES_RL_REPO", str(workspace / "fight-caves-RL"))),
        rsps_repo=Path(env_map.get("RSPS_REPO", str(workspace / "RSPS"))),
        python_baseline=env_map.get("PYTHON_BASELINE", "3.11"),
        pufferlib_version=env_map.get("PUFFERLIB_VERSION", "3.0.0"),
        wandb_project=env_map.get("WANDB_PROJECT", "fight-caves-rl"),
        wandb_mode=env_map.get("WANDB_MODE", "offline"),
    )

