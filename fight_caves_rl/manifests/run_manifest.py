from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from platform import python_version

from fight_caves_rl.utils.config import BootstrapConfig


@dataclass(frozen=True)
class BootstrapRunManifest:
    created_at: str
    rl_repo: str
    sim_repo: str
    rsps_repo: str
    python_version: str
    python_baseline: str
    pufferlib_distribution: str
    pufferlib_version: str
    wandb_project: str
    wandb_mode: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_bootstrap_manifest(config: BootstrapConfig) -> BootstrapRunManifest:
    return BootstrapRunManifest(
        created_at=datetime.now(UTC).isoformat(),
        rl_repo=str(Path(config.rl_repo)),
        sim_repo=str(Path(config.sim_repo)),
        rsps_repo=str(Path(config.rsps_repo)),
        python_version=python_version(),
        python_baseline=config.python_baseline,
        pufferlib_distribution=config.pufferlib_distribution,
        pufferlib_version=config.pufferlib_version,
        wandb_project=config.wandb_project,
        wandb_mode=config.wandb_mode,
    )
