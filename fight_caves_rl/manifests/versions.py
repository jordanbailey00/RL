from dataclasses import dataclass
from importlib import import_module, metadata
from typing import Callable

PUFFERLIB_BASELINE_DISTRIBUTION = "pufferlib-core"
PUFFERLIB_BASELINE_VERSION = "3.0.17"
PUFFERLIB_IMPORT_NAME = "pufferlib"
PUFFERLIB_LEGACY_DISTRIBUTION = "pufferlib"


@dataclass(frozen=True)
class PufferLibRuntimeVersion:
    distribution_name: str
    distribution_version: str
    import_name: str
    import_version: str | None


def resolve_pufferlib_runtime_version(
    version_resolver: Callable[[str], str] = metadata.version,
    module_loader: Callable[[str], object] = import_module,
) -> PufferLibRuntimeVersion:
    distribution_name = PUFFERLIB_BASELINE_DISTRIBUTION
    distribution_version = PUFFERLIB_BASELINE_VERSION

    for candidate in (PUFFERLIB_BASELINE_DISTRIBUTION, PUFFERLIB_LEGACY_DISTRIBUTION):
        try:
            distribution_version = version_resolver(candidate)
            distribution_name = candidate
            break
        except metadata.PackageNotFoundError:
            continue

    try:
        module = module_loader(PUFFERLIB_IMPORT_NAME)
        raw_import_version = getattr(module, "__version__", None)
        import_version = None if raw_import_version is None else str(raw_import_version)
    except ModuleNotFoundError:
        import_version = None

    return PufferLibRuntimeVersion(
        distribution_name=distribution_name,
        distribution_version=distribution_version,
        import_name=PUFFERLIB_IMPORT_NAME,
        import_version=import_version,
    )
