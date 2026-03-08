from importlib import metadata
from types import SimpleNamespace

from fight_caves_rl.manifests.versions import (
    PUFFERLIB_BASELINE_DISTRIBUTION,
    PUFFERLIB_BASELINE_VERSION,
    PUFFERLIB_IMPORT_NAME,
    resolve_pufferlib_runtime_version,
)


def test_resolve_pufferlib_runtime_version_prefers_core_distribution():
    versions = {
        "pufferlib-core": "3.0.17",
        "pufferlib": "3.0.0",
    }

    resolved = resolve_pufferlib_runtime_version(
        version_resolver=versions.__getitem__,
        module_loader=lambda _: SimpleNamespace(__version__="3.0.3"),
    )

    assert resolved.distribution_name == "pufferlib-core"
    assert resolved.distribution_version == "3.0.17"
    assert resolved.import_name == "pufferlib"
    assert resolved.import_version == "3.0.3"


def test_resolve_pufferlib_runtime_version_falls_back_to_repo_baseline():
    def missing_version(_: str) -> str:
        raise metadata.PackageNotFoundError

    def missing_module(_: str) -> object:
        raise ModuleNotFoundError

    resolved = resolve_pufferlib_runtime_version(
        version_resolver=missing_version,
        module_loader=missing_module,
    )

    assert resolved.distribution_name == PUFFERLIB_BASELINE_DISTRIBUTION
    assert resolved.distribution_version == PUFFERLIB_BASELINE_VERSION
    assert resolved.import_name == PUFFERLIB_IMPORT_NAME
    assert resolved.import_version is None
