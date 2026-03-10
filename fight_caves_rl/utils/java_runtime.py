from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from os import environ
from pathlib import Path
import shutil

from fight_caves_rl.utils.paths import repo_root, workspace_root


@dataclass(frozen=True)
class JavaRuntimePaths:
    java_home: Path
    java_executable: Path
    jvm_library: Path


@lru_cache(maxsize=1)
def resolve_java_runtime() -> JavaRuntimePaths | None:
    for candidate in _candidate_java_homes():
        runtime = _resolve_runtime_from_candidate(candidate)
        if runtime is not None:
            return runtime
    return None


def resolve_java_home() -> Path | None:
    runtime = resolve_java_runtime()
    if runtime is None:
        return None
    return runtime.java_home


def resolve_java_executable() -> Path | None:
    runtime = resolve_java_runtime()
    if runtime is None:
        return None
    return runtime.java_executable


def resolve_jvm_library_path() -> Path | None:
    runtime = resolve_java_runtime()
    if runtime is None:
        return None
    return runtime.jvm_library


def _candidate_java_homes() -> tuple[Path, ...]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(candidate: Path | None) -> None:
        if candidate is None:
            return
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    for env_name in ("FC_RL_JAVA_HOME", "JAVA_HOME", "JDK_HOME"):
        value = environ.get(env_name)
        if value:
            add(Path(value))

    for candidate in _iter_workspace_toolchain_candidates():
        add(candidate)

    which_java = shutil.which("java")
    if which_java:
        add(Path(which_java).resolve().parent.parent)

    for candidate in _iter_system_java_candidates():
        add(candidate)

    return tuple(candidates)


def _iter_workspace_toolchain_candidates() -> tuple[Path, ...]:
    roots = (
        workspace_root() / ".workspace-tools",
        repo_root() / "artifacts" / "toolchains",
    )
    candidates: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for name in ("jdk-21", "jdk", "default-java"):
            candidate = root / name
            if candidate.exists():
                candidates.append(candidate)
        for pattern in ("jdk-*", "jre-*"):
            candidates.extend(sorted(root.glob(pattern)))
    return tuple(candidates)


def _iter_system_java_candidates() -> tuple[Path, ...]:
    root = Path("/usr/lib/jvm")
    if not root.is_dir():
        return ()
    candidates: list[Path] = []
    for name in ("default-java",):
        candidate = root / name
        if candidate.exists():
            candidates.append(candidate)
    candidates.extend(sorted(candidate for candidate in root.iterdir() if candidate.is_dir()))
    return tuple(candidates)


def _resolve_runtime_from_candidate(candidate: Path) -> JavaRuntimePaths | None:
    if candidate.is_file():
        if candidate.name == "java":
            candidate = candidate.parent.parent
        elif candidate.name == "libjvm.so":
            candidate = candidate.parents[2]
        else:
            return None

    java_executable = _select_first_existing(
        (
            candidate / "bin" / "java",
            candidate / "jre" / "bin" / "java",
        )
    )
    jvm_library = _select_first_existing(
        (
            candidate / "lib" / "server" / "libjvm.so",
            candidate / "lib" / "amd64" / "server" / "libjvm.so",
            candidate / "jre" / "lib" / "server" / "libjvm.so",
            candidate / "jre" / "lib" / "amd64" / "server" / "libjvm.so",
        )
    )
    if java_executable is None or jvm_library is None:
        return None
    return JavaRuntimePaths(
        java_home=candidate.resolve(),
        java_executable=java_executable.resolve(),
        jvm_library=jvm_library.resolve(),
    )


def _select_first_existing(candidates: tuple[Path, ...]) -> Path | None:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
