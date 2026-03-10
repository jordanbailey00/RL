from __future__ import annotations

from pathlib import Path

from fight_caves_rl.utils import java_runtime


def _create_fake_java_home(root: Path) -> Path:
    java_home = root / "jdk-21"
    (java_home / "bin").mkdir(parents=True, exist_ok=True)
    (java_home / "lib" / "server").mkdir(parents=True, exist_ok=True)
    (java_home / "bin" / "java").write_text("", encoding="utf-8")
    (java_home / "lib" / "server" / "libjvm.so").write_text("", encoding="utf-8")
    return java_home


def test_resolve_java_runtime_prefers_explicit_home(tmp_path: Path, monkeypatch):
    java_home = _create_fake_java_home(tmp_path)
    java_runtime.resolve_java_runtime.cache_clear()
    monkeypatch.setenv("FC_RL_JAVA_HOME", str(java_home))
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.delenv("JDK_HOME", raising=False)
    monkeypatch.setattr(java_runtime, "_iter_workspace_toolchain_candidates", lambda: ())
    monkeypatch.setattr(java_runtime, "_iter_system_java_candidates", lambda: ())
    monkeypatch.setattr(java_runtime.shutil, "which", lambda _: None)

    try:
        runtime = java_runtime.resolve_java_runtime()
        assert runtime is not None
        assert runtime.java_home == java_home.resolve()
        assert runtime.java_executable == (java_home / "bin" / "java").resolve()
        assert runtime.jvm_library == (java_home / "lib" / "server" / "libjvm.so").resolve()
    finally:
        java_runtime.resolve_java_runtime.cache_clear()


def test_resolve_java_runtime_uses_workspace_toolchain(tmp_path: Path, monkeypatch):
    workspace_root = tmp_path / "workspace"
    repo_root = workspace_root / "RL"
    java_home = _create_fake_java_home(workspace_root / ".workspace-tools")
    repo_root.mkdir(parents=True, exist_ok=True)
    java_runtime.resolve_java_runtime.cache_clear()
    monkeypatch.delenv("FC_RL_JAVA_HOME", raising=False)
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.delenv("JDK_HOME", raising=False)
    monkeypatch.setattr(java_runtime, "workspace_root", lambda: workspace_root)
    monkeypatch.setattr(java_runtime, "repo_root", lambda: repo_root)
    monkeypatch.setattr(java_runtime, "_iter_system_java_candidates", lambda: ())
    monkeypatch.setattr(java_runtime.shutil, "which", lambda _: None)

    try:
        runtime = java_runtime.resolve_java_runtime()
        assert runtime is not None
        assert runtime.java_home == java_home.resolve()
    finally:
        java_runtime.resolve_java_runtime.cache_clear()
