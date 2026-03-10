from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths
from fight_caves_rl.utils.java_runtime import resolve_java_home, resolve_jvm_library_path


def require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
        if resolve_jvm_library_path() is None:
            raise BridgeError(
                "No Linux JVM runtime resolved. Set FC_RL_JAVA_HOME or JAVA_HOME "
                "to a JDK/JRE home containing lib/server/libjvm.so."
            )
    except BridgeError as exc:
        pytest.skip(str(exc))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_script(
    script_name: str,
    *args: str,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout_seconds: float = 120.0,
) -> subprocess.CompletedProcess[str]:
    root = repo_root()
    allowed_keys = {
        "HOME",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "PATH",
        "PYTHONNOUSERSITE",
        "SSL_CERT_FILE",
        "REQUESTS_CA_BUNDLE",
        "TMPDIR",
        "USER",
        "JAVA_HOME",
        "JDK_HOME",
    }
    process_env = {
        key: value
        for key, value in os.environ.items()
        if key in allowed_keys or key.startswith("XDG_") or key.startswith("FC_RL_")
    }
    process_env.setdefault("HOME", str(Path.home()))
    process_env.setdefault("LANG", "C.UTF-8")
    process_env.setdefault("PATH", os.defpath)
    java_home = resolve_java_home()
    if java_home is not None:
        process_env.setdefault("JAVA_HOME", str(java_home))
        process_env.setdefault("JDK_HOME", str(java_home))
    if env:
        process_env.update(env)
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stdout_file:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr_file:
            try:
                result = subprocess.run(
                    [sys.executable, str(root / "scripts" / script_name), *args],
                    cwd=str(cwd or root),
                    env=process_env,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    check=False,
                    timeout=timeout_seconds,
                )
                returncode = result.returncode
                command = result.args
            except subprocess.TimeoutExpired as exc:
                returncode = 124
                command = exc.cmd
            stdout_file.seek(0)
            stderr_file.seek(0)
            stderr = stderr_file.read()
            if returncode == 124:
                stderr = (
                    f"{stderr}\nSubprocess timed out after {float(timeout_seconds):.1f}s.\n"
                ).lstrip()
            return subprocess.CompletedProcess(
                args=command,
                returncode=returncode,
                stdout=stdout_file.read(),
                stderr=stderr,
            )


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def offline_wandb_env(tmp_path: Path, *, tags: str) -> dict[str, str]:
    return {
        "WANDB_PROJECT": "fight-caves-rl-tests",
        "WANDB_MODE": "offline",
        "WANDB_GROUP": "smoke",
        "WANDB_TAGS": tags,
        "WANDB_DIR": str(tmp_path / "wandb"),
        "WANDB_DATA_DIR": str(tmp_path / "wandb-data"),
        "WANDB_CACHE_DIR": str(tmp_path / "wandb-cache"),
    }
