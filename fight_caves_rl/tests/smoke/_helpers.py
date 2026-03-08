from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fight_caves_rl.bridge.errors import BridgeError
from fight_caves_rl.bridge.launcher import assert_sim_runtime_ready, discover_headless_runtime_paths


def require_live_runtime() -> None:
    try:
        paths = discover_headless_runtime_paths()
        assert_sim_runtime_ready(paths)
    except BridgeError as exc:
        pytest.skip(str(exc))


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_script(script_name: str, *args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    root = repo_root()
    return subprocess.run(
        [sys.executable, str(root / "scripts" / script_name), *args],
        cwd=str(cwd or root),
        capture_output=True,
        text=True,
        check=False,
    )


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
