from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def workspace_root() -> Path:
    return repo_root().parent

