from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re

ARTIFACT_VERSION = 0

ARTIFACT_TYPE_BY_CATEGORY = {
    "checkpoint": "model",
    "checkpoint_metadata": "metadata",
    "run_manifest": "run-manifest",
    "eval_summary": "eval-summary",
    "replay_pack": "replay-pack",
    "replay_index": "replay-index",
}

_TOKEN_PATTERN = re.compile(r"[^a-z0-9._-]+")
_DASH_PATTERN = re.compile(r"-{2,}")


@dataclass(frozen=True)
class ArtifactRecord:
    category: str
    artifact_type: str
    name: str
    version: int
    path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sanitize_artifact_token(value: str) -> str:
    lowered = value.strip().lower().replace(" ", "-")
    normalized = _TOKEN_PATTERN.sub("-", lowered)
    normalized = _DASH_PATTERN.sub("-", normalized).strip("-")
    return normalized or "artifact"


def artifact_type_for_category(category: str) -> str:
    if category not in ARTIFACT_TYPE_BY_CATEGORY:
        raise ValueError(f"Unsupported artifact category: {category!r}")
    return ARTIFACT_TYPE_BY_CATEGORY[category]


def build_artifact_name(
    *,
    run_kind: str,
    config_id: str,
    run_id: str,
    category: str,
    version: int = ARTIFACT_VERSION,
) -> str:
    tokens = (
        "fight-caves-rl",
        run_kind,
        config_id,
        run_id,
        category,
        f"v{version}",
    )
    return "-".join(sanitize_artifact_token(token) for token in tokens)


def build_artifact_record(
    *,
    run_kind: str,
    config_id: str,
    run_id: str,
    category: str,
    path: str | Path,
    version: int = ARTIFACT_VERSION,
) -> ArtifactRecord:
    return ArtifactRecord(
        category=category,
        artifact_type=artifact_type_for_category(category),
        name=build_artifact_name(
            run_kind=run_kind,
            config_id=config_id,
            run_id=run_id,
            category=category,
            version=version,
        ),
        version=version,
        path=str(Path(path)),
    )
