from fight_caves_rl.logging.artifact_naming import (
    ARTIFACT_VERSION,
    artifact_type_for_category,
    build_artifact_record,
)


def test_build_artifact_record_normalizes_tokens_and_versions():
    record = build_artifact_record(
        run_kind="Train Smoke",
        config_id="smoke/ppo v0",
        run_id="Run ID 123",
        category="checkpoint_metadata",
        path="/tmp/checkpoint.metadata.json",
    )

    assert record.category == "checkpoint_metadata"
    assert record.artifact_type == "metadata"
    assert record.version == ARTIFACT_VERSION
    assert record.name == "fight-caves-rl-train-smoke-smoke-ppo-v0-run-id-123-checkpoint_metadata-v0"
    assert record.path == "/tmp/checkpoint.metadata.json"


def test_artifact_type_for_category_rejects_unknown_categories():
    try:
        artifact_type_for_category("unknown")
    except ValueError as exc:
        assert "Unsupported artifact category" in str(exc)
    else:
        raise AssertionError("Expected unsupported artifact category to raise ValueError.")
