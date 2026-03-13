from fight_caves_rl.benchmarks.train_bench import (
    CORE_INPROCESS_RUNNER_MODE,
    PROTOTYPE_SYNC_RUNNER_MODE,
    SMOKE_SUBPROCESS_RUNNER_MODE,
    parse_runner_mode,
)
from fight_caves_rl.benchmarks.train_ceiling_bench import parse_env_counts


def test_parse_env_counts_none():
    assert parse_env_counts(None) is None


def test_parse_env_counts_csv():
    assert parse_env_counts("4, 16,64") == (4, 16, 64)


def test_parse_runner_mode_defaults_to_smoke():
    assert parse_runner_mode(None) == SMOKE_SUBPROCESS_RUNNER_MODE


def test_parse_runner_mode_returns_explicit_value():
    assert parse_runner_mode(CORE_INPROCESS_RUNNER_MODE) == CORE_INPROCESS_RUNNER_MODE


def test_parse_runner_mode_returns_prototype_value():
    assert parse_runner_mode(PROTOTYPE_SYNC_RUNNER_MODE) == PROTOTYPE_SYNC_RUNNER_MODE
