from fight_caves_rl.benchmarks.train_ceiling_bench import parse_env_counts


def test_parse_env_counts_none():
    assert parse_env_counts(None) is None


def test_parse_env_counts_csv():
    assert parse_env_counts("4, 16,64") == (4, 16, 64)
