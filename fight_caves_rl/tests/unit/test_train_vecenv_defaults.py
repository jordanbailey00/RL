from fight_caves_rl.envs.shared_memory_transport import INFO_PAYLOAD_MODE_MINIMAL
from fight_caves_rl.puffer.factory import (
    ENV_BACKEND_V1_BRIDGE,
    ENV_BACKEND_V2_FAST,
    load_smoke_train_config,
    make_vecenv,
    resolve_train_env_backend,
)


def test_load_smoke_train_config_defaults_to_minimal_train_info_payload():
    config = load_smoke_train_config()

    assert config["env"]["include_future_leakage"] is False
    assert config["env"]["info_payload_mode"] == INFO_PAYLOAD_MODE_MINIMAL


def test_make_vecenv_defaults_missing_info_payload_mode_to_minimal(monkeypatch):
    config = load_smoke_train_config()
    del config["env"]["info_payload_mode"]
    captured: dict[str, object] = {}

    def _fake_fast_vecenv(batch_config, *, reward_adapter):
        captured["batch_config"] = batch_config
        captured["reward_adapter"] = reward_adapter
        return object()

    monkeypatch.setattr("fight_caves_rl.puffer.factory.FastKernelVecEnv", _fake_fast_vecenv)

    make_vecenv(config, backend="embedded")

    batch_config = captured["batch_config"]
    assert batch_config.include_future_leakage is False
    assert batch_config.info_payload_mode == INFO_PAYLOAD_MODE_MINIMAL


def test_load_smoke_train_config_defaults_to_v2_fast_backend():
    config = load_smoke_train_config()

    assert resolve_train_env_backend(config) == ENV_BACKEND_V2_FAST


def test_make_vecenv_defaults_to_v2_fast_backend(monkeypatch):
    config = load_smoke_train_config()
    captured: dict[str, object] = {}

    def _fake_fast_vecenv(batch_config, *, reward_adapter):
        captured["batch_config"] = batch_config
        captured["reward_adapter"] = reward_adapter
        return object()

    monkeypatch.setattr("fight_caves_rl.puffer.factory.FastKernelVecEnv", _fake_fast_vecenv)

    make_vecenv(config, backend="embedded")

    assert captured["batch_config"].info_payload_mode == INFO_PAYLOAD_MODE_MINIMAL
    assert captured["reward_adapter"].config_id == str(config["reward_config"])


def test_make_vecenv_can_still_route_v1_bridge_fallback(monkeypatch):
    config = load_smoke_train_config()
    config["env"]["env_backend"] = ENV_BACKEND_V1_BRIDGE
    config["reward_config"] = "reward_shaped_v0"
    captured: dict[str, object] = {}

    def _fake_vecenv(batch_config, *, reward_fn):
        captured["batch_config"] = batch_config
        captured["reward_fn"] = reward_fn
        return object()

    monkeypatch.setattr("fight_caves_rl.puffer.factory.HeadlessBatchVecEnv", _fake_vecenv)

    make_vecenv(config, backend="embedded")

    assert captured["batch_config"].info_payload_mode == INFO_PAYLOAD_MODE_MINIMAL


def test_make_vecenv_routes_v2_fast_embedded_backend_without_using_v1_transport(monkeypatch):
    config = load_smoke_train_config()
    config["env"]["env_backend"] = ENV_BACKEND_V2_FAST
    captured: dict[str, object] = {}

    def _fake_fast_vecenv(batch_config, *, reward_adapter):
        captured["batch_config"] = batch_config
        captured["reward_adapter"] = reward_adapter
        return object()

    monkeypatch.setattr("fight_caves_rl.puffer.factory.FastKernelVecEnv", _fake_fast_vecenv)

    result = make_vecenv(config, backend="embedded")

    assert result is not None
    batch_config = captured["batch_config"]
    assert batch_config.info_payload_mode == INFO_PAYLOAD_MODE_MINIMAL
    assert captured["reward_adapter"].config_id == str(config["reward_config"])


def test_make_vecenv_routes_v2_fast_subprocess_backend_via_subprocess_wrapper(monkeypatch):
    config = load_smoke_train_config()
    config["env"]["env_backend"] = ENV_BACKEND_V2_FAST
    config["env"]["subprocess_worker_count"] = 2
    captured: dict[str, object] = {}

    def _fake_subprocess_vecenv(subprocess_config):
        captured["config"] = subprocess_config
        return object()

    monkeypatch.setattr(
        "fight_caves_rl.puffer.factory.SubprocessHeadlessBatchVecEnv",
        _fake_subprocess_vecenv,
    )

    result = make_vecenv(config, backend="subprocess")

    assert result is not None
    subprocess_config = captured["config"]
    assert subprocess_config.env_backend == ENV_BACKEND_V2_FAST
    assert subprocess_config.worker_count == 2
    assert subprocess_config.info_payload_mode == INFO_PAYLOAD_MODE_MINIMAL
