from __future__ import annotations

import os

import pufferlib.pufferl

from fight_caves_rl.puffer.trainer import ConfigurablePuffeRL, should_enable_dashboard


def test_should_enable_dashboard_requires_request_and_tty():
    assert should_enable_dashboard({"logging": {"dashboard": True}}, stdout_isatty=True, stderr_isatty=True)
    assert not should_enable_dashboard(
        {"logging": {"dashboard": True}},
        stdout_isatty=False,
        stderr_isatty=True,
    )
    assert not should_enable_dashboard(
        {"logging": {"dashboard": False}},
        stdout_isatty=True,
        stderr_isatty=True,
    )


def test_configurable_pufferl_skips_dashboard_when_disabled(monkeypatch):
    calls: list[str] = []

    def fake_print_dashboard(self, *args, **kwargs):
        calls.append("called")

    monkeypatch.setattr(pufferlib.pufferl.PuffeRL, "print_dashboard", fake_print_dashboard)
    trainer = object.__new__(ConfigurablePuffeRL)
    trainer._dashboard_enabled = False

    trainer.print_dashboard(clear=True)

    assert calls == []


def test_configurable_pufferl_delegates_dashboard_when_enabled(monkeypatch):
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_print_dashboard(self, *args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(pufferlib.pufferl.PuffeRL, "print_dashboard", fake_print_dashboard)
    trainer = object.__new__(ConfigurablePuffeRL)
    trainer._dashboard_enabled = True

    trainer.print_dashboard(clear=True)

    assert calls == [((), {"clear": True})]


def test_configurable_pufferl_skips_checkpoint_when_disabled(tmp_path):
    trainer = object.__new__(ConfigurablePuffeRL)
    trainer._checkpointing_enabled = False
    trainer.config = {"data_dir": str(tmp_path)}
    trainer.logger = type("Logger", (), {"run_id": "null"})()

    checkpoint_path = trainer.save_checkpoint()

    assert checkpoint_path == os.path.join(str(tmp_path), "null.pt")


def test_configurable_pufferl_close_skips_checkpoint_when_disabled(tmp_path):
    vecenv_calls: list[str] = []
    util_calls: list[str] = []

    trainer = object.__new__(ConfigurablePuffeRL)
    trainer._checkpointing_enabled = False
    trainer.config = {"data_dir": str(tmp_path)}
    trainer.logger = type("Logger", (), {"run_id": "null"})()
    trainer.vecenv = type("VecEnv", (), {"close": lambda self: vecenv_calls.append("closed")})()
    trainer.utilization = type(
        "Utilization", (), {"stop": lambda self: util_calls.append("stopped")}
    )()

    checkpoint_path = trainer.close()

    assert checkpoint_path == os.path.join(str(tmp_path), "null.pt")
    assert vecenv_calls == ["closed"]
    assert util_calls == ["stopped"]


def test_configurable_pufferl_mean_and_log_returns_empty_when_logging_disabled():
    trainer = object.__new__(ConfigurablePuffeRL)
    trainer._logging_enabled = False

    assert trainer.mean_and_log() == {}
