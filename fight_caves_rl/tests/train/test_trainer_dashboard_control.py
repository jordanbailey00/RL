from __future__ import annotations

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
