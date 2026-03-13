from pathlib import Path
import subprocess
from types import SimpleNamespace

from fight_caves_rl.benchmarks import common


def test_resolve_java_runtime_profile_tolerates_missing_java(monkeypatch):
    common._resolve_java_runtime_profile.cache_clear()
    monkeypatch.setattr(common, "resolve_java_executable", lambda: Path("/tmp/missing-java"))

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("java")

    monkeypatch.setattr(common.subprocess, "run", fake_run)

    try:
        assert common._resolve_java_runtime_profile() == (None, None)
    finally:
        common._resolve_java_runtime_profile.cache_clear()


def test_resolve_java_runtime_profile_uses_resolved_executable(monkeypatch):
    common._resolve_java_runtime_profile.cache_clear()
    monkeypatch.setattr(common, "resolve_java_executable", lambda: Path("/tmp/custom-java"))

    def fake_run(args, **kwargs):
        assert args[0] == "/tmp/custom-java"
        return subprocess.CompletedProcess(
            args=args,
            returncode=0,
            stdout="",
            stderr=(
                "Property settings:\n"
                "    java.runtime.version = 21.0.10+7-LTS\n"
                "    java.vm.name = OpenJDK 64-Bit Server VM\n"
            ),
        )

    monkeypatch.setattr(common.subprocess, "run", fake_run)

    try:
        assert common._resolve_java_runtime_profile() == (
            "21.0.10+7-LTS",
            "OpenJDK 64-Bit Server VM",
        )
    finally:
        common._resolve_java_runtime_profile.cache_clear()


def test_capture_peak_memory_profile_aggregates_self_and_children(monkeypatch):
    def fake_getrusage(which):
        if which == common.resource.RUSAGE_SELF:
            return SimpleNamespace(ru_maxrss=128)
        if which == common.resource.RUSAGE_CHILDREN:
            return SimpleNamespace(ru_maxrss=64)
        raise AssertionError(which)

    monkeypatch.setattr(common.resource, "getrusage", fake_getrusage)

    assert common.capture_peak_memory_profile() == {
        "process_peak_rss_kib": 128,
        "children_peak_rss_kib": 64,
        "combined_peak_rss_kib": 192,
    }
