from __future__ import annotations

import pytest


@pytest.fixture
def disable_subprocess_capture(capfd: pytest.CaptureFixture[str]):
    # Live train/eval smoke tests launch subprocesses that bring up the
    # embedded JVM and native logging stacks. Suspending pytest's fd capture
    # around these tests avoids capture-layer interference with that runtime.
    with capfd.disabled():
        yield
