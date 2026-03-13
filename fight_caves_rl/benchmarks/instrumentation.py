from __future__ import annotations

from collections import defaultdict


InstrumentationSnapshot = dict[str, dict[str, float | int]]


class BucketInstrumentation:
    def __init__(self) -> None:
        self._seconds: defaultdict[str, float] = defaultdict(float)
        self._calls: defaultdict[str, int] = defaultdict(int)

    def record(self, bucket: str, seconds: float, *, calls: int = 1) -> None:
        self._seconds[str(bucket)] += max(0.0, float(seconds))
        self._calls[str(bucket)] += max(0, int(calls))

    def snapshot(self) -> InstrumentationSnapshot:
        return {
            name: {
                "seconds": float(self._seconds[name]),
                "calls": int(self._calls[name]),
            }
            for name in sorted(self._seconds)
        }

    def clear(self) -> None:
        self._seconds.clear()
        self._calls.clear()
