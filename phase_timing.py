"""Wall-clock seconds per phase of a screener run - provenance, never data (2026-10-26).

The 02:00 data loop took 27 minutes on 2026-10-26 and nothing recorded where they went:
``meta.json`` carried one ``fetch_time`` of 1,065 s that covered the ticker fetch, the SEC
passes, the context layer and the metric computation alike, and the loop's log keeps only the
last 25 lines the screener prints. ``research/2026-10-26-data-run-timing.md`` is the
measurement this module was built for.

How it is used: ``RunContext`` owns one ``PhaseClock``; ``run_screener`` calls ``lap(name)`` at
the end of each stage, so every second of the run between two laps is attributed to exactly one
named phase and the phases add up to the whole (contiguous laps - no gaps, no overlaps). At the
end of the run the totals go into the run's ``meta.json`` as ``phase_seconds`` and one line is
printed, which is the last line of the screener's output and therefore inside the data loop's
captured tail.

**Provenance only.** Nothing here may reach ``raw``, ``pct``, a category score, the composite
or the published payload: no scoring module and not ``generate_dashboard`` imports this module
or reads ``phase_seconds`` (``tests/test_phase_timing.py`` fails if one does).
"""
from __future__ import annotations

import time
from typing import Callable


class PhaseClock:
    """Contiguous laps: ``lap(name)`` books the time since the previous lap to ``name``.

    A name lapped twice accumulates. ``note(name, **values)`` attaches small facts about a
    phase (how many tickers, the sum of per-ticker fetch times) that explain its duration.
    """

    def __init__(self, clock: Callable[[], float] = time.perf_counter):
        self._clock = clock
        self._start = clock()
        self._last = self._start
        self.seconds: dict[str, float] = {}
        self.detail: dict[str, dict] = {}

    def lap(self, name: str) -> float:
        """Book the time since the previous lap to ``name``; return this lap's seconds."""
        now = self._clock()
        dt = max(0.0, now - self._last)
        self._last = now
        self.seconds[name] = self.seconds.get(name, 0.0) + dt
        return dt

    def note(self, name: str, **values) -> None:
        self.detail.setdefault(name, {}).update(values)

    def total(self) -> float:
        return self._last - self._start

    def as_meta(self) -> dict:
        """The ``meta.json`` record: phase seconds to 0.1 s, in run order, and their total."""
        return {
            "phase_seconds": {k: round(v, 1) for k, v in self.seconds.items()},
            "phase_detail": self.detail,
            "phase_total_seconds": round(self.total(), 1),
        }

    def summary_line(self, min_seconds: float = 1.0) -> str:
        """One line for the data loop's log: the total, then every phase of at least
        ``min_seconds`` in run order, then how much the shorter phases came to together."""
        shown = [(k, v) for k, v in self.seconds.items() if v >= min_seconds]
        rest = sum(v for v in self.seconds.values() if v < min_seconds)
        parts = [f"{k} {v:.0f}" for k, v in shown]
        if rest > 0:
            parts.append(f"other {rest:.0f}")
        return f"PHASE TIMING (s): total {self.total():.0f} | " + " | ".join(parts)
