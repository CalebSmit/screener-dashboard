#!/usr/bin/env python3
"""
Instrumentation Layer for the Multi-Factor Stock Screener
==========================================================
Provides lightweight, observational event tracing that records every
significant operation (NET, IO, CALC, INIT, IDLE) to an in-memory log,
then flushes to CSV/Markdown/HTML at pipeline end.

Usage:
    from instrumentation import EventLog, trace_event

    log = EventLog()
    with trace_event(log, "CALC", "Winsorize metrics"):
        df = winsorize_metrics(df, 0.01, 0.01)

    log.flush_all("reports/")
"""

import csv
import inspect
import io
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional


class Event:
    """Single instrumentation event."""
    __slots__ = ("seq", "wall_time", "event_type", "duration_ms",
                 "operation", "caller", "status", "details")

    def __init__(self, seq: int, wall_time: str, event_type: str,
                 duration_ms: float, operation: str, caller: str,
                 status: str, details: str):
        self.seq = seq
        self.wall_time = wall_time
        self.event_type = event_type
        self.duration_ms = duration_ms
        self.operation = operation
        self.caller = caller
        self.status = status
        self.details = details

    def duration_human(self) -> str:
        ms = self.duration_ms
        if ms < 1000:
            return f"{ms:.0f} ms ({ms/1000:.3f}s)"
        elif ms < 60000:
            return f"{ms:.0f} ms ({ms/1000:.1f}s)"
        else:
            return f"{ms:.0f} ms ({ms/1000:.1f}s / {ms/60000:.1f}m)"

    def to_dict(self) -> dict:
        return {
            "#": self.seq,
            "Time": self.wall_time,
            "Type": self.event_type,
            "Duration": self.duration_human(),
            "Operation": self.operation,
            "Caller": self.caller,
            "Status": self.status,
            "Details": self.details,
        }


class EventLog:
    """In-memory event log that flushes to CSV/MD/HTML."""

    def __init__(self):
        self.events: list[Event] = []
        self._seq = 0
        self._t0 = time.monotonic()

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def record(self, event_type: str, operation: str, duration_ms: float,
               status: str = "OK", details: str = "",
               caller: Optional[str] = None) -> Event:
        if caller is None:
            caller = _get_caller(skip=2)
        evt = Event(
            seq=self._next_seq(),
            wall_time=datetime.now().strftime("%H:%M:%S"),
            event_type=event_type,
            duration_ms=round(duration_ms, 1),
            operation=operation,
            caller=caller,
            status=status,
            details=details,
        )
        self.events.append(evt)
        return evt

    def flush_csv(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            cols = ["#", "Time", "Type", "Duration", "Operation",
                    "Caller", "Status", "Details"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for evt in self.events:
                w.writerow(evt.to_dict())
        return str(path)

    def flush_md(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["#", "Time", "Type", "Duration", "Operation",
                "Caller", "Status", "Details"]
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Run Log — Full Chronological Event Trace\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary stats
            total_ms = sum(e.duration_ms for e in self.events)
            type_counts = {}
            for e in self.events:
                type_counts[e.event_type] = type_counts.get(e.event_type, 0) + 1
            f.write("## Summary\n\n")
            f.write(f"- Total events: {len(self.events)}\n")
            f.write(f"- Total traced time: {total_ms/1000:.1f}s\n")
            f.write(f"- Event types: {', '.join(f'{k}={v}' for k,v in sorted(type_counts.items()))}\n")
            failures = sum(1 for e in self.events if e.status != "OK")
            f.write(f"- Failures: {failures}\n\n")

            # Table
            f.write("## Event Log\n\n")
            f.write("| " + " | ".join(cols) + " |\n")
            f.write("| " + " | ".join("---" for _ in cols) + " |\n")
            for evt in self.events:
                d = evt.to_dict()
                row = " | ".join(str(d.get(c, "")).replace("|", "\\|")
                                 for c in cols)
                f.write(f"| {row} |\n")
        return str(path)

    def flush_html(self, path: str | Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = ["#", "Time", "Type", "Duration", "Operation",
                "Caller", "Status", "Details"]
        with open(path, "w", encoding="utf-8") as f:
            f.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
            f.write("<title>Run Log</title>")
            f.write("<style>")
            f.write("body{font-family:monospace;margin:20px}")
            f.write("table{border-collapse:collapse;width:100%}")
            f.write("th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;font-size:12px}")
            f.write("th{background:#1F4E79;color:#fff}")
            f.write("tr:nth-child(even){background:#f9f9f9}")
            f.write(".FAIL{background:#FFC7CE}")
            f.write(".NET{color:#0066cc}.CALC{color:#006600}")
            f.write(".INIT{color:#666}.IDLE{color:#999}")
            f.write(".READ{color:#8B4513}.WRITE{color:#8B0000}")
            f.write("</style></head><body>")
            f.write(f"<h1>Run Log — Full Chronological Event Trace</h1>")
            f.write(f"<p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")
            f.write("<table><thead><tr>")
            for c in cols:
                f.write(f"<th>{c}</th>")
            f.write("</tr></thead><tbody>")
            for evt in self.events:
                d = evt.to_dict()
                cls = evt.status if evt.status != "OK" else evt.event_type
                f.write(f"<tr class='{cls}'>")
                for c in cols:
                    val = str(d.get(c, "")).replace("<", "&lt;").replace(">", "&gt;")
                    f.write(f"<td>{val}</td>")
                f.write("</tr>")
            f.write("</tbody></table></body></html>")
        return str(path)

    def flush_all(self, report_dir: str | Path):
        d = Path(report_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.flush_csv(d / "run_log_full.csv")
        self.flush_md(d / "run_log_full.md")
        self.flush_html(d / "run_log_full.html")


def _get_caller(skip: int = 2) -> str:
    """Get caller info as file:function:line."""
    try:
        frame = inspect.stack()[skip]
        filename = Path(frame.filename).name
        return f"{filename}:{frame.function}:{frame.lineno}"
    except (IndexError, AttributeError):
        return "unknown"


@contextmanager
def trace_event(log: EventLog, event_type: str, operation: str,
                details: str = "", caller: Optional[str] = None):
    """Context manager that records a timed event to the log.

    Usage:
        with trace_event(log, "CALC", "Winsorize metrics"):
            df = winsorize_metrics(df, 0.01, 0.01)
    """
    # Capture caller BEFORE entering the generator (contextlib.__enter__
    # would be the caller from inside the yield).
    if caller is None:
        # skip=2: trace_event -> contextmanager wrapper -> actual caller
        try:
            frame = inspect.stack()[2]
            caller = f"{Path(frame.filename).name}:{frame.function}:{frame.lineno}"
        except (IndexError, AttributeError):
            caller = _get_caller(skip=2)
    t0 = time.monotonic()
    status = "OK"
    try:
        yield
    except Exception as exc:
        status = "FAIL"
        details = f"{details}; ERROR: {type(exc).__name__}: {exc}" if details else f"ERROR: {type(exc).__name__}: {exc}"
        raise
    finally:
        elapsed_ms = (time.monotonic() - t0) * 1000
        log.record(event_type, operation, elapsed_ms,
                   status=status, details=details, caller=caller)


def trace_net_call(log: EventLog, operation: str, hostname: str = "",
                   status_code: int = 0, retries: int = 0,
                   nbytes: int = 0, duration_ms: float = 0,
                   status: str = "OK", caller: Optional[str] = None):
    """Record a network call event."""
    parts = []
    if hostname:
        parts.append(f"host={hostname}")
    if status_code:
        parts.append(f"status={status_code}")
    if retries:
        parts.append(f"retries={retries}")
    if nbytes:
        parts.append(f"bytes={nbytes}")
    details = "; ".join(parts)
    if caller is None:
        caller = _get_caller(skip=1)
    log.record("NET", operation, duration_ms,
               status=status, details=details, caller=caller)


def trace_io(log: EventLog, event_type: str, operation: str,
             path: str = "", nbytes: int = 0,
             duration_ms: float = 0, status: str = "OK",
             caller: Optional[str] = None):
    """Record a disk I/O event (READ or WRITE)."""
    parts = []
    if path:
        parts.append(f"path={path}")
    if nbytes:
        parts.append(f"bytes={nbytes}")
    details = "; ".join(parts)
    if caller is None:
        caller = _get_caller(skip=1)
    log.record(event_type, operation, duration_ms,
               status=status, details=details, caller=caller)
