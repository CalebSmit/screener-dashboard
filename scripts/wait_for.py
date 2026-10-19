"""Wait for a long-running job IN THE FOREGROUND, so an unattended session never ends while it runs.

WHY THIS EXISTS - 2026-10-18 and 2026-10-19 (research/2026-10-19-session-ended-on-a-background-job.md).

The code sessions run under ``claude --print``. In that mode the session ends the moment the model
writes a reply without a tool call: there is no later turn, so "you will be notified when it
completes" is never delivered, and the background job is killed when the CLI exits. Two of three
06:00 sessions started the research engine's full rerun in the background, wrote "the rerun will
notify me when it finishes", and ended - 07:03:41 on 2026-10-19 with 14 of 29 experiments written
(the last row is stamped 07:02:56; none after the CLI exited at 07:03:46), and 06:13:45 on
2026-10-18 with 1 of 29. Both days the tree was dirty when the gates ran, gates 1 and 4 failed and
the day's work went to a ``nightly/*`` branch for a later session to salvage.

The two waits those sessions reached for were refused by the tool guard (a shell ``until ... sleep``
loop and a ``Monitor`` command: "contains shell syntax that cannot be statically analyzed"). The
2026-10-18 13:00 session finished the same rerun by polling in the foreground with ``python -c``
loops under ten minutes each. This is that pattern, made one command:

    python scripts/wait_for.py --file logs/_rerun.out --contains "STALE REMAINING: 0"
    python scripts/wait_for.py --file research/experiments/ledger.jsonl --min-lines 201

Each call blocks for at most ``--max-seconds`` (default 540, capped at 570 because the Bash tool's
hard ceiling is 600 seconds, after which it moves the command to the background - which is the
failure this file exists to avoid). Exit status: 0 the condition holds, 3 still waiting (call it
again), 2 bad arguments. It reads files only and writes nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

MAX_SECONDS_CAP = 570      # the Bash tool kills or backgrounds a foreground call at 600 s
DEFAULT_SECONDS = 540
STILL_WAITING = 3


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def condition_met(path: Path, contains: str | None, min_lines: int | None) -> tuple[bool, str]:
    """Whether ``path`` satisfies every condition given, and a one-line progress reading."""
    text = _read(path)
    lines = sum(1 for ln in text.splitlines() if ln.strip())
    ok = path.exists()
    if contains is not None:
        ok = ok and contains in text
    if min_lines is not None:
        ok = ok and lines >= min_lines
    return ok, f"{path.name}: {lines} non-empty lines" + (" (exists)" if path.exists() else " (missing)")


def wait(path: Path, contains: str | None, min_lines: int | None, max_seconds: float,
         every: float, clock=time.monotonic, sleep=time.sleep, out=print) -> int:
    start = clock()
    while True:
        ok, reading = condition_met(path, contains, min_lines)
        elapsed = clock() - start
        if ok:
            out(f"DONE after {elapsed:.0f}s - {reading}", flush=True)
            return 0
        if elapsed + every > max_seconds:
            out(f"STILL WAITING after {elapsed:.0f}s - {reading}. Call this again in the foreground; "
                "do not end your reply while the job runs.", flush=True)
            return STILL_WAITING
        out(f"waiting {elapsed:.0f}s - {reading}", flush=True)
        sleep(every)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", required=True, help="the file the job writes")
    ap.add_argument("--contains", help="finish when this text appears in the file")
    ap.add_argument("--min-lines", type=int, help="finish when the file has at least this many non-empty lines")
    ap.add_argument("--max-seconds", type=float, default=DEFAULT_SECONDS)
    ap.add_argument("--every", type=float, default=30.0, help="seconds between checks")
    a = ap.parse_args(argv)
    if a.contains is None and a.min_lines is None:
        print("give --contains and/or --min-lines", file=sys.stderr)
        return 2
    if a.every <= 0:
        print("--every must be positive", file=sys.stderr)
        return 2
    return wait(Path(a.file), a.contains, a.min_lines, min(a.max_seconds, MAX_SECONDS_CAP), a.every)


if __name__ == "__main__":
    sys.exit(main())
