#!/usr/bin/env python3
"""Decide, from evidence visible *outside* this machine, whether the loops ran.

Why this exists
---------------
`CLAUDE.md` priority -1: "Nothing watches whether the loop is running. A run
that never fires writes no log, so its absence is invisible until someone counts
files." The 2026-08-21 retrospective measured the cost: of 11 scheduled
code-loop slots from 2026-08-06 to 2026-08-20, **5 never fired at all**, four of
them consecutive weekdays (08-17..08-20) while the machine sat logged out.
Nobody noticed for six days.

`scripts/write_brief.py` already prints a "THE ROUTINE IS NOT RUNNING" banner,
and it cannot solve this, because **the watchdog runs inside the thing it is
watching**: `write_brief.py` is invoked only from `data-run.ps1` (line 90) and
`nightly-screener.ps1` (line 107). If neither loop fires, the brief is never
regenerated, so the banner never appears and `MORNING_BRIEF.md` goes on
cheerfully describing the last run that *did* happen. That banner can only ever
catch "one loop died while the other lived" - never "the machine was off",
which is the documented dominant failure mode.

So the heartbeat has to be read by something that is not the heart. This module
is the decision logic; `.github/workflows/loop-watchdog.yml` is the external
observer that runs it on GitHub's infrastructure, where a dead PC cannot
silence it.

The heartbeat
-------------
Both loops publish a commit to `main` from a `finally` block, so it lands
whether the run succeeded, was discarded by a gate, or crashed:

    data loop   `brief: data run <date>`      (also `data: screener run <date>`)
    code loop   `brief: code session <date>`  (suffixed "- SESSION DID NOT RUN"
                                               when the session failed)

That suffix is deliberately still a heartbeat: it means the loop *fired*, which
is the question this file answers. Whether the session then did anything useful
is what the brief and `NIGHTLY_LOG.md` are for. A missing commit is the only
thing that means "nothing ran".

`brief: evening session <date>` is deliberately **not** matched. Those are
owner-initiated interactive sessions; they prove a human was present, not that
the 06:00 task fired.

Deliberately stdlib-only, for the same reason `write_brief.py` is: this is what
reports that everything else is broken, so it must not be able to break with it.

Usage:
    python scripts/check_loop_health.py            # human-readable report
    python scripts/check_loop_health.py --json     # machine-readable verdict
    python scripts/check_loop_health.py --strict   # exit non-zero if stalled
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# How far back to look. 14 calendar days is ~10 weekday slots - long enough to
# show the 4-consecutive-weekday outage that motivated this file, short enough
# that the report stays readable.
LOOKBACK_DAYS = 14

# Verdicts, worst last. A single missed weekday is a blip (one reboot, one
# network drop) and alarming on it would train the owner to ignore the alarm;
# two consecutive is the shape the real outages took.
OK, WARN, STALLED = "ok", "warn", "stalled"
EXIT_CODES = {OK: 0, WARN: 0, STALLED: 2}


@dataclass(frozen=True)
class Loop:
    """One scheduled loop and how to tell whether it fired."""

    key: str
    label: str
    slot: str  # human description of when it is meant to run
    deadline_hour: int  # local hour by which that day's run must have landed
    patterns: tuple[str, ...]
    script: str

    def matches(self, subject: str) -> bool:
        return any(re.match(p, subject) for p in self.patterns)


# `deadline_hour` is not the slot time. Both tasks carry an at-logon trigger
# (see scripts/register-tasks.ps1) so a slot missed while the machine sat at the
# login screen is picked up when the owner next signs in - the same morning, in
# practice. Judging a 02:00 slot at 02:15 would therefore report a stall that
# the catch-up was about to fix. The deadline is the hour after which a
# same-day catch-up has plainly not happened.
LOOPS: tuple[Loop, ...] = (
    Loop(
        key="data",
        label="Data run",
        slot="02:00 Mon-Fri",
        deadline_hour=12,
        patterns=(r"^brief: data run\b", r"^data: screener run\b"),
        script="scripts/data-run.ps1",
    ),
    Loop(
        key="code",
        label="Code session",
        slot="06:00 Mon-Fri",
        deadline_hour=16,
        patterns=(r"^brief: code session\b",),
        script="scripts/nightly-screener.ps1",
    ),
)


@dataclass
class LoopStatus:
    loop: Loop
    last_heartbeat: date | None
    missed: list[date] = field(default_factory=list)
    consecutive_missed: int = 0
    weekdays_checked: int = 0

    @property
    def verdict(self) -> str:
        if self.consecutive_missed >= 2:
            return STALLED
        if self.consecutive_missed == 1:
            return WARN
        return OK


# ---------------------------------------------------------------------------
# Decision logic - the part worth testing. No git, no clock, no filesystem.
# ---------------------------------------------------------------------------


def closed_weekdays(now: datetime, deadline_hour: int, lookback_days: int = LOOKBACK_DAYS) -> list[date]:
    """Weekdays in the window whose deadline has passed, oldest first.

    `now` is naive, in the scheduling timezone - see `schedule_now()`.

    A weekday whose deadline has not yet arrived is not yet evidence of
    anything, so it is excluded rather than counted as missed. This is what
    stops the watchdog reporting a stall every morning before the loops have
    had their chance to run.
    """
    out: list[date] = []
    for back in range(lookback_days, -1, -1):
        d = (now - timedelta(days=back)).date()
        if d.weekday() >= 5:  # Saturday, Sunday - the loops are Mon-Fri
            continue
        deadline = datetime.combine(d, datetime.min.time()).replace(hour=deadline_hour)
        if now >= deadline:
            out.append(d)
    return out


def assess(loop: Loop, heartbeats: set[date], now: datetime,
           lookback_days: int = LOOKBACK_DAYS) -> LoopStatus:
    """Compare a loop's heartbeat dates against the weekdays it owed."""
    checked = closed_weekdays(now, loop.deadline_hour, lookback_days)
    missed = [d for d in checked if d not in heartbeats]

    # Consecutive misses counting back from the most recent closed weekday.
    # "3 missed out of 10" is a flaky machine; "the last 4 in a row" is an
    # outage, and only the second one is worth waking somebody for.
    consecutive = 0
    for d in reversed(checked):
        if d in heartbeats:
            break
        consecutive += 1

    known = [d for d in heartbeats if d <= now.date()]
    return LoopStatus(
        loop=loop,
        last_heartbeat=max(known) if known else None,
        missed=missed,
        consecutive_missed=consecutive,
        weekdays_checked=len(checked),
    )


def overall(statuses: list[LoopStatus]) -> str:
    """The worst verdict across the loops - one bad loop is a bad routine."""
    order = {OK: 0, WARN: 1, STALLED: 2}
    worst = OK
    for s in statuses:
        if order[s.verdict] > order[worst]:
            worst = s.verdict
    return worst


# ---------------------------------------------------------------------------
# Evidence gathering
# ---------------------------------------------------------------------------


def git_log(ref: str, lookback_days: int = LOOKBACK_DAYS) -> list[tuple[datetime, str]]:
    """(commit datetime with its own UTC offset, subject) for recent commits."""
    try:
        proc = subprocess.run(
            ["git", "log", ref, f"--since={lookback_days + 2} days ago",
             "--format=%cI%x09%s", "-n", "1000"],
            cwd=ROOT, capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []

    out: list[tuple[datetime, str]] = []
    for line in proc.stdout.splitlines():
        if "\t" not in line:
            continue
        stamp, subject = line.split("\t", 1)
        try:
            out.append((datetime.fromisoformat(stamp.strip()), subject.strip()))
        except ValueError:
            continue
    return out


def resolve_ref() -> str:
    """Prefer origin/main, then main, then HEAD.

    In CI the checkout is detached at a commit, and `main` may not exist as a
    local branch; on the owner's machine a nightly session runs from a branch,
    so HEAD is not main. Trying them in this order gets the published history
    in both places.
    """
    for ref in ("origin/main", "main", "HEAD"):
        try:
            proc = subprocess.run(
                ["git", "rev-parse", "--verify", "--quiet", ref],
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode == 0 and proc.stdout.strip():
            return ref
    return "HEAD"


def heartbeat_dates(commits: list[tuple[datetime, str]], loop: Loop) -> set[date]:
    """The local calendar dates on which this loop demonstrably ran.

    The commit's *own* recorded date is used, not the observer's. A watchdog
    running in UTC on GitHub must not decide that a 02:13 Chicago run happened
    on the previous day.
    """
    return {ts.date() for ts, subject in commits if loop.matches(subject)}


def schedule_now(commits: list[tuple[datetime, str]]) -> datetime:
    """'Now' in the timezone the loops are scheduled in, as a naive datetime.

    The offset is taken from the most recent commit rather than from a named
    timezone: `zoneinfo` needs the `tzdata` package on Windows, which this
    stdlib-only script will not depend on, and the loops' own commits carry the
    scheduling machine's offset by construction. Across a DST boundary this can
    be an hour out, which is immaterial against deadlines measured in hours.
    """
    now = datetime.now().astimezone()
    if commits:
        offset = max(commits, key=lambda c: c[0])[0].utcoffset()
        if offset is not None:
            return now.astimezone(timezone(offset)).replace(tzinfo=None)
    return now.replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render(statuses: list[LoopStatus], verdict: str, now: datetime) -> str:
    lines: list[str] = []
    a = lines.append

    a(f"Loop health as of {now:%Y-%m-%d %H:%M} (scheduling timezone)")
    a("")
    for s in statuses:
        last = s.last_heartbeat.isoformat() if s.last_heartbeat else "never"
        a(f"  {s.loop.label:<14} {s.verdict.upper():<8} "
          f"last heartbeat {last}; "
          f"{len(s.missed)} of the last {s.weekdays_checked} weekdays missed"
          + (f", {s.consecutive_missed} in a row" if s.consecutive_missed else ""))
        if s.missed:
            a(f"    missed: {', '.join(d.isoformat() for d in s.missed)}")
    a("")

    if verdict == OK:
        a("Both loops are firing.")
        return "\n".join(lines)

    a("A loop that never fires writes no log, so nothing on the machine will")
    a("report this. Most likely cause, by past frequency: the PC rebooted and")
    a("nobody signed back in - the tasks run only while a user is logged on.")
    a("")
    a("What to check:")
    a("  1. Is the machine on and logged in?")
    a("  2. Task Scheduler: are both tasks present and enabled?")
    a("     Re-register from version control with:")
    a("       powershell -ExecutionPolicy Bypass -File scripts\\register-tasks.ps1")
    a("  3. `logs/` - a task that fired and failed leaves a log; a task that")
    a("     never fired leaves nothing, which is what this alert means.")
    return "\n".join(lines)


def to_dict(statuses: list[LoopStatus], verdict: str, now: datetime) -> dict:
    return {
        "verdict": verdict,
        "checked_at": now.isoformat(timespec="seconds"),
        "loops": [
            {
                "key": s.loop.key,
                "label": s.loop.label,
                "slot": s.loop.slot,
                "verdict": s.verdict,
                "last_heartbeat": s.last_heartbeat.isoformat() if s.last_heartbeat else None,
                "missed": [d.isoformat() for d in s.missed],
                "consecutive_missed": s.consecutive_missed,
                "weekdays_checked": s.weekdays_checked,
            }
            for s in statuses
        ],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit the verdict as JSON")
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero when a loop is stalled")
    ap.add_argument("--lookback-days", type=int, default=LOOKBACK_DAYS)
    args = ap.parse_args(argv)

    commits = git_log(resolve_ref(), args.lookback_days)
    now = schedule_now(commits)

    statuses = [
        assess(loop, heartbeat_dates(commits, loop), now, args.lookback_days)
        for loop in LOOPS
    ]
    verdict = overall(statuses)

    if args.json:
        print(json.dumps(to_dict(statuses, verdict, now), indent=2))
    else:
        print(render(statuses, verdict, now))

    return EXIT_CODES[verdict] if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
