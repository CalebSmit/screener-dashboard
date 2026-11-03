"""Print CLAUDE.md rule 8's five health numbers as the line a NIGHTLY_LOG.md entry opens with.

WHY THIS EXISTS - 2026-11-02. Every session works the five numbers out by hand before it does
anything else, and the hand-made versions have gone wrong in exactly the ways rule 8 records: the
evidence line read off the wrong horizon for weeks, and a hand-written "age 1 day" stayed in a file
for eight days. The brief already computes most of them (``write_brief.ic_observations``,
``outcome``); this prints them in the log's own shape, from the same functions, so the entry and
the brief cannot disagree. Read the line before pasting it: it reports, it does not judge whether
the session should change its plan - rule 8's "if it exceeds 40, finding out why is that session's
work" still needs a reader.

Read-only. Never raises; a number it cannot read says "unknown" and why.

    python scripts/health_numbers.py
"""
from __future__ import annotations

import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import write_brief as wb  # noqa: E402

# A code-session log is finished once the runner has written its brief (it does so from
# `finally`, whatever happened). The session reading this is usually inside the newest log, which
# has no such line yet - so "the last code session" is the newest *finished* one.
FINISHED = ("Morning brief published", "Morning brief written", "=== Run ")
OPEN_ITEM = re.compile(r"^\*\*(\d+(?:\.\d+)?)\. ([^*(]+?)\s*\(opened (\d{4}-\d{2}-\d{2})", re.M)

# The runner's own marker for a session that never started, and nothing else.
#
# WHY THIS IS A PATTERN AND NOT A SUBSTRING - 2026-11-03. This function used to ask
# ``"SESSION DID NOT RUN" in text``, and the runner *logs this module's output* into every run log
# ("[INFO] Health: **Health (rule 8, all five):** last code session ran? **no** - ... SESSION DID
# NOT RUN (...)"). So the phrase appears in the log of a session that ran perfectly, carried there
# by the previous session's verdict - and the next run reads it back as its own.
#
# That makes the alarm **self-perpetuating**: from the first genuine failure onward, every later
# session is reported as "did not run" for ever, however healthy. Measured that day:
# `logs/nightly-2026-11-03_080346.log` ends "=== Run complete: shipped to main ===", merged six
# commits and tagged `good/2026-11-03`, and was reported as "**no** - SESSION DID NOT RUN ()" -
# with an empty reason, because the echoed copy reads "RUN (" where the real marker reads "RUN:".
# Both the 06:00 and 13:00 sessions were handed that false line in their prompts.
#
# It is the stuck-alarm twin of the stuck-green defect rule 8 exists for (2026-09-18: "a tripwire
# wired to a number that cannot stand still is decoration"). A health line that cannot leave the
# alarm state gets ignored, and an ignored line still looks like coverage - the watchdog row's own
# argument. So: match the runner's line shape (`[ERROR] SESSION DID NOT RUN: <reason>`), never the
# phrase wherever it appears.
DID_NOT_RUN = re.compile(r"\[ERROR\]\s*SESSION DID NOT RUN:\s*(.*)")


def last_code_session() -> str:
    for path in sorted(wb.LOGS.glob("nightly-*.log"), key=lambda p: p.stat().st_mtime, reverse=True):
        lines = wb.tail(path, 400)
        text = "\n".join(lines)
        if any(m in text for m in wb.NOT_A_SESSION) or not any(m in text for m in FINISHED):
            continue
        marker = next((m for m in (DID_NOT_RUN.search(ln) for ln in lines) if m), None)
        if marker:
            why = marker.group(1).strip()
            return f"**no** - `logs/{path.name}`: SESSION DID NOT RUN ({why[:140]})"
        state, _ = wb.outcome(lines)
        verdict = "yes" if state.startswith("completed") else f"**{state}**"
        return f"{verdict} - `logs/{path.name}` ({state})"
    return "unknown - no finished code-session log in logs/"


def data_loop() -> str:
    path = wb.newest("datarun-*.log")
    if not path:
        return "unknown - no data-run log in logs/"
    lines = wb.tail(path, 2000)
    text = "\n".join(lines)
    state, _ = wb.outcome(lines)
    health = "HEALTH: PASS" if "HEALTH: PASS" in text else "no HEALTH: PASS line"
    published = "Data loop complete" in text
    verdict = "yes" if published else "**no**"
    return f"{verdict} - `logs/{path.name}` ({state}; {health})"


def evidence_base(today: datetime | None = None) -> str:
    return wb.ic_observations(today)


def priority_0() -> str:
    try:
        cfg = (ROOT / "config.yaml").read_text(encoding="utf-8")
    except OSError:
        return "unknown - config.yaml unreadable"
    m = re.search(r"^\s*allow_auto_apply:\s*(\w+)", cfg, re.M)
    flag = m.group(1) if m else "unknown"
    return f"`allow_auto_apply` **{flag}**" + ("" if flag == "false" else " - rule 4 says false")


def open_items(claude_md: str | None = None, today: datetime | None = None) -> list[tuple[str, str, int]]:
    """(number, title, age in days) for each item under "### Open" carrying an opened date and
    not marked closed or done in its own heading, in file order."""
    text = claude_md if claude_md is not None else (ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    start = text.find("### Open")
    if start < 0:
        return []
    today = today or datetime.now()
    out = []
    for m in OPEN_ITEM.finditer(text, start):
        line = text[m.start(): text.find("\n", m.start())]
        if re.search(r"\b(CLOSED|DONE)\b", line):
            continue
        age = (today.date() - datetime.strptime(m.group(3), "%Y-%m-%d").date()).days
        out.append((m.group(1), m.group(2).strip().rstrip(".").strip(), age))
    return out


def top_roadmap(today: datetime | None = None) -> str:
    try:
        items = open_items(today=today)
    except OSError:
        return "unknown - CLAUDE.md unreadable"
    if not items:
        return "unknown - no item under \"### Open\" carries an opened date"
    num, title, age = items[0]
    rest = "; ".join(f"{n} {a}d" for n, _, a in items[1:])
    return f"**{num}, {title}** (age **{age} days**)" + (f"; also open: {rest}" if rest else "")


def line(today: datetime | None = None) -> str:
    return ("**Health (rule 8, all five):** last code session ran? " + last_code_session()
            + ". Data loop published? " + data_loop()
            + ". Evidence base at `1m`: " + evidence_base(today)
            + ". Priority 0: " + priority_0()
            + ". Top open roadmap item: " + top_roadmap(today) + ".")


def main() -> int:
    try:
        print(line())
    except Exception as e:  # noqa: BLE001 - a helper that prints a status must not be the failure
        print(f"health_numbers: could not compute ({type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
