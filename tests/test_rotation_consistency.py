"""The rotation the runner hands a session must be the rotation the docs describe.

WHY THIS EXISTS - 2026-10-16 (retrospective).

``nightly-screener.ps1`` interpolates ``$FocusByDay[<today>]`` into ``{{FOCUS}}``,
which is the **first line** of the prompt and the heading of the log entry. It is
therefore the single most-read instruction of the session, and it is the one piece
of the rotation that nothing checked.

Two had drifted, both into outright contradiction with the file the prompt tells
the session overrides it:

* **Tuesday** read "Ship a dashboard change", against ``CLAUDE.md`` row 0.8 and
  ``prompts/nightly.md`` ("The visual design is finished (owner, 2026-10-07) ...
  Do not start design passes, restyling or 'fine-tuning'"). A session was told to
  ship a dashboard change and, two screens later, told not to.
* **Friday** read "HARDEN AND TEACH" with no mention of the Factor Lab, which
  ``CLAUDE.md`` had said was "the lab's day" since 2026-10-15.

Rule 9 says keep these files true. This module is what makes that checkable
rather than remembered: every weekday the runner names must have a matching day
bullet in the prompt, and the forbidden-work rules are asserted against the
string the session actually receives.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "nightly-screener.ps1"
PROMPT = ROOT / "prompts" / "nightly.md"
CLAUDE = ROOT / "CLAUDE.md"

WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
WEEKEND = ("Saturday", "Sunday")


def _focus_by_day() -> dict[str, str]:
    """Parse $FocusByDay out of the runner. One entry per line, PS 5.1 quoting."""
    source = RUNNER.read_text(encoding="utf-8-sig")
    start = source.find("$FocusByDay = @{")
    assert start != -1, "nightly-screener.ps1 no longer defines $FocusByDay"
    block = source[start : source.find("\n}", start)]
    found = {}
    for day, body in re.findall(r"'(\w+)'\s*=\s*'((?:[^']|'')*)'", block):
        found[day] = body.replace("''", "'")
    return found


def _prompt_day_bullets() -> dict[str, str]:
    """The '**Monday - research.**' style bullets in prompts/nightly.md."""
    text = PROMPT.read_text(encoding="utf-8")
    return {m.group(1): m.group(2) for m in re.finditer(r"\*\*(\w+day) - ([^*]+)\*\*", text)}


@pytest.fixture(scope="module")
def focus():
    return _focus_by_day()


def test_the_runner_names_every_day_of_the_week(focus):
    for day in WEEKDAYS + WEEKEND:
        assert day in focus, f"$FocusByDay has no entry for {day}"
        assert focus[day].strip(), f"{day}'s focus string is empty"


def test_every_weekday_the_runner_names_has_a_day_bullet_in_the_prompt(focus):
    bullets = _prompt_day_bullets()
    missing = [d for d in WEEKDAYS if d not in bullets]
    assert missing == [], (
        "the runner hands these days a focus the prompt never describes: "
        + ", ".join(missing)
    )


def test_the_weekend_slots_say_weekend(focus):
    for day in WEEKEND:
        assert "WEEKEND" in focus[day], (
            f"{day} is worked by the weekend slot, which takes the top open item"
        )


# ---------------------------------------------------------------------------
# The two contradictions found on 2026-10-16. Asserted against the string the
# session actually receives, not against the prose describing it.
# ---------------------------------------------------------------------------

# Owner direction 2026-10-07, CLAUDE.md row 0.8: nightly sessions do not take on
# design or fine-tuning work. A focus string may not commission it.
_DESIGN_WORK = (
    "ship a dashboard change",
    "redesign",
    "restyl",
    "fine-tun",
    "make it look",
)


@pytest.mark.parametrize("day", WEEKDAYS + WEEKEND)
def test_no_focus_string_commissions_design_work(focus, day):
    body = focus[day].lower()
    offenders = [p for p in _DESIGN_WORK if p in body]
    assert offenders == [], (
        f"{day}'s focus asks for {offenders}, which CLAUDE.md row 0.8 forbids "
        "(owner, 2026-10-07). Fix the focus string, not the rule."
    )


def test_tuesday_sends_the_session_at_data_and_methodology(focus):
    body = focus["Tuesday"].lower()
    assert "methodology" in body and "data" in body, (
        "Tuesday exists so somebody opens the live page; since 2026-10-07 the fix "
        "it is allowed to make is a data or methodology one"
    )


def test_friday_is_the_labs_day_in_the_runner_and_in_claude_md(focus):
    assert "lab" in focus["Friday"].lower(), (
        "CLAUDE.md says Friday is the Factor Lab's day; the runner must say so too"
    )
    claude = CLAUDE.read_text(encoding="utf-8")
    assert "Friday is the lab's day" in claude or "**Fri** | **The Factor Lab**" in claude, (
        "if Friday stops being the lab's day, change CLAUDE.md and this test together"
    )
