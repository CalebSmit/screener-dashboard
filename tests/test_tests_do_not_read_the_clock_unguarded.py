"""A test that reads the wall clock must say why the answer does not depend on it.

Three times now a green suite has gone red with no commit touching the code the failing test
covers, because the test's *premise* was what the clock said when it ran:

- **2026-10-11, 23:25** - ``tests/test_forward_eps_growth.py`` built a "six months from now"
  consensus year-end from ``datetime.now()`` and asserted the blend to a tolerance tighter than
  the drift one day of rounding produces. It failed at 23:25 and passed in the morning.
- **2026-10-17, the first Saturday session** - ``tests/test_option_quote_hours.py`` stamped its
  cache fixture with ``today`` itself, and ``options_cache.reading_for`` deliberately refuses a
  quote date naming a day the market never opened. Two days in seven it failed. Nobody had seen
  it because code sessions ran Mon-Fri until 2026-10-14.
- **2026-10-31, the first weekend after it shipped** - ``tests/test_forward_return_download_skip.py``
  built its snapshot dates from ``datetime.now()``. With an empty performance history the only
  skippable download is the one whose 1-week end is today, which needs a *market-day* snapshot
  exactly seven days earlier, and seven days before a weekend is a weekend.

Each was fixed in its own file, and the lesson did not transfer: the second arrived six days
after the first, the third fourteen days after the second. The cost is not the test - it is that
a red ``main`` strands the whole of that session's work on a branch (gate 1 refuses to merge onto
it), so an hour of someone's morning goes on a defect the clock introduced.

So this module is the transfer. Every call to ``.now()``, ``.today()`` or ``.utcnow()`` in a test
module must carry a ``# clock:`` note saying why the test's verdict is the same whatever the
clock says - on the same line or the line above. It cannot check that the reason is *true*; what
it does is make the question unavoidable at the moment the test is written, which is the only
moment anyone is in a position to answer it. The strongest answer is the one
``test_option_quote_hours`` and ``test_forward_return_download_skip`` now give: don't read the
clock - parametrize over a fixed week.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ``datetime.now``, ``date.today``, ``pd.Timestamp.now``, ``datetime.utcnow`` and the bare forms.
CLOCK_NAMES = {"now", "today", "utcnow"}

MARKER = "# clock:"
MIN_REASON = 20          # characters, so the marker cannot be satisfied by "# clock: ok"


def _test_sources() -> list[pathlib.Path]:
    """Every module pytest collects, plus the root conftest tripwire."""
    files = sorted(ROOT.joinpath("tests").glob("*.py"))
    for extra in ("test_screener.py", "conftest.py"):
        p = ROOT / extra
        if p.exists():
            files.append(p)
    return files


def _clock_reads(path: pathlib.Path) -> list[tuple[int, str]]:
    """(line number, the source line) for each clock read. AST, not a regex: a regex counts the
    ``datetime.now()`` in a docstring explaining one of the faults above."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    out = []
    for node in ast.walk(ast.parse(text, filename=str(path))):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
        if name in CLOCK_NAMES:
            out.append((node.lineno, lines[node.lineno - 1]))
    return sorted(set(out))


def _reason(path: pathlib.Path, lineno: int) -> str | None:
    """The ``# clock:`` reason on the call's own line, or in the comment block directly above it.

    The whole block counts, not just its first line: a reason worth reading usually needs two or
    three lines, and the marker sits on the first of them.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    block = []
    ln = lineno - 1
    while ln >= 1 and lines[ln - 1].strip().startswith("#"):
        block.insert(0, lines[ln - 1].strip().lstrip("#").strip())
        ln -= 1
    own = lines[lineno - 1] if 1 <= lineno <= len(lines) else ""
    if MARKER in own:
        return own.split(MARKER, 1)[1].strip()
    text = " ".join(block)
    marker = MARKER.lstrip("#").strip()          # "clock:" - the '#' is already stripped above
    if marker in text:
        return text.split(marker, 1)[1].strip()
    return None


def _unguarded() -> list[str]:
    bad = []
    for path in _test_sources():
        for lineno, src in _clock_reads(path):
            reason = _reason(path, lineno)
            rel = path.relative_to(ROOT).as_posix()
            if reason is None:
                bad.append(f"{rel}:{lineno}: no '{MARKER}' note\n      {src.strip()}")
            elif len(reason) < MIN_REASON:
                bad.append(f"{rel}:{lineno}: '{MARKER}' note is {len(reason)} chars, "
                           f"needs {MIN_REASON}: {reason!r}")
    return bad


def test_every_clock_read_in_a_test_says_why_the_day_does_not_matter():
    bad = _unguarded()
    assert not bad, (
        "A test that reads the wall clock must say why its verdict does not depend on what the "
        "clock says - a '# clock: <reason>' note on the same line or the line above. Three red "
        "ship gates have come from this (see this module's docstring). The best fix is usually "
        "not a note: take the date from a fixed, parametrized day instead.\n  "
        + "\n  ".join(bad))


def test_the_sweep_finds_the_reads_it_is_supposed_to_find(tmp_path):
    """The floor. A sweep that has stopped matching anything passes vacuously, which is exactly
    the state the three faults above were in."""
    src = tmp_path / "test_sample.py"
    src.write_text(
        "from datetime import datetime, date\n"
        "import pandas as pd\n"
        '"""A docstring mentioning datetime.now() must not count."""\n'
        "def t1():\n"
        "    return datetime.now()\n"
        "def t2():\n"
        "    return date.today()\n"
        "def t3():\n"
        "    return pd.Timestamp.now().normalize()\n"
        "def t4():\n"
        "    return datetime.utcnow()\n"
        "def t5():\n"
        "    return pd.Timestamp('2026-10-31')   # not a clock read\n",
        encoding="utf-8")
    found = _clock_reads(src)
    assert [ln for ln, _ in found] == [5, 7, 9, 11], found


@pytest.mark.parametrize("line,wants", [
    ("    x = datetime.now()   # clock: elapsed days only, so any day gives the same answer", True),
    ("    x = datetime.now()   # clock: fine", False),
    ("    x = datetime.now()", False),
])
def test_the_marker_needs_a_reason_not_just_the_token(tmp_path, line, wants):
    src = tmp_path / "test_sample.py"
    src.write_text("from datetime import datetime\ndef t():\n" + line + "\n", encoding="utf-8")
    (lineno, _), = _clock_reads(src)
    reason = _reason(src, lineno)
    assert bool(reason and len(reason) >= MIN_REASON) is wants


def test_the_note_is_accepted_on_the_line_above(tmp_path):
    src = tmp_path / "test_sample.py"
    src.write_text("from datetime import datetime\n"
                   "def t():\n"
                   "    # clock: a comment on its own line above the call is the usual form\n"
                   "    return datetime.now()\n", encoding="utf-8")
    (lineno, _), = _clock_reads(src)
    assert _reason(src, lineno).startswith("a comment on its own line")


def test_the_whole_comment_block_above_the_call_counts(tmp_path):
    """A reason worth reading usually runs to two or three lines, with the marker on the first."""
    src = tmp_path / "test_sample.py"
    src.write_text("from datetime import datetime\n"
                   "def t():\n"
                   "    # clock: the first line of the reason,\n"
                   "    # and the second line, which the length check must also see.\n"
                   "    return datetime.now()\n", encoding="utf-8")
    (lineno, _), = _clock_reads(src)
    reason = _reason(src, lineno)
    assert reason.startswith("the first line") and "second line" in reason


def test_a_comment_block_that_is_not_above_the_call_does_not_count(tmp_path):
    """The block has to be contiguous with the call - a note three statements earlier is not an
    answer about this one."""
    src = tmp_path / "test_sample.py"
    src.write_text("from datetime import datetime\n"
                   "def t():\n"
                   "    # clock: a reason sitting above some other statement entirely\n"
                   "    x = 1\n"
                   "    return datetime.now()\n", encoding="utf-8")
    (lineno, _), = _clock_reads(src)
    assert _reason(src, lineno) is None


def test_the_sweep_covers_the_three_modules_the_faults_came_from():
    """Named literally, because the sweep is only worth having if it reaches them. If a module is
    renamed or its clock read is removed, delete the row - do not weaken the sweep."""
    covered = {p.relative_to(ROOT).as_posix() for p in _test_sources()}
    for name in ("tests/test_forward_eps_growth.py",
                 "tests/test_option_quote_hours.py",
                 "tests/test_forward_return_download_skip.py"):
        assert name in covered, name
        assert _clock_reads(ROOT / name), f"{name}: no clock read found - has it been renamed?"
