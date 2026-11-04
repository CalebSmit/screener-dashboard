"""``scripts/health_numbers.py`` prints rule 8's five numbers from the brief's own functions.

Pinned: the last code session is the newest *finished* one (the session running it is inside an
unfinished log), a dry run or a stand-down is never "the last session", the top roadmap item is the
first dated item under "### Open" that its own heading does not mark closed, ages are computed
rather than copied, and the script never fails its caller.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import health_numbers as hn  # noqa: E402

DAY = datetime(2026, 11, 2, 19, 0)

CLAUDE_FIXTURE = """# x
### Settled
**0.9. Old thing (opened 2026-09-01)** - not under Open, ignored.
### Open
**0.8 and 0.9 are CLOSED** (2026-10-07).
**0.11. The context layer - CLOSED 2026-10-20** (record ...).
**0.13. The Factor Lab (opened 2026-10-12).** Owner item.
**0.10. Keep the inputs (opened 2026-10-07).**
**0.14. The inputs against the filings (opened 2026-10-14, owner-run).**
**0.6. One observation per run date - CLOSED 2026-10-08 (opened 2026-09-30).**
"""


def test_open_items_are_read_in_order_with_computed_ages():
    items = hn.open_items(CLAUDE_FIXTURE, today=DAY)
    assert items == [("0.13", "The Factor Lab", 21), ("0.10", "Keep the inputs", 26),
                     ("0.14", "The inputs against the filings", 19)]


def test_the_real_file_has_a_dated_open_item():
    """Asserts the premise, not which item: the list changes as items close."""
    items = hn.open_items(today=DAY)
    assert items, "no dated item under ### Open - rule 8's roadmap line has nothing to read"
    assert all(age >= 0 for _, _, age in items)


def _log(folder: Path, name: str, body: str, mtime: float):
    p = folder / name
    p.write_text(body, encoding="utf-8")
    os.utime(p, (mtime, mtime))


def test_the_last_session_is_the_newest_finished_one(tmp_path, monkeypatch):
    monkeypatch.setattr(hn.wb, "LOGS", tmp_path)
    _log(tmp_path, "nightly-a.log", "[06:00:00] [INFO] === Run complete: shipped to main ===\n"
         "[06:40:00] [INFO] Writing morning brief...\n[06:40:00] [INFO] Morning brief published.\n", 1000)
    _log(tmp_path, "nightly-b.log", "[13:00:00] [INFO] === Dry run complete ===\n"
         "[13:00:01] [INFO] Writing morning brief...\n[13:00:01] [INFO] Morning brief published.\n", 2000)
    _log(tmp_path, "nightly-c.log", "[13:00:00] [INFO] Invoking Claude Code...\n", 3000)   # running now
    got = hn.last_code_session()
    assert got.startswith("yes - `logs/nightly-a.log`")


def test_a_session_that_did_not_run_says_so_and_why(tmp_path, monkeypatch):
    monkeypatch.setattr(hn.wb, "LOGS", tmp_path)
    _log(tmp_path, "nightly-a.log",
         "[06:00:04] [ERROR] SESSION DID NOT RUN: session reported an error - Failed to authenticate\n"
         "[06:08:00] [INFO] Writing morning brief...\n[06:08:00] [INFO] Morning brief published.\n", 1000)
    got = hn.last_code_session()
    assert got.startswith("**no**") and "Failed to authenticate" in got


# The health line the runner prints into every log, verbatim in shape: it quotes a *previous*
# session's verdict, so it carries the phrase "SESSION DID NOT RUN" into the log of a session
# that ran perfectly. Note "RUN (" rather than the runner's "RUN:".
ECHOED_HEALTH_LINE = (
    "[08:07:01] [INFO] Health: **Health (rule 8, all five):** last code session ran? "
    "**no** - `logs/nightly-2026-11-02_130000-midday.log`: SESSION DID NOT RUN "
    "(session reported an error - Failed to authenticate: OAuth session expired and could "
    "not be refreshed). Data loop published? yes - `logs/datarun-2026-11-03_020000.log` "
    "(completed; HEALTH: PASS).\n"
)


def test_a_healthy_session_is_not_called_dead_by_the_previous_verdict_in_its_own_log(
        tmp_path, monkeypatch):
    """The echo defect, 2026-11-03: the alarm could not leave the alarm state.

    `last_code_session` asked `"SESSION DID NOT RUN" in text`, and the runner logs this
    module's own output into every run log. So from the first genuine failure onward every
    later session read the previous verdict back out of its own log and reported itself dead -
    for ever, however healthy. `logs/nightly-2026-11-03_080346.log` shipped six commits and
    tagged `good/2026-11-03`, and was reported "**no** - SESSION DID NOT RUN ()".

    This is the stuck-alarm twin of the stuck-green defect rule 8 exists for. A line that can
    only say "broken" gets ignored, and an ignored line still looks like coverage.
    """
    monkeypatch.setattr(hn.wb, "LOGS", tmp_path)
    _log(tmp_path, "nightly-a.log",
         "[08:03:46] [INFO] === Code loop 2026-11-03 ===\n"
         + ECHOED_HEALTH_LINE +
         "[09:14:34] [INFO] === Run complete: shipped to main ===\n"
         "[09:14:41] [INFO] Writing morning brief...\n[09:14:41] [INFO] Morning brief published.\n", 1000)
    got = hn.last_code_session()
    assert got.startswith("yes - `logs/nightly-a.log`"), (
        f"a session that shipped to main must read as healthy whatever the previous "
        f"session's verdict was; got {got!r}"
    )
    assert "DID NOT RUN" not in got


def test_a_real_marker_still_fires_when_the_echo_sits_in_the_same_log(tmp_path, monkeypatch):
    """A guard against the fix swinging the other way, NOT a detector of the echo defect.

    Named for what it does, because the first name claimed more than the body delivers and
    `research/measurements/2026-11-03-health-echo-mutations.py` measured it: with the fix
    reverted this test still passes, since the substring match happens to reach the right
    verdict when a genuine marker is present too. Its value is the opposite direction - a
    narrower pattern could have stopped seeing real failures, and this is what says it does
    not. (2026-10-25's lesson: a test's name is read far more often than its body.)
    """
    monkeypatch.setattr(hn.wb, "LOGS", tmp_path)
    _log(tmp_path, "nightly-a.log",
         ECHOED_HEALTH_LINE +
         "[13:04:04] [ERROR] SESSION DID NOT RUN: usage limit reached\n"
         "[13:08:00] [INFO] Writing morning brief...\n[13:08:00] [INFO] Morning brief published.\n", 1000)
    got = hn.last_code_session()
    assert got.startswith("**no**"), f"a genuine marker must still fire; got {got!r}"
    assert "usage limit reached" in got, (
        "the reason must come from the runner's own marker, not from the echoed line - the "
        "echo is why the reason printed empty on 2026-11-03"
    )
    assert "OAuth" not in got, "the reason was taken from the echoed previous verdict"


def test_the_reason_is_never_empty_when_the_verdict_is_no(tmp_path, monkeypatch):
    """Pins the visible symptom - the empty "()" - on a log with no echo in it.

    Measured as a non-detector of the echo defect (the reverted fix passes this too, because
    without an echo the substring match reads the real marker). It is here so that a future
    change cannot reintroduce a reasonless verdict by some other route.
    """
    monkeypatch.setattr(hn.wb, "LOGS", tmp_path)
    _log(tmp_path, "nightly-a.log",
         "[13:04:04] [ERROR] SESSION DID NOT RUN: tree was dirty\n"
         "[13:08:00] [INFO] Writing morning brief...\n[13:08:00] [INFO] Morning brief published.\n", 1000)
    got = hn.last_code_session()
    assert "SESSION DID NOT RUN ()" not in got, (
        "a verdict of no with no reason tells the next session nothing; it meant the phrase "
        "had been matched somewhere that is not the runner's marker"
    )
    assert "tree was dirty" in got


def test_the_real_logs_do_not_report_a_shipped_session_as_dead():
    """A tripwire on the live folder, since the echo is in every log written since 2026-11-02.

    Asserts its own premise rather than today's outcome: if the newest finished log says it
    shipped, the verdict must not be "no". Says nothing when no such log exists.
    """
    got = hn.last_code_session()
    if not got.startswith("yes") and "`logs/" in got:
        name = got.split("`logs/", 1)[1].split("`", 1)[0]
        path = hn.wb.LOGS / name
        if path.exists():
            text = "\n".join(hn.wb.tail(path, 400))
            assert "shipped to main" not in text, (
                f"{name} says it shipped to main and the health line calls it dead: {got!r}"
            )


def test_the_top_item_line_names_the_first_and_counts_the_rest(monkeypatch):
    monkeypatch.setattr(hn, "open_items", lambda claude_md=None, today=None:
                        [("0.13", "The Factor Lab", 21), ("0.10", "Keep the inputs", 26)])
    assert hn.top_roadmap(DAY) == "**0.13, The Factor Lab** (age **21 days**); also open: 0.10 26d"


def test_the_script_prints_one_line_and_never_fails_its_caller():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "health_numbers.py")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    out = r.stdout.strip()
    assert out.count("\n") == 0 and out.startswith("**Health (rule 8, all five):**")
    for part in ("last code session ran?", "Data loop published?", "Evidence base at `1m`",
                 "Priority 0", "Top open roadmap item"):
        assert part in out


def test_the_nightly_prompt_points_at_it():
    assert "scripts/health_numbers.py" in (ROOT / "prompts" / "nightly.md").read_text(encoding="utf-8")


def test_the_runner_hands_the_line_to_the_session():
    """The runner computes the line before invoking the session and fills {{HEALTH}} with it, so a
    session starts with rule 8's numbers instead of re-deriving them (2026-11-02)."""
    ps = (ROOT / "scripts" / "nightly-screener.ps1").read_text(encoding="utf-8")
    compute = ps.index("@('scripts/health_numbers.py')")
    fill = ps.index(".Replace('{{HEALTH}}', $HealthText)")
    invoke = ps.index('Write-Log "Invoking Claude Code..."')
    assert compute < fill < invoke
    block = ps[compute:fill]
    assert "Stop-Run" not in block and "not computed" in block      # never fatal
    assert "{{HEALTH}}" in (ROOT / "prompts" / "nightly.md").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Work the gates held back (2026-11-04): a refused branch on origin is named in the line every
# session's prompt carries, and only while it holds commits origin/main lacks.
# ---------------------------------------------------------------------------
def _git(tmp: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=tmp, check=True, capture_output=True, text=True).stdout


def test_a_refused_branch_on_origin_is_named_and_a_merged_one_is_not(tmp_path):
    origin, clone = tmp_path / "origin.git", tmp_path / "clone"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], check=True, capture_output=True)
    for k, v in (("user.email", "t@example.invalid"), ("user.name", "t")):
        _git(clone, "config", k, v)
    _git(clone, "commit", "-q", "--allow-empty", "-m", "base")
    _git(clone, "push", "-q", "origin", "main")
    _git(clone, "checkout", "-q", "-b", "nightly/2026-11-03")       # merged: nothing main lacks
    _git(clone, "push", "-q", "origin", "nightly/2026-11-03")
    _git(clone, "checkout", "-q", "-b", "nightly/2026-11-04")       # refused: two commits ahead
    _git(clone, "commit", "-q", "--allow-empty", "-m", "first")
    _git(clone, "commit", "-q", "--allow-empty", "-m", "Investment stays standalone")
    _git(clone, "push", "-q", "origin", "nightly/2026-11-04")
    _git(clone, "fetch", "-q", "origin")

    def run(args):
        return subprocess.run(["git", *args], cwd=clone, capture_output=True, text=True)

    got = hn.held_back(run)
    assert got == [("origin/nightly/2026-11-04", 2, "Investment stays standalone")]
    text = hn.held_back_text(got)
    assert "`origin/nightly/2026-11-04` (2 commits, newest: Investment stays standalone)" in text
    assert "Work the gates held back" in text
    assert "Work the gates held back" in (ROOT / "prompts" / "nightly.md").read_text(encoding="utf-8")


def test_no_held_back_work_adds_nothing_and_an_unreadable_repo_is_none():
    assert hn.held_back_text([]) == ""

    class Failed:
        returncode, stdout = 128, ""
    assert hn.held_back(lambda args: Failed()) == []

    def boom(args):
        raise OSError("git missing")
    assert hn.held_back(boom) == []
