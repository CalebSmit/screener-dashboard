"""The option-quote refresh is a third scheduled task, and it must stay harmless.

WHY THIS EXISTS - 2026-10-09. The two existing loops both publish: they take the repo lock, run
git, and commit. A third task that did any of that would be a third publish path, and CLAUDE.md
already records the lesson twice over - *the strongest checks guard the path that publishes least
often, so the weaker runner is usually the bug.*

So this one publishes nothing. It fetches option quotes after the close into
``data/options/quotes.json`` (gitignored) and stops. That is what makes it safe to add without
giving it gates of its own, and it is what this module pins:

* it is registered from version control, like the other two (settled row -1);
* it runs **no git command** and takes **no repo lock**, so it cannot collide with either loop or
  with an owner session working in the tree;
* it has **no logon trigger and no catch-up**, because a refresh outside the quote window does
  nothing useful - and the staggered logon delays the two loops depend on stay a pair;
* a dead-hour run exits **0**, so a genuine fault is not lost among expected red runs;
* nothing it writes can reach a score.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "scripts" / "register-tasks.ps1"
REFRESH = ROOT / "scripts" / "refresh-option-quotes.ps1"
DATA_RUN = ROOT / "scripts" / "data-run.ps1"
NIGHTLY = ROOT / "scripts" / "nightly-screener.ps1"

TASK_NAME = "Screener Option Quotes"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig")


def _spec_block() -> str:
    """The task's own entry in ``$Specs``, not the mention of its name in the doc comment."""
    src = _src(REGISTER)
    m = re.search(r"Name\s*=\s*'" + re.escape(TASK_NAME) + r"'", src)
    assert m, f"{TASK_NAME} has no entry in $Specs"
    block = src[m.start():]
    end = block.index("Description")
    return block[:end]


# --------------------------------------------------------------------------- registration
def test_the_refresh_script_exists():
    assert REFRESH.exists(), "register-tasks.ps1 would SKIP a task whose script is missing"


def test_the_task_definition_lives_in_version_control():
    """Settled row -1: scheduled-task definitions stay in the repo, not only in Task Scheduler."""
    src = _src(REGISTER)
    assert TASK_NAME in src
    assert "refresh-option-quotes.ps1" in src


def test_the_task_runs_after_the_close_not_overnight():
    """The whole point. Measured 2026-10-09: 394 of 503 usable at 21:27 ET, 0 of 503 at 03:00 ET."""
    block = _spec_block()
    at = re.search(r"At\s*=\s*'([^']+)'", block)
    assert at, block
    hour = at.group(1)
    assert re.match(r"^(?:[5-9]|1[01]):\d{2}PM$", hour), (
        f"scheduled at {hour}; option quotes are only served between the close and midnight ET"
    )


def test_the_task_has_no_logon_trigger():
    """A catch-up at an arbitrary hour correctly does nothing, so it is only log noise - and
    adding one would break test_logon_delays_are_staggered, which counts exactly two."""
    block = _spec_block()
    assert re.search(r"NoLogon\s*=\s*\$true", block), block
    assert "LogonDelay" not in block, "the refresh must not join the staggered logon pair"


def test_the_two_loops_still_hold_the_only_staggered_logon_delays():
    """Guards the settled constraint from the other side: adding this task must not have
    disturbed the data PT3M / code PT20M pair."""
    delays = re.findall(r"LogonDelay\s*=\s*'(PT\d+M)'", _src(REGISTER))
    assert delays == ["PT3M", "PT20M"], delays


def test_the_logon_flag_is_actually_honoured():
    """A per-task flag that is declared and never read is the bug this catches."""
    src = _src(REGISTER)
    assert "-not $s.NoLogon" in src, "NoLogon is set on the spec but never consulted"
    assert "if ($s.NoCatchUp)" in src, "NoCatchUp is set on the spec but never consulted"


def test_the_refresh_does_not_opt_into_running_late():
    block = _spec_block()
    assert re.search(r"NoCatchUp\s*=\s*\$true", block), block


def test_the_docstring_names_three_tasks():
    """Rule 9: a process doc that has gone stale sends the next session to rebuild something."""
    src = _src(REGISTER)
    head = src[:src.index("[CmdletBinding()]")]
    assert "three tasks" in head
    assert TASK_NAME in head


# --------------------------------------------------------------------------- harmlessness
def test_the_refresh_takes_no_repo_lock_because_it_needs_none():
    src = _src(REFRESH)
    assert "Enter-RepoLock" not in src and "repo-lock.ps1" not in src, (
        "if this ever needs the lock it has started touching tracked files - rethink it"
    )


def test_the_refresh_runs_no_mutating_git_command():
    src = _src(REFRESH)
    forbidden = ("git add", "git commit", "git push", "git checkout", "git merge",
                 "git reset", "git stash", "git tag", "git pull")
    for cmd in forbidden:
        assert cmd not in src, f"the refresh must not run `{cmd}` - it publishes nothing"


def test_the_refresh_only_reads_git_to_confirm_it_changed_nothing():
    src = _src(REFRESH)
    assert "git status --porcelain" in src
    assert "dirty" in src.lower()


def test_the_refresh_exits_zero_even_at_a_dead_hour():
    """A task that goes red every time it correctly does nothing trains everyone to ignore it."""
    src = _src(REFRESH)
    assert re.search(r"^exit 0\s*$", src, re.M), "no unconditional exit 0"
    assert "exit 1" not in src


def test_the_refresh_writes_a_log_like_the_other_two():
    src = _src(REFRESH)
    assert "options-" in src and "logs" in src
    assert "function Write-Log" in src


def test_the_loops_do_not_call_the_refresh():
    """The 02:00 and 06:00 loops sit inside the dead window; calling it there would fetch ~1,000
    chains for nothing, which is the defect this whole change removes."""
    for p in (DATA_RUN, NIGHTLY):
        assert "refresh-option-quotes" not in _src(p), p.name
        assert "options_cache.py" not in _src(p), p.name


def test_the_task_name_is_the_one_the_script_is_registered_under():
    """So a session reading the log can find the task with Get-ScheduledTask."""
    assert TASK_NAME in _src(REGISTER)
    assert TASK_NAME in _src(REFRESH) or "register-tasks.ps1" in _src(REFRESH)
