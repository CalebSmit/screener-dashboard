"""A session cut off mid-work must not strand it, and must not leak it.

WHY THIS EXISTS - 2026-10-06.

The 06:00 code session started the owner's "premium" redesign, finished stage 1
(a design-system pass in `generate_dashboard.py`, with its written argument,
a contrast checker and a screenshot harness) and was then killed at 06:18:47 by
`API error 429 - You've hit your session limit`. It had committed nothing and
written no log entry.

What the runner did next was wrong in two ways:

1. Gate 4 (clean tree) failed, so it pushed the nightly branch - **empty**, 0
   new commits - and left the work uncommitted in the working tree, on the failed
   branch. Nothing about it was recoverable from the pushed branch.
2. The 02:00 data loop begins with `git checkout main` and no look at the tree,
   then regenerates the **live** dashboard from whatever `generate_dashboard.py`
   is on disk and publishes it. An uncommitted tracked change survives a
   checkout, so the half-finished redesign would have been built into
   `index.html` and pushed to GitHub Pages unreviewed. Its own gates (parse
   check, 237 claim tests) would have caught a broken page, not an unfinished
   one. It was caught by hand, at 19:05, seven hours before it would have fired.

The failure is not rare in kind - a usage limit ends a session at an arbitrary
point - and design work is the most token-hungry work the routine does (rendered
screenshots, rebuilds), so the owner's redesign directive makes it likelier.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NIGHTLY = ROOT / "scripts" / "nightly-screener.ps1"
DATA_RUN = ROOT / "scripts" / "data-run.ps1"
PROMPT = ROOT / "prompts" / "nightly.md"

GIT = shutil.which("git")
needs_git = pytest.mark.skipif(GIT is None, reason="needs git")


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _failure_block() -> str:
    src = _src(NIGHTLY)
    i = src.index("SHIP GATES FAILED")
    return src[i:i + 6000]


# ---------------------------------------------------------------------------
# The code loop: a failed run commits what it left, then leaves a clean main
# ---------------------------------------------------------------------------

def test_failed_run_commits_leftovers_before_pushing_the_branch():
    block = _failure_block()
    salvage = block.index("salvaged from an interrupted session")
    push = block.index("'push', '-u', 'origin', $Branch")
    assert salvage < push, (
        "the leftover work must be committed BEFORE the branch is pushed, or the "
        "pushed branch is empty and the work is unrecoverable")


def test_salvage_is_only_ever_a_branch_commit():
    """It must never apply when the session committed onto main, and it must not
    be able to merge anything - the gates still decide that."""
    block = _failure_block()
    guard = block.index("if ($workBranch -ne 'main')")
    salvage = block.index("salvaged from an interrupted session")
    assert guard < salvage
    assert "'merge'" not in block[guard:salvage]


def test_failed_run_returns_the_folder_to_a_clean_main():
    block = _failure_block()
    stop = block.index("Stop-Run \"Run finished with failing gates")
    tail = block[:stop]
    assert "'checkout', 'main'" in tail[tail.index("'push', '-u', 'origin', $Branch"):], (
        "a failed run must not leave the shared folder sitting on the failed branch")


def test_checkout_back_to_main_only_happens_on_a_clean_tree():
    """Switching branches with a dirty tree would carry the work across - the
    exact leak this exists to stop."""
    block = _failure_block()
    back = block.index("'checkout', 'main'")
    assert "if (-not $left.Text.Trim())" in block[:back]


# ---------------------------------------------------------------------------
# The data loop: never publish from a dirty tree
# ---------------------------------------------------------------------------

def test_data_loop_stashes_a_dirty_tree_before_it_touches_main():
    src = _src(DATA_RUN)
    stash = src.index("'stash', 'push', '-u'")
    checkout = src.index("$co = Invoke-Native 'git' @('checkout', 'main')")
    assert stash < checkout, (
        "the rescue has to come before git checkout main, which carries tracked "
        "modifications across and would publish them")


def test_data_loop_refuses_to_publish_if_it_cannot_stash():
    src = _src(DATA_RUN)
    i = src.index("'stash', 'push', '-u'")
    assert "Stop-Run" in src[i:i + 400]


def test_data_loop_rescue_is_labelled_so_it_can_be_found():
    assert "auto-rescue data-run" in _src(DATA_RUN)


# ---------------------------------------------------------------------------
# The sessions are told to checkpoint and to look for rescued work
# ---------------------------------------------------------------------------

def test_prompt_tells_sessions_to_commit_as_they_go():
    text = _src(PROMPT)
    assert "Commit as you go" in text
    assert "usage limit" in text


def test_prompt_tells_sessions_to_read_the_stash():
    text = _src(PROMPT)
    assert "git stash list" in text
    assert "auto-rescue" in text


# ---------------------------------------------------------------------------
# Behaviour: the exact git sequence, in a sandbox
# ---------------------------------------------------------------------------

def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([GIT, *args], cwd=cwd, capture_output=True, text=True, timeout=60)


@needs_git
def test_salvage_sequence_preserves_the_work_and_leaves_a_clean_main(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "generator.py").write_text("PALETTE = 'old'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")

    # The cut-off session: on its branch, work in the tree, nothing committed.
    _git(repo, "checkout", "-q", "-b", "nightly/2026-10-06")
    (repo / "generator.py").write_text("PALETTE = 'new, half-finished'\n", encoding="utf-8")
    (repo / "design.md").write_text("the argument\n", encoding="utf-8")

    # Exactly what the runner now does.
    assert _git(repo, "status", "--porcelain").stdout.strip(), "precondition: dirty"
    _git(repo, "add", "-A")
    assert _git(repo, "commit", "-q", "-m", "wip: salvaged").returncode == 0
    assert not _git(repo, "status", "--porcelain").stdout.strip(), "tree must be clean"
    assert _git(repo, "checkout", "main").returncode == 0

    # main is untouched and clean - nothing downstream can publish the work...
    assert (repo / "generator.py").read_text(encoding="utf-8") == "PALETTE = 'old'\n"
    assert not (repo / "design.md").exists()
    assert not _git(repo, "status", "--porcelain").stdout.strip()
    # ...and the work is all on the branch.
    shown = _git(repo, "show", "nightly/2026-10-06:generator.py").stdout
    assert "half-finished" in shown
    assert _git(repo, "show", "nightly/2026-10-06:design.md").stdout.strip() == "the argument"


@needs_git
def test_the_old_behaviour_leaks_a_tracked_edit_across_checkout(tmp_path: Path):
    """The defect, demonstrated: without a stash, `git checkout main` from a
    branch with an uncommitted tracked edit succeeds and keeps the edit - so a
    loop that then regenerates from the tree publishes it."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "generator.py").write_text("PALETTE = 'old'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-q", "-b", "nightly/2026-10-06")
    (repo / "generator.py").write_text("PALETTE = 'unreviewed'\n", encoding="utf-8")

    assert _git(repo, "checkout", "main").returncode == 0
    assert (repo / "generator.py").read_text(encoding="utf-8") == "PALETTE = 'unreviewed'\n"


@needs_git
def test_stash_rescue_makes_the_data_loop_start_from_the_committed_generator(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "T")
    (repo / "generator.py").write_text("PALETTE = 'old'\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-q", "-b", "nightly/2026-10-06")
    (repo / "generator.py").write_text("PALETTE = 'unreviewed'\n", encoding="utf-8")
    (repo / "notes.md").write_text("untracked work\n", encoding="utf-8")

    assert _git(repo, "stash", "push", "-u", "-m", "auto-rescue data-run 2026-10-07").returncode == 0
    assert _git(repo, "checkout", "main").returncode == 0

    assert (repo / "generator.py").read_text(encoding="utf-8") == "PALETTE = 'old'\n"
    assert not (repo / "notes.md").exists()
    assert "auto-rescue data-run" in _git(repo, "stash", "list").stdout
