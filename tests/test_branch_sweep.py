"""Merged `nightly/*` branches are swept from origin as well as locally.

Why this file exists
--------------------
Each code session creates `nightly/<date>`, pushes it to origin, and merges it
into main. The 2026-09-01 session fixed *local* branch accumulation, but nothing
ever deleted the branch on origin, so origin gained one dead ref per session
indefinitely. By 2026-09-03 there were **11**, `nightly/2026-08-10` through
`nightly/2026-09-02`, every one fully merged.

The 2026-09-02 evening session noticed and wrote "worth a `git push origin
--delete` sweep of anything `--merged main` on a future session" - i.e. left it
as manual work. That is the pattern `CLAUDE.md` rule 11 exists to stop. The
sweep now runs in `nightly-screener.ps1` right after the branch is created.

Why this is tested rather than eyeballed
----------------------------------------
The sweep runs `git push origin --delete` in a loop, unattended, against the
repository that serves the public site. The thing standing between it and
deleting live branches is the `--merged origin/main` filter, so that filter's
semantics are worth asserting directly rather than assuming: the first test
builds a real origin+clone sandbox with one merged and one unmerged branch and
checks git's own answer.

The remaining tests are static assertions on the runner. PowerShell execution is
blocked in the nightly sandbox, so the sweep cannot be driven end to end from a
session; what can be pinned is that the destructive call keeps its guards.
"""

import re
import subprocess
from pathlib import Path

import pytest

RUNNER = Path(__file__).resolve().parent.parent / "scripts" / "nightly-screener.ps1"
SOURCE = RUNNER.read_text(encoding="utf-8-sig")


def _git(cwd, *args):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"
    return r.stdout


@pytest.fixture
def sandbox(tmp_path):
    """A real origin + clone, with one merged and one unmerged nightly branch."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   capture_output=True, check=True)
    subprocess.run(["git", "clone", str(origin), str(work)],
                   capture_output=True, check=True)
    _git(work, "config", "user.email", "t@example.com")
    _git(work, "config", "user.name", "T")

    (work / "f.txt").write_text("base\n")
    _git(work, "add", "f.txt")
    _git(work, "commit", "-m", "base")
    _git(work, "push", "-u", "origin", "main")

    # A branch that gets merged into main - safe to delete.
    _git(work, "checkout", "-b", "nightly/2026-01-01")
    (work / "f.txt").write_text("merged work\n")
    _git(work, "commit", "-am", "merged work")
    _git(work, "push", "origin", "nightly/2026-01-01")
    _git(work, "checkout", "main")
    _git(work, "merge", "--no-ff", "-m", "merge", "nightly/2026-01-01")
    _git(work, "push", "origin", "main")

    # A branch with work that never landed - must NOT be deleted.
    _git(work, "checkout", "-b", "nightly/2026-01-02")
    (work / "g.txt").write_text("unmerged work\n")
    _git(work, "add", "g.txt")
    _git(work, "commit", "-m", "unmerged work")
    _git(work, "push", "origin", "nightly/2026-01-02")
    _git(work, "checkout", "main")

    _git(work, "fetch", "--prune", "origin")
    return work


def _sweep_list(work):
    """Exactly what the runner computes, in Python."""
    out = _git(work, "branch", "-r", "--merged", "origin/main")
    return [
        b.strip()[len("origin/"):]
        for b in out.splitlines()
        if b.strip().startswith("origin/nightly/")
    ]


def test_merged_filter_selects_the_merged_branch(sandbox):
    assert "nightly/2026-01-01" in _sweep_list(sandbox)


def test_merged_filter_protects_the_unmerged_branch(sandbox):
    """The load-bearing safety property. If this ever fails, the sweep would
    destroy work that never reached main."""
    assert "nightly/2026-01-02" not in _sweep_list(sandbox)


def test_deleting_a_merged_branch_discards_no_commits(sandbox):
    """Deleting the ref is safe precisely because main still reaches the tip."""
    tip = _git(sandbox, "rev-parse", "origin/nightly/2026-01-01").strip()
    _git(sandbox, "push", "origin", "--delete", "nightly/2026-01-01")
    _git(sandbox, "fetch", "--prune", "origin")

    assert "nightly/2026-01-01" not in _sweep_list(sandbox)
    # The commit itself is still reachable from main.
    r = subprocess.run(["git", "merge-base", "--is-ancestor", tip, "origin/main"],
                       cwd=sandbox, capture_output=True)
    assert r.returncode == 0, "the swept branch's commit is no longer on main"


def test_runner_sweeps_remote_branches_at_all():
    assert "push', 'origin', '--delete'" in SOURCE or \
           "'push', 'origin', '--delete'" in SOURCE, \
           "nightly-screener.ps1 no longer deletes merged remote branches"


def test_remote_sweep_is_guarded_by_the_merged_filter():
    """An unguarded delete loop over origin/nightly/* would be destructive."""
    assert "'branch', '-r', '--merged', 'origin/main'" in SOURCE, (
        "the remote sweep no longer filters on --merged origin/main; it could "
        "delete branches whose work never reached main"
    )


def test_remote_sweep_skips_the_branch_this_run_is_using():
    marker = "Sweep stray merged nightly/* branches on the REMOTE"
    assert marker in SOURCE, "the remote branch sweep is gone from the runner"
    block = SOURCE.split(marker)[1].split("--- Prompt ---")[0]
    assert re.search(r"Where-Object\s*\{\s*\$_\s*-ne\s*\$Branch\s*\}", block), (
        "the remote sweep no longer excludes $Branch, so a same-day rerun could "
        "delete the branch it is about to push"
    )


def test_remote_sweep_failure_is_a_warning_not_fatal():
    """Dead refs on origin are untidy, not dangerous. Killing the session over
    one would turn a cosmetic problem into a lost day."""
    marker = "Sweep stray merged nightly/* branches on the REMOTE"
    assert marker in SOURCE, "the remote branch sweep is gone from the runner"
    block = SOURCE.split(marker)[1].split("--- Prompt ---")[0]
    assert "Stop-Run" not in block, "a failed branch delete must not abort the run"
    assert "WARN: could not delete remote branch" in block


def test_local_sweep_uses_the_safe_delete_form():
    """'git branch -D' force-deletes regardless of merge status; '-d' refuses.

    The sweep walks branches this run did not create, so it must use '-d' and
    let git veto anything unmerged.
    """
    block = SOURCE.split("Sweep stray merged nightly/* branches ---")[1]
    block = block.split("--- Branch ---")[0]
    assert "'branch', '-d'" in block
    assert "'branch', '-D'" not in block


def test_force_delete_is_only_ever_used_on_this_runs_own_branch():
    """`-D` is legitimate for $Branch - the run created it from main moments
    earlier, so there is nothing to lose - and legitimate nowhere else."""
    for m in re.finditer(r"@\(\s*'branch',\s*'-D',\s*([^\)]+)\)", SOURCE):
        target = m.group(1).strip()
        assert target == "$Branch", (
            f"force-delete targets {target}, not the run's own branch; it could "
            "discard unmerged work"
        )


def test_no_script_deletes_main():
    for line in SOURCE.splitlines():
        if "--delete" in line:
            assert "main" not in line, f"a delete targets main: {line.strip()}"
