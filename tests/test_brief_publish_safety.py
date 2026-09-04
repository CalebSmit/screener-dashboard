"""The morning brief must never carry anything else onto `main`.

WHY THIS EXISTS - 2026-09-04 retrospective.

Both runners published the brief from a `finally` block with:

    git push origin HEAD:main

On the happy path HEAD is `main`, so that was correct, and it is what ran on
every successful day. On a *ship-gate failure* it is not. Then HEAD is the
nightly branch carrying the work the gates just refused, and that command
fast-forwards `origin/main` onto every commit on it - publishing exactly that
work to the branch GitHub Pages serves, from a `finally` block, after the gates
said no. `prompts/nightly.md` explicitly tells a session that fails its own
gates to leave the work on the branch, which is the shape that publishes.

The first test below drives the old command against a real origin+clone sandbox
and asserts the broken file arrives on `origin/main`. Everything after it
asserts the replacement (`scripts/publish-brief.ps1`) cannot do that, because it
builds a single-file commit on top of `origin/main` with plumbing and never
pushes a local ref.

These are unattended-infrastructure tests: they drive real git and real
PowerShell rather than mocking either, on the 2026-08-29 / 2026-09-02 precedent.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
MODULE = SCRIPTS_DIR / "publish-brief.ps1"
NIGHTLY = SCRIPTS_DIR / "nightly-screener.ps1"
DATARUN = SCRIPTS_DIR / "data-run.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
GIT = shutil.which("git")

requires_shell = pytest.mark.skipif(
    POWERSHELL is None or GIT is None, reason="needs git and PowerShell"
)


def git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(
        [GIT, "-C", str(repo), *args], capture_output=True, text=True, timeout=120
    )
    if check and r.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed:\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def bare(repo: Path, *args: str) -> str:
    r = subprocess.run(
        [GIT, "--git-dir", str(repo), *args], capture_output=True, text=True, timeout=120
    )
    return r.stdout.strip() if r.returncode == 0 else f"<<error:{r.stderr.strip()}>>"


@pytest.fixture
def sandbox(tmp_path: Path):
    """A bare origin plus a clone holding one commit on `main`.

    Mirrors the real repo closely enough for the refspec semantics under test:
    `main` is the published branch, and `MORNING_BRIEF.md` is tracked on it.
    """
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run([GIT, "init", "-q", "--bare", str(origin)], check=True, timeout=120)
    subprocess.run(
        [GIT, "clone", "-q", str(origin), str(work)], check=True, timeout=120
    )
    git(work, "config", "user.email", "runner@example.invalid")
    git(work, "config", "user.name", "Runner")
    (work / "MORNING_BRIEF.md").write_text("day one\n", encoding="utf-8")
    (work / "index.html").write_text("<html>live site</html>\n", encoding="utf-8")
    git(work, "add", "-A")
    git(work, "commit", "-qm", "base")
    git(work, "branch", "-M", "main")
    git(work, "push", "-q", "-u", "origin", "main")
    # `git init --bare` may leave HEAD on `master`; the real origin resolves
    # origin/HEAD -> origin/main, and a second clone must land on main too.
    bare(origin, "symbolic-ref", "HEAD", "refs/heads/main")
    return origin, work, git(work, "rev-parse", "HEAD")


def failing_branch(work: Path) -> None:
    """Put the clone in the state a ship-gate failure leaves it in.

    A nightly branch is checked out, it carries work that failed the gates, and
    `write_brief.py` has just rewritten MORNING_BRIEF.md on top of it.
    """
    git(work, "checkout", "-qb", "nightly/2026-09-04")
    (work / "index.html").write_text("BROKEN - gates refused this\n", encoding="utf-8")
    git(work, "add", "-A")
    git(work, "commit", "-qm", "work the ship gates refused")
    (work / "MORNING_BRIEF.md").write_text("day two\n", encoding="utf-8")


def publish(work: Path, label: str = "code session 2026-09-04") -> subprocess.CompletedProcess:
    cmd = (
        f". '{MODULE.as_posix()}'; "
        f"$ok = Publish-BriefToMain -RepoPath '{work.as_posix()}' -Label '{label}'; "
        "if (-not $ok) { exit 3 }"
    )
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", cmd],
        capture_output=True,
        text=True,
        timeout=300,
    )


# ---------------------------------------------------------------------------
# 1. The defect, demonstrated against the old command.
# ---------------------------------------------------------------------------


@requires_shell
def test_pushing_head_to_main_publishes_the_whole_branch(sandbox):
    """`git push origin HEAD:main` from a nightly branch publishes its work.

    This is the old Publish-Brief, run in the state a gate failure leaves the
    repo in. It is here so the replacement is measured against a demonstrated
    failure rather than an argument.
    """
    origin, work, base = sandbox
    failing_branch(work)
    git(work, "add", "--", "MORNING_BRIEF.md")
    git(work, "commit", "-qm", "brief: code session 2026-09-04")

    git(work, "push", "origin", "HEAD:main")

    assert bare(origin, "show", "main:index.html") == "BROKEN - gates refused this", (
        "the old command put gate-refused work on the branch Pages serves"
    )
    assert bare(origin, "rev-parse", "main") != base


# ---------------------------------------------------------------------------
# 2. The replacement cannot.
# ---------------------------------------------------------------------------


@requires_shell
def test_publish_from_a_failed_branch_carries_only_the_brief(sandbox):
    origin, work, base = sandbox
    failing_branch(work)

    r = publish(work)
    assert r.returncode == 0, f"{r.stdout}{r.stderr}"

    assert bare(origin, "show", "main:MORNING_BRIEF.md") == "day two"
    assert bare(origin, "show", "main:index.html") == "<html>live site</html>", (
        "gate-refused work reached origin/main"
    )


@requires_shell
def test_the_published_commit_touches_exactly_one_path(sandbox):
    origin, work, base = sandbox
    failing_branch(work)

    assert publish(work).returncode == 0

    head = bare(origin, "rev-parse", "main")
    changed = bare(origin, "diff-tree", "--no-commit-id", "--name-only", "-r", head)
    assert changed.split() == ["MORNING_BRIEF.md"]
    parents = bare(origin, "rev-list", "--parents", "-n", "1", head).split()[1:]
    assert parents == [base], "the brief commit must sit directly on the old origin/main"


@requires_shell
def test_the_refused_commit_is_not_even_reachable_from_main(sandbox):
    """Not merely absent from the tree - absent from main's history.

    A merge would leave the correct files at the tip while still making the
    refused commit an ancestor of the branch that gets rolled back to.
    """
    origin, work, base = sandbox
    failing_branch(work)
    refused = git(work, "rev-parse", "HEAD")

    assert publish(work).returncode == 0

    reachable = bare(origin, "rev-list", "main")
    assert refused not in reachable.split()


@requires_shell
def test_happy_path_publishes_and_fast_forwards_local_main(sandbox):
    """On the normal path the runner is on main and the tree must end clean.

    The next run opens with `git checkout main` and `git pull`; a tracked file
    left modified here that the pull also changes aborts both, and that costs
    the following day.
    """
    origin, work, base = sandbox
    (work / "MORNING_BRIEF.md").write_text("day two\n", encoding="utf-8")

    assert publish(work, "data run 2026-09-04").returncode == 0

    assert bare(origin, "show", "main:MORNING_BRIEF.md") == "day two"
    assert git(work, "rev-parse", "main") == bare(origin, "rev-parse", "main")
    assert git(work, "status", "--porcelain") == "", "working tree left dirty"


@requires_shell
def test_publishing_from_a_branch_leaves_the_tree_clean(sandbox):
    origin, work, base = sandbox
    failing_branch(work)

    assert publish(work).returncode == 0

    assert git(work, "status", "--porcelain") == "", (
        "MORNING_BRIEF.md left modified or staged after publishing"
    )
    assert git(work, "rev-parse", "--abbrev-ref", "HEAD") == "nightly/2026-09-04", (
        "publishing must not move the runner off its branch"
    )


@requires_shell
def test_an_unchanged_brief_adds_no_commit(sandbox):
    origin, work, base = sandbox
    failing_branch(work)
    (work / "MORNING_BRIEF.md").write_text("day one\n", encoding="utf-8")

    r = publish(work)
    assert r.returncode == 0, f"{r.stdout}{r.stderr}"
    assert bare(origin, "rev-parse", "main") == base


@requires_shell
def test_it_retries_when_origin_main_moved_underneath_it(sandbox, tmp_path):
    """The data loop pushes to main while a code session may be finishing.

    The old code committed locally and pushed HEAD, so a moved origin/main was
    a rejected non-fast-forward and the brief - the watchdog heartbeat - was
    silently dropped. Rebuilding on the *current* origin/main and retrying is
    what makes the heartbeat survive that.
    """
    origin, work, base = sandbox
    failing_branch(work)

    other = tmp_path / "other"
    subprocess.run([GIT, "clone", "-q", str(origin), str(other)], check=True, timeout=120)
    git(other, "config", "user.email", "other@example.invalid")
    git(other, "config", "user.name", "Other")
    (other / "dashboard_data.js").write_text("window.SCREENER_DATA={}\n", encoding="utf-8")
    git(other, "add", "-A")
    git(other, "commit", "-qm", "data: screener run")
    git(other, "push", "-q", "origin", "main")
    moved = bare(origin, "rev-parse", "main")

    assert publish(work).returncode == 0

    assert bare(origin, "show", "main:MORNING_BRIEF.md") == "day two"
    assert bare(origin, "show", "main:dashboard_data.js") == "window.SCREENER_DATA={}", (
        "the concurrent data run's commit was clobbered"
    )
    parents = bare(origin, "rev-list", "--parents", "-n", "1", bare(origin, "rev-parse", "main"))
    assert parents.split()[1:] == [moved]


@requires_shell
def test_it_does_not_disturb_the_repository_index(sandbox):
    """The brief is published from `finally`, alongside whatever else is staged.

    Building the tree in a throwaway index is what keeps that true; staging into
    the real one would leave the next `git status` - and ship gate 4 - dirty.
    """
    origin, work, base = sandbox
    (work / "notes.txt").write_text("mid-run\n", encoding="utf-8")
    git(work, "add", "--", "notes.txt")
    before = git(work, "diff", "--cached", "--name-only")
    (work / "MORNING_BRIEF.md").write_text("day two\n", encoding="utf-8")

    publish(work)

    assert git(work, "diff", "--cached", "--name-only") == before == "notes.txt"


# ---------------------------------------------------------------------------
# 3. The old command must not come back.
# ---------------------------------------------------------------------------


def code_of(path: Path) -> str:
    """Source with comments removed.

    These files explain the defect at length in their comments, so a plain
    substring search for the old command matches the explanation of why it is
    gone. Only executable text counts.
    """
    src = re.sub(r"<#.*?#>", "", path.read_text(encoding="utf-8-sig"), flags=re.S)
    return re.sub(r"(?m)#.*$", "", src)


@pytest.mark.parametrize("path", [NIGHTLY, DATARUN], ids=lambda p: p.name)
def test_no_runner_pushes_head_to_main(path: Path):
    src = code_of(path)
    assert "HEAD:main" not in src, (
        f"{path.name} pushes local HEAD to main. Publishing must go through "
        f"Publish-BriefToMain, which can only ever move MORNING_BRIEF.md."
    )


@pytest.mark.parametrize("path", [NIGHTLY, DATARUN], ids=lambda p: p.name)
def test_both_runners_use_the_shared_publisher(path: Path):
    src = path.read_text(encoding="utf-8-sig")
    assert "publish-brief.ps1" in src, f"{path.name} does not dot-source the publisher"
    assert "Publish-BriefToMain" in src, f"{path.name} does not call the publisher"


def test_the_publisher_never_names_a_local_ref_in_a_push():
    """The safety property is structural: the refspec's left side is an object.

    `git push origin <sha>:refs/heads/main` cannot pick up a branch's other
    commits the way `HEAD:main` does.
    """
    src = code_of(MODULE)
    assert "${commit}:refs/heads/main" in src
    assert "HEAD:main" not in src
