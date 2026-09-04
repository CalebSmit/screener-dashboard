"""The ship gates must run *before* anything reaches `main`, not after.

WHY THIS FILE EXISTS - 2026-09-04 retrospective.

`CLAUDE.md` rule 1: "You may only push to `main` when all of them pass." Until
today the routine did not work that way. `prompts/nightly.md` told the session
to merge and push `main` itself as its final step, and `nightly-screener.ps1`
then re-verified the gates *afterwards*. The gates were an audit, not a
precondition.

That is not theoretical. On 2026-09-02 the session pushed `origin/main` at
06:24:22; the wrapper's independent test run failed two minutes later; the
recovery path logged "main is untouched", which was false. The evening session
built `scripts/revert-bad-merge.ps1` to undo an already-published push - a
mitigation for a hole that should not exist.

The fix is ordering, not more recovery machinery: the session pushes its branch
and stops, and the runner merges only after all four gates pass. Every
successful run in `logs/` shows "HEAD is on 'main'", so the runner's own merge
path had never executed in production; these tests exercise its git sequence
against a real origin+clone sandbox before it becomes load-bearing.

`revert-bad-merge.ps1` stays. It is now the second line of defence rather than
the first.
"""

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "nightly-screener.ps1"
PROMPT = ROOT / "prompts" / "nightly.md"
SOURCE = RUNNER.read_text(encoding="utf-8-sig")
CODE = re.sub(r"(?m)#.*$", "", re.sub(r"<#.*?#>", "", SOURCE, flags=re.S))


def _git(cwd, *args, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=120)
    if check:
        assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"
    return r


@pytest.fixture
def sandbox(tmp_path):
    """A real origin + clone with a nightly branch holding unmerged work."""
    origin = tmp_path / "origin.git"
    work = tmp_path / "work"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(origin)],
                   capture_output=True, check=True, timeout=120)
    subprocess.run(["git", "clone", str(origin), str(work)],
                   capture_output=True, check=True, timeout=120)
    _git(work, "config", "user.email", "t@example.com")
    _git(work, "config", "user.name", "T")
    (work / "index.html").write_text("<html>base</html>\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "base")
    _git(work, "push", "-u", "origin", "main")
    base = _git(work, "rev-parse", "HEAD").stdout.strip()

    _git(work, "checkout", "-b", "nightly/2026-09-04")
    (work / "feature.py").write_text("shipped\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "the session's work")
    _git(work, "push", "-u", "origin", "nightly/2026-09-04")
    return origin, work, base


# ---------------------------------------------------------------------------
# The runner's merge path, which no production run has ever taken.
# ---------------------------------------------------------------------------


def test_the_runner_merge_sequence_lands_the_branch_on_main(sandbox):
    """The git sequence at nightly-screener.ps1's "Merging to main" block."""
    origin, work, base = sandbox

    _git(work, "checkout", "main")
    _git(work, "merge", "--no-ff", "nightly/2026-09-04", "-m", "nightly 2026-09-04 - RETROSPECTIVE")
    _git(work, "push", "origin", "main")

    remote = subprocess.run(
        ["git", "--git-dir", str(origin), "show", "main:feature.py"],
        capture_output=True, text=True, timeout=120,
    )
    assert remote.stdout.strip() == "shipped"
    assert _git(work, "status", "--porcelain").stdout.strip() == ""
    assert _git(work, "rev-parse", "main").stdout.strip() != base


def test_a_conflicting_merge_leaves_main_exactly_where_it_was(sandbox):
    """The failure branch: `merge --abort`, push the branch, main untouched.

    This is the outcome that makes deferring the merge to the runner safe. The
    worst case is a delayed day with the work preserved on origin - not a
    published tree that failed its gates.
    """
    origin, work, base = sandbox

    # main moves under the branch, in the same file, so the merge conflicts.
    _git(work, "checkout", "main")
    (work / "feature.py").write_text("something else\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "conflicting change on main")
    _git(work, "push", "origin", "main")
    before = _git(work, "rev-parse", "main").stdout.strip()

    merged = _git(work, "merge", "--no-ff", "nightly/2026-09-04", "-m", "nightly", check=False)
    assert merged.returncode != 0
    _git(work, "merge", "--abort")

    assert _git(work, "rev-parse", "main").stdout.strip() == before
    assert _git(work, "status", "--porcelain").stdout.strip() == ""
    on_origin = subprocess.run(
        ["git", "--git-dir", str(origin), "rev-parse", "nightly/2026-09-04"],
        capture_output=True, text=True, timeout=120,
    )
    assert on_origin.returncode == 0, "the session's work must survive on origin"


# ---------------------------------------------------------------------------
# Ordering, asserted on the runner's own source.
# ---------------------------------------------------------------------------


def test_nothing_reaches_main_before_the_gates_have_decided():
    """`push origin main` appears only after the gate-failure early return."""
    gate_failure = CODE.index("SHIP GATES FAILED")
    merging = CODE.index("All gates passed. Merging to main.")
    push = CODE.index("'push', 'origin', 'main'")
    assert gate_failure < merging < push, (
        "the push to main must sit downstream of the gate decision"
    )


def test_the_gate_failure_path_still_stops_the_run():
    """A failed gate must exit non-zero, not fall through to the merge."""
    tail = CODE[CODE.index("SHIP GATES FAILED"):CODE.index("All gates passed")]
    assert "Stop-Run" in tail
    assert "revert-bad-merge.ps1" in tail, (
        "the second line of defence stays: a session may still push by mistake"
    )


def test_the_merge_is_no_ff_and_aborts_cleanly():
    assert "'merge', '--no-ff', $Branch" in CODE
    block = CODE[CODE.index("All gates passed"):]
    assert "'merge', '--abort'" in block
    assert "Merge conflict. Left on $Branch; main untouched." in SOURCE


def test_the_rollback_tag_is_written_only_after_a_successful_push():
    """The tag is the documented rollback point (ROLLBACK.md); it must not
    mark a commit that never reached origin."""
    push = CODE.index("'push', 'origin', 'main'")
    tag = CODE.index("'tag', '-a', $tag")
    assert push < tag


# ---------------------------------------------------------------------------
# The prompt must not send the session round the gates.
# ---------------------------------------------------------------------------


def test_the_prompt_does_not_tell_the_session_to_push_main():
    # Imperative forms only. The prompt necessarily *mentions* pushing main in
    # order to forbid it, so a bare "push `main`" search flags its own rule.
    text = PROMPT.read_text(encoding="utf-8")
    banned = [
        "merge it into `main`",
        "and push `main`",
        "then push `main`",
        "merge to `main` yourself",
    ]
    found = [p for p in banned if p in text]
    assert not found, (
        f"prompts/nightly.md still instructs the session to publish main ({found}). "
        f"The runner merges, after the gates pass."
    )


def test_the_prompt_tells_the_session_to_stop_at_its_branch():
    text = PROMPT.read_text(encoding="utf-8")
    assert "push the branch" in text
    assert "does not merge" in text or "do not merge" in text.lower()
