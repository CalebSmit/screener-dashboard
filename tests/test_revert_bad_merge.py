"""origin/main must actually be undone when the wrapper's independent
ship-gate re-check catches something the session's own check missed.

WHY THIS EXISTS - 2026-09-02.

`prompts/nightly.md` has the Claude Code session merge and push `main` itself,
as its own final step, before `nightly-screener.ps1`'s independent
re-verification ever runs (that re-run is the "trust but verify" layer -
CLAUDE.md's ship-gate table: "The runner script re-checks them independently
and will refuse the push if you got it wrong"). On 2026-09-02 the session's
own gate check passed and it pushed a merge to origin/main at 06:24:22; the
wrapper's fresh test run at 06:26:53 hit a flaky `test_parquet_roundtrip`
failure. The wrapper's recovery path reset the LOCAL copy of main and logged
"main is untouched" - but origin/main, the branch GitHub Pages actually
serves, had already moved. Local reset cannot undo a push it did not make.
Nothing was actually wrong with the site that morning (the failure turned out
to be a flake, confirmed by re-running the same suite clean afterward), but
the safety net had a real hole exactly where "ship gates are absolute" is
supposed to hold.

scripts/revert-bad-merge.ps1 is the fix: check what origin/main actually
carries, and if it is ahead of the last known-good commit, revert it there -
a new commit, never a rewrite (CLAUDE.md rule 2) - verifying the reverted tree
is byte-identical to the last known-good state before ever pushing.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "revert-bad-merge.ps1"
NIGHTLY = ROOT / "scripts" / "nightly-screener.ps1"

POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
needs_powershell = pytest.mark.skipif(POWERSHELL is None, reason="no PowerShell on this platform")
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="no git on this platform")


# ---------------------------------------------------------------------------
# Static: the wiring is present, and the old unconditional claim is gone
# ---------------------------------------------------------------------------

def test_script_exists():
    assert SCRIPT.exists()


def test_nightly_calls_the_revert_script_on_gate_failure():
    src = NIGHTLY.read_text(encoding="utf-8-sig")
    i = src.index("SHIP GATES FAILED")
    block = src[i:i + 2500]
    assert "revert-bad-merge.ps1" in block
    assert "-BaseSha" in block and "-RepoPath" in block


def test_the_unconditional_false_claim_is_gone():
    """The exact bug: this line used to run no matter what actually happened
    on origin. It must not appear unconditionally attached to a bare push of
    the inspection branch any more."""
    src = NIGHTLY.read_text(encoding="utf-8-sig")
    i = src.index("SHIP GATES FAILED")
    block = src[i:i + 2500]
    # The old text is fine to still appear as part of a conditional success
    # message, but the specific pattern - printed right after pushing the
    # branch, with nothing checking origin first - must be gone.
    push_pos = block.index("push', '-u', 'origin', $Branch")
    after_push = block[push_pos:push_pos + 300]
    assert "revert-bad-merge" in after_push or "$rv" in block[push_pos:], (
        "origin/main must be checked before claiming it is untouched"
    )


# ---------------------------------------------------------------------------
# Behavioural: real git operations against a real origin, in tmp_path
# ---------------------------------------------------------------------------

def _run_git(args, cwd, check=True):
    r = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True, timeout=30)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {args} failed: {r.stdout}\n{r.stderr}")
    return r


def _make_origin_and_clone(tmp_path: Path) -> tuple[Path, Path]:
    origin = tmp_path / "origin.git"
    clone = tmp_path / "clone"
    _run_git(["init", "-q", "--bare", "-b", "main", str(origin)], cwd=tmp_path)
    _run_git(["init", "-q", "-b", "main", str(clone)], cwd=tmp_path)
    _run_git(["config", "user.email", "test@test.com"], cwd=clone)
    _run_git(["config", "user.name", "Test"], cwd=clone)
    _run_git(["remote", "add", "origin", str(origin)], cwd=clone)
    (clone / "f.txt").write_text("base\n", encoding="utf-8")
    _run_git(["add", "f.txt"], cwd=clone)
    _run_git(["commit", "-q", "-m", "base commit"], cwd=clone)
    _run_git(["push", "-q", "-u", "origin", "main"], cwd=clone)
    return origin, clone


def _run_revert_script(base_sha: str, repo_path: Path):
    r = subprocess.run(
        [POWERSHELL, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
         "-BaseSha", base_sha, "-RepoPath", str(repo_path)],
        capture_output=True, text=True, timeout=60,
    )
    return r


@needs_powershell
@needs_git
def test_noop_when_origin_already_matches_base(tmp_path):
    """The common case: the session's own gate check caught the problem and
    origin/main was never touched. Nothing should happen, and nothing should
    be pushed."""
    origin, clone = _make_origin_and_clone(tmp_path)
    base = _run_git(["rev-parse", "HEAD"], cwd=clone).stdout.strip()

    r = _run_revert_script(base, clone)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "nothing to revert" in r.stdout.lower()

    origin_head = _run_git(["log", "--oneline", "-1"], cwd=origin).stdout.strip()
    assert origin_head.startswith(_run_git(["rev-parse", "--short", base], cwd=clone).stdout.strip())


@needs_powershell
@needs_git
def test_reverts_a_merge_commit_that_reached_origin(tmp_path):
    """The exact 2026-09-02 shape: a merge commit landed on origin/main
    before the failure was caught."""
    origin, clone = _make_origin_and_clone(tmp_path)
    base = _run_git(["rev-parse", "HEAD"], cwd=clone).stdout.strip()

    _run_git(["checkout", "-q", "-b", "feature"], cwd=clone)
    (clone / "f.txt").write_text("base\nfeature work\n", encoding="utf-8")
    _run_git(["add", "f.txt"], cwd=clone)
    _run_git(["commit", "-q", "-m", "feature: the bad commit"], cwd=clone)
    _run_git(["checkout", "-q", "main"], cwd=clone)
    _run_git(["merge", "-q", "--no-ff", "feature", "-m", "merge: feature (bad)"], cwd=clone)
    _run_git(["push", "-q", "origin", "main"], cwd=clone)

    # Simulate the wrapper's existing local-only recovery: local main moves
    # back to base, but origin/main is still on the bad merge.
    _run_git(["checkout", "-q", "-B", "main", base], cwd=clone)

    r = _run_revert_script(base, clone)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "pushed a verified revert" in r.stdout.lower()

    _run_git(["fetch", "-q", "origin", "main"], cwd=clone)
    content = (subprocess.run(["git", "show", "origin/main:f.txt"], cwd=clone,
                               capture_output=True, text=True).stdout)
    assert content.strip() == "base", "reverted content must match the pre-merge file exactly"

    diff = subprocess.run(["git", "diff", "--quiet", base, "origin/main"], cwd=clone)
    assert diff.returncode == 0, "reverted tree must be byte-identical to the last known-good commit"


@needs_powershell
@needs_git
def test_reverts_a_fast_forward_that_reached_origin(tmp_path):
    """No merge commit at all - a session that fast-forwarded main still
    needs its commits undone the same way."""
    origin, clone = _make_origin_and_clone(tmp_path)
    base = _run_git(["rev-parse", "HEAD"], cwd=clone).stdout.strip()

    (clone / "f.txt").write_text("base\ncommit A\n", encoding="utf-8")
    _run_git(["add", "f.txt"], cwd=clone)
    _run_git(["commit", "-q", "-m", "commit A (bad)"], cwd=clone)
    _run_git(["push", "-q", "origin", "main"], cwd=clone)
    _run_git(["checkout", "-q", "-B", "main", base], cwd=clone)

    r = _run_revert_script(base, clone)
    assert r.returncode == 0, r.stdout + r.stderr

    _run_git(["fetch", "-q", "origin", "main"], cwd=clone)
    diff = subprocess.run(["git", "diff", "--quiet", base, "origin/main"], cwd=clone)
    assert diff.returncode == 0


@needs_powershell
@needs_git
def test_callers_branch_and_temp_refs_are_cleaned_up(tmp_path):
    """The caller's repo must come back exactly as it was handed over -
    original branch checked out, no stray revert-bad-merge-* branch left."""
    origin, clone = _make_origin_and_clone(tmp_path)
    base = _run_git(["rev-parse", "HEAD"], cwd=clone).stdout.strip()

    (clone / "f.txt").write_text("base\ncommit A\n", encoding="utf-8")
    _run_git(["add", "f.txt"], cwd=clone)
    _run_git(["commit", "-q", "-m", "commit A (bad)"], cwd=clone)
    _run_git(["push", "-q", "origin", "main"], cwd=clone)
    _run_git(["checkout", "-q", "-B", "main", base], cwd=clone)

    r = _run_revert_script(base, clone)
    assert r.returncode == 0, r.stdout + r.stderr

    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=clone).stdout.strip()
    assert branch == "main"

    all_branches = _run_git(["branch"], cwd=clone).stdout
    assert "revert-bad-merge-" not in all_branches
