"""The two loops must not run git against the same working tree at once.

WHY THIS EXISTS - 2026-08-29.

`scripts/register-tasks.ps1` gives both scheduled tasks an at-logon catch-up
trigger. Until today both used a PT3M delay, so on the first logon of the day
they started in the same second. On 2026-08-29 they did:

    logs/datarun-2026-08-29_121115.log   [12:11:15] === Data loop 2026-08-29 ===
    logs/nightly-2026-08-29_121115.log   [12:11:15] === Code loop 2026-08-29 ===

One second later the data loop ran `git checkout main` while the code loop was
inside Restore-Artifacts running `git status`, which takes .git/index.lock to
refresh the index. git exits 128. The data loop treated that as fatal, logged
"Could not check out main." with none of git's own explanation, and stopped
before running the screener.

The reproduction is in the changelog: with .git/index.lock present, `git
checkout main` on an otherwise clean repo exits 128 rather than 0.

Each script had a single-instance lock of its own; neither excluded the other.
These tests cover the shared lock that now does, and the two smaller defects
found alongside it: a swallowed error message and a fatal-on-first-failure
checkout.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
REPO_LOCK = SCRIPTS / "repo-lock.ps1"
DATA_RUN = SCRIPTS / "data-run.ps1"
NIGHTLY = SCRIPTS / "nightly-screener.ps1"
REGISTER = SCRIPTS / "register-tasks.ps1"

POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")
needs_powershell = pytest.mark.skipif(
    POWERSHELL is None, reason="no PowerShell on this platform"
)


def _src(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Static: the wiring is present and in the right order
# ---------------------------------------------------------------------------


def test_repo_lock_module_exists():
    assert REPO_LOCK.exists(), "scripts/repo-lock.ps1 is the shared lock"
    src = _src(REPO_LOCK)
    for fn in ("Enter-RepoLock", "Exit-RepoLock"):
        assert re.search(rf"(?im)^\s*function\s+{fn}\b", src), f"{fn} not defined"


@pytest.mark.parametrize("path", [DATA_RUN, NIGHTLY], ids=lambda p: p.name)
def test_both_loops_source_and_take_the_lock(path: Path):
    src = _src(path)
    assert "repo-lock.ps1" in src, f"{path.name} does not dot-source the shared lock"
    assert "Enter-RepoLock" in src, f"{path.name} never acquires the repo lock"
    assert "Exit-RepoLock" in src, f"{path.name} never releases the repo lock"


@pytest.mark.parametrize("path", [DATA_RUN, NIGHTLY], ids=lambda p: p.name)
def test_lock_is_taken_before_the_first_git_command(path: Path):
    """Acquiring after the first git call would leave the race wide open.

    The 2026-08-29 collision was on the *first* git command each script runs -
    `git checkout main` in the data loop, `git status` (inside
    Restore-Artifacts) in the code loop.
    """
    src = _src(path)
    acquire = src.index("Enter-RepoLock -LogDir")

    # First real git invocation, ignoring the helper definitions above it.
    git_calls = [m.start() for m in re.finditer(r"Invoke-Native 'git'", src)]
    body_start = src.index("\ntry {")
    in_body = [i for i in git_calls if i > body_start]
    assert in_body, f"{path.name}: no git calls found after the try block"

    if path is NIGHTLY:
        # Restore-Artifacts is a helper defined near the top; what matters is
        # where it is *called*.
        first_git = src.index("Restore-Artifacts 'pre-existing'")
    else:
        first_git = in_body[0]

    assert acquire < first_git, (
        f"{path.name} runs git before taking the repo lock, so the two loops "
        f"can still collide at logon"
    )


@pytest.mark.parametrize("path", [DATA_RUN, NIGHTLY], ids=lambda p: p.name)
def test_lock_is_released_after_the_brief_not_before(path: Path):
    """Publish-Brief runs git add/commit/push from `finally`.

    Releasing the lock before it would put the two loops back in the same
    window the lock exists to close.
    """
    src = _src(path)
    tail = src[src.rindex("finally {"):]
    assert "Exit-RepoLock" in tail, f"{path.name}: lock not released in finally"
    assert tail.index("Publish-Brief") < tail.index("Exit-RepoLock"), (
        f"{path.name} releases the repo lock before Publish-Brief runs git"
    )


def test_checkout_failure_reports_gits_own_words():
    """"Could not check out main." alone cost a reproduction to diagnose."""
    src = _src(DATA_RUN)
    block = src[src.index("$co = Invoke-Native 'git' @('checkout', 'main')"):]
    block = block[: block.index("Could not check out main.") + 40]
    assert "Write-NativeOutput $co 'ERROR'" in block, (
        "the fatal checkout path discards git's stderr, so the log cannot say "
        "why it failed"
    )


def test_checkout_retries_before_giving_up_the_day():
    """A momentarily-held index lock must not cost the whole data run."""
    src = _src(DATA_RUN)
    idx = src.index("$co = Invoke-Native 'git' @('checkout', 'main')")
    window = src[max(0, idx - 600): idx]
    assert re.search(r"for \(\$attempt = 1; \$attempt -le \d+;", window), (
        "git checkout main is still fatal on its first failure"
    )


def test_logon_delays_are_staggered():
    """Identical PT3M delays are what put both loops in the same second."""
    src = _src(REGISTER)
    delays = re.findall(r"LogonDelay\s*=\s*'(PT\d+M)'", src)
    assert len(delays) == 2, f"expected one logon delay per task, got {delays}"
    assert len(set(delays)) == 2, (
        f"both tasks still share a logon delay ({delays[0]}), so they start together"
    )
    assert "$logon.Delay = $s.LogonDelay" in src, (
        "the per-task delay is declared but not applied to the trigger"
    )


# ---------------------------------------------------------------------------
# Behavioural: run real PowerShell processes against the real lock
# ---------------------------------------------------------------------------


def _ps(script: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(script)],
        capture_output=True, text=True, timeout=120,
    )


def _write(path: Path, body: str) -> Path:
    path.write_text(f". '{REPO_LOCK.as_posix()}'\n{body}\n", encoding="ascii")
    return path


@needs_powershell
def test_second_loop_cannot_take_a_held_lock(tmp_path: Path):
    """The core guarantee: two processes, one lock."""
    holder = _write(tmp_path / "holder.ps1", f"""
$p = Enter-RepoLock -LogDir '{tmp_path.as_posix()}' -Holder 'data loop' -PollSeconds 1
if (-not $p) {{ exit 3 }}
Set-Content -Path '{tmp_path.as_posix()}/acquired' -Value $PID
Start-Sleep -Seconds 20
Exit-RepoLock $p
""")
    a = subprocess.Popen(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
         "-File", str(holder)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.time() + 60
        while not (tmp_path / "acquired").exists():
            assert a.poll() is None, "holder exited before acquiring"
            assert time.time() < deadline, "holder never acquired the lock"
            time.sleep(0.2)

        contender = _write(tmp_path / "contender.ps1", f"""
$p = Enter-RepoLock -LogDir '{tmp_path.as_posix()}' -Holder 'code loop' -MaxWaitMinutes 0 -PollSeconds 1
if ($p) {{ Write-Host 'GOT-IT'; exit 0 }}
Write-Host 'BLOCKED'; exit 1
""")
        r = _ps(contender)
        assert "BLOCKED" in r.stdout, (
            f"a second loop took a lock another process holds: {r.stdout} {r.stderr}"
        )
        assert "data loop" in r.stdout, "the wait message should name the holder"
    finally:
        a.kill()
        a.wait(timeout=30)


@needs_powershell
def test_lock_is_available_again_once_released(tmp_path: Path):
    """Exclusion that never releases would jam the loop instead of racing it."""
    first = _write(tmp_path / "first.ps1", f"""
$p = Enter-RepoLock -LogDir '{tmp_path.as_posix()}' -Holder 'data loop' -PollSeconds 1
Exit-RepoLock $p
if (Test-Path '{tmp_path.as_posix()}/.repo.lock') {{ Write-Host 'STILL-LOCKED'; exit 1 }}
Write-Host 'RELEASED'
""")
    assert "RELEASED" in _ps(first).stdout

    second = _write(tmp_path / "second.ps1", f"""
$p = Enter-RepoLock -LogDir '{tmp_path.as_posix()}' -Holder 'code loop' -MaxWaitMinutes 0 -PollSeconds 1
if ($p) {{ Write-Host 'GOT-IT' }} else {{ Write-Host 'BLOCKED' }}
""")
    assert "GOT-IT" in _ps(second).stdout


@needs_powershell
def test_lock_left_by_a_dead_process_is_reclaimed(tmp_path: Path):
    """A crashed run must not jam every run after it.

    PID 999999 is above Windows' range and cannot be live.
    """
    (tmp_path / ".repo.lock").write_text("999999\ndata loop\n2026-08-29\n", encoding="ascii")
    script = _write(tmp_path / "reclaim.ps1", f"""
$p = Enter-RepoLock -LogDir '{tmp_path.as_posix()}' -Holder 'code loop' -MaxWaitMinutes 0 -PollSeconds 1
if ($p) {{ Write-Host 'GOT-IT' }} else {{ Write-Host 'BLOCKED' }}
""")
    r = _ps(script)
    assert "GOT-IT" in r.stdout, f"a stale lock jammed the loop: {r.stdout} {r.stderr}"
    assert "Reclaiming" in r.stdout


@needs_powershell
def test_release_does_not_drop_someone_elses_lock(tmp_path: Path):
    """After a reclaim the lock belongs to another process.

    Deleting it on the way out would let a third process in alongside them.
    """
    (tmp_path / ".repo.lock").write_text("999999\nsomeone else\n2026-08-29\n", encoding="ascii")
    script = _write(tmp_path / "release.ps1", f"""
Exit-RepoLock '{tmp_path.as_posix()}/.repo.lock'
if (Test-Path '{tmp_path.as_posix()}/.repo.lock') {{ Write-Host 'KEPT' }} else {{ Write-Host 'DELETED' }}
""")
    assert "KEPT" in _ps(script).stdout


@needs_powershell
def test_exactly_one_of_several_simultaneous_starts_wins(tmp_path: Path):
    """Test-Path-then-write would let two racers both pass; CreateNew cannot.

    This is the shape of the real failure: processes starting in the same
    second, not neatly one after another.

    The winner holds for longer than the others take to start, so a loser sees
    a *live* holder. Without the hold every racer would find a lock owned by an
    exited process and correctly reclaim it - which is the stale-lock path
    covered above, not the race this test is about.
    """
    racers = []
    for i in range(6):
        s = _write(tmp_path / f"race{i}.ps1", f"""
$p = Enter-RepoLock -LogDir '{tmp_path.as_posix()}' -Holder 'racer {i}' -MaxWaitMinutes 0 -PollSeconds 1
if ($p) {{
    Add-Content -Path '{tmp_path.as_posix()}/winners' -Value $PID
    Start-Sleep -Seconds 15
    Exit-RepoLock $p
}}
""")
        racers.append(subprocess.Popen(
            [POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(s)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ))
    for p in racers:
        p.wait(timeout=120)

    winners = (tmp_path / "winners").read_text().split()
    assert len(winners) == 1, f"{len(winners)} processes held the repo at once: {winners}"
