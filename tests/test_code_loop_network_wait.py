"""The code loop waits for the network before giving up on the day, as the data loop does.

WHY THIS EXISTS - 2026-11-03. The 06:00 session logged
``fatal: unable to access 'https://github.com/...': Could not resolve host: github.com`` at
06:00:01, one second after the task woke the machine, and skipped the day. ``data-run.ps1`` has
always waited up to five minutes for the network for exactly this reason ("the machine may have
just woken from sleep for this task"); ``nightly-screener.ps1`` tried once. The skip also left an
unpublished brief in the tree, which stands the 13:00 slot down - see
``tests/test_brief_publish_safety.py::test_an_unpublished_brief_is_set_aside_and_the_tree_left_clean``.

These run the runner's **own** block in PowerShell - cut out of the script, not retyped - with
``git`` and the sleep stubbed, so they check what the code does rather than how it reads.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts" / "nightly-screener.ps1"
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")


def _block() -> str:
    src = RUNNER.read_text(encoding="utf-8-sig")
    start = src.index("    $RemoteTries = 10")
    end = src.index('Write-Log "Remote reachable', start)
    end = src.index("\n", end) + 1
    return src[start:end]


def _run(codes: list[int]) -> subprocess.CompletedProcess:
    seq = ", ".join(str(c) for c in codes)
    script = (
        "$codes = @(" + seq + "); $script:i = 0; $script:sleeps = 0\n"
        "function Invoke-Native { param($exe, $argv) "
        "$c = $codes[[Math]::Min($script:i, $codes.Count - 1)]; $script:i++; "
        "[pscustomobject]@{ ExitCode = $c; Text = 'Could not resolve host: github.com' } }\n"
        "function Start-Sleep { param([int]$Seconds) $script:sleeps++ }\n"
        "function Write-Log { param($m, $l = 'INFO') Write-Output \"LOG[$l] $m\" }\n"
        "function Write-NativeOutput { param($r, $l) }\n"
        "function Stop-Run { param($m) Write-Output \"STOP $m\"; Write-Output \"CALLS $script:i SLEEPS $script:sleeps\"; exit 2 }\n"
        + _block()
        + "Write-Output \"CALLS $script:i SLEEPS $script:sleeps\"\n"
    )
    return subprocess.run([POWERSHELL, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-Command", script], capture_output=True, text=True, timeout=120)


requires_shell = pytest.mark.skipif(POWERSHELL is None, reason="PowerShell not available")


@requires_shell
def test_a_reachable_remote_costs_one_call_and_no_wait():
    r = _run([0])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "LOG[INFO] Remote reachable." in r.stdout
    assert "CALLS 1 SLEEPS 0" in r.stdout


@requires_shell
def test_a_network_that_comes_up_late_is_waited_for():
    """The 2026-11-03 case: DNS fails on waking, then works."""
    r = _run([128, 128, 0])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "Remote reachable on attempt 3." in r.stdout
    assert "CALLS 3 SLEEPS 2" in r.stdout
    assert r.stdout.count("Remote not reachable yet") == 2


@requires_shell
def test_a_remote_that_never_answers_still_stops_the_run():
    r = _run([128] * 10)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "STOP Cannot reach the git remote after 10 attempts" in r.stdout
    assert "CALLS 10 SLEEPS 9" in r.stdout, "ten tries, no sleep after the last"


def test_the_wait_matches_the_data_loops_budget():
    """Same budget as data-run.ps1's network wait: 10 tries, 30 s apart."""
    data = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8-sig")
    assert re.search(r"\$attempt -le 10;", data) and "Start-Sleep -Seconds 30" in data
    block = _block()
    assert "$RemoteTries = 10" in block and "Start-Sleep -Seconds 30" in block
