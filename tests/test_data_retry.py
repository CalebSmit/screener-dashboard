"""The same-day retry of the data loop, and the code loop updating its own CLI (2026-11-10).

WHY THIS EXISTS. On 2026-11-06, 11-09 and 11-10 the 02:00 data run was refused by its own publish
gate, each time for a fault in a test's premise rather than in the page; each time the 06:00 session
fixed it within hours; and each time the live site still sat on old data until the next 02:00 run -
by 11-10 it was three trading days stale. Separately, the 11-10 13:00 code session died with "Claude
Code 2.1.229 does not support this model; version 2.1.290 or newer is required", and the 06:00
session the next morning would have died the same way.

What is pinned here:

* **"Screener Data Retry", 18:00 Mon-Fri, runs `data-run.ps1 -Retry`.** The success marker the
  data loop already writes makes it a no-op on any day that published - so it costs nothing on a
  good day and cannot publish twice.
* **It never touches a folder someone is working in.** An owner session stands it down before
  anything else; a dirty tree or a branch off `main` stands it down under the repo lock, **before
  the stash and the checkout** the 02:00 run uses as its rescue - and without writing the morning
  brief into the tree. The 02:00 run's own behaviour is unchanged.
* **No logon trigger and no late catch-up**, like the extra code slots: firing at an arbitrary
  hour is how it would land on the owner working.
* **The code runner updates the CLI once and retries the session once** when the API says the CLI
  is too old for the model - and only then.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "scripts" / "register-tasks.ps1"
DATA_RUN = ROOT / "scripts" / "data-run.ps1"
NIGHTLY = ROOT / "scripts" / "nightly-screener.ps1"


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def _spec(name: str) -> str:
    src = _src(REGISTER)
    m = re.search(r"Name\s*=\s*'" + re.escape(name) + r"'", src)
    assert m, f"{name} has no entry in $Specs"
    block = src[m.start():]
    return block[:block.index("Description")]


def test_the_retry_is_registered_from_version_control_after_the_close():
    block = _spec("Screener Data Retry")
    assert "Script      = 'data-run.ps1'" in block
    assert "Arguments   = '-Retry'" in block
    assert "At          = '6:00PM'" in block
    assert "Days" not in block                       # the default: Monday to Friday, as the 02:00 run


def test_the_retry_neither_catches_up_late_nor_joins_the_logon_pair():
    block = _spec("Screener Data Retry")
    assert re.search(r"NoLogon\s*=\s*\$true", block) and re.search(r"NoCatchUp\s*=\s*\$true", block)
    assert "LogonDelay" not in block


def test_the_02_00_run_is_unchanged():
    block = _spec("Screener Data Run")
    assert "At          = '2:00AM'" in block and "Arguments" not in block and "AlsoAt" not in block


def test_a_day_that_published_makes_the_retry_a_no_op():
    """The success marker is checked before the retry's own logic, and written only after the push."""
    src = _src(DATA_RUN)
    marker_check = src.index("if ((Test-Path $SuccessMarker) -and -not $Force)")
    retry = src.index("if ($Retry -and -not $Force) {")
    assert marker_check < retry
    assert src.index("[System.IO.File]::WriteAllText($SuccessMarker, $Date)") > src.index("'push'")


def test_an_owner_session_stands_the_retry_down_before_anything_runs():
    src = _src(DATA_RUN)
    first = src.index("if ($Retry -and -not $Force) {")
    block = src[first:src.index("\n}\n", first)]
    assert "'.owner-session'" in block and "TotalHours -lt 8" in block and "exit 0" in block
    assert "Invoke-Native 'git'" not in block                    # no git before the repo lock
    assert first < src.index("Enter-RepoLock -LogDir")
    assert first < src.index("$args = @('run_screener.py'")     # nor the screener


def test_work_in_progress_stands_the_retry_down_under_the_lock_and_before_the_rescue():
    src = _src(DATA_RUN)
    lock = src.index("Enter-RepoLock -LogDir")
    second = src.index("if ($Retry -and -not $Force) {", lock)
    assert lock < second
    assert second < src.index("'stash', 'push'")                 # never reaches the stash
    assert second < src.index("@('checkout', 'main')")           # nor the branch switch
    block = src[second:src.index("'stash', 'push'")]
    assert "$here -ne 'main' -or $wip" in block and "exit 0" in block
    assert "$SkipBrief = $true" in block                         # no brief written into the tree


def test_the_brief_is_skipped_only_when_asked():
    src = _src(DATA_RUN)
    assert "$SkipBrief = $false" in src
    assert 'if (-not $SkipBrief) { Publish-Brief "data run $Date" }' in src
    assert src.count("$SkipBrief = $true") == 1                  # only the retry's stand-down sets it


def test_the_runner_updates_the_cli_once_when_the_api_says_it_is_too_old():
    src = _src(NIGHTLY)
    i = src.index("if (-not $session.Ok -and $session.Reason -match 'does not support this model|or newer is required')")
    block = src[i:src.index("$SessionFailed = -not $session.Ok", i)]
    assert "@('install', '-g', '@anthropic-ai/claude-code@latest')" in block
    assert block.count("claude --print") == 1                    # one retry, never a loop
    assert "if ($upd.ExitCode -eq 0)" in block                   # no retry on a failed update
    assert "Get-SessionOutcome" in block                         # the retry's outcome is what counts
    assert i > src.index("$session = Get-SessionOutcome")         # after the first attempt
    assert i < src.index("Write-Log \"SESSION DID NOT RUN")       # before it is reported failed


def test_the_ascii_rule_still_holds():
    """Windows PowerShell 5.1 reads these as cp1252 without a BOM; non-ASCII breaks the parser."""
    for p in (DATA_RUN, REGISTER, NIGHTLY):
        raw = p.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf"), p.name
        assert all(b < 128 for b in raw[3:]), p.name
