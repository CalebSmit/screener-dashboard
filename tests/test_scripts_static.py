"""Static checks on the PowerShell runner scripts.

These exist because on 2026-08-11 the 2 AM data run crashed with
"The term 'Write-NativeOutput' is not recognized" - a helper that was defined in
nightly-screener.ps1 but called in data-run.ps1. The bug shipped because the
only pre-flight check available at the time counted braces and parentheses,
which cannot see an undefined function. The data loop did not publish that day.

The scripts are unattended infrastructure: a typo in them means a silently
skipped run, so they get the same regression coverage as the Python.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPTS = sorted(SCRIPTS_DIR.glob("*.ps1"))
POWERSHELL = shutil.which("powershell") or shutil.which("pwsh")

# Cmdlets and keywords the scripts legitimately rely on. Anything Verb-Noun
# shaped that is neither here nor defined in the file is treated as a typo.
KNOWN = {
    # flow / language
    "if", "else", "elseif", "foreach", "for", "while", "do", "switch",
    "try", "catch", "finally", "function", "param", "return", "break",
    "continue", "exit", "throw",
    # cmdlets actually used
    "Write-Host", "Write-Output", "Write-Warning", "Write-Error",
    "Get-Date", "Get-Content", "Set-Content", "Add-Content", "Get-Item",
    "Get-ChildItem", "Test-Path", "New-Item", "Remove-Item", "Copy-Item",
    "Join-Path", "Split-Path", "Out-Null", "Out-File", "Out-String",
    "Select-Object", "Where-Object", "ForEach-Object", "Measure-Object",
    "Sort-Object", "Select-String", "Format-Table", "Import-Csv",
    "ConvertFrom-Json", "ConvertTo-Json", "Start-Sleep", "Read-Host",
    "Invoke-WebRequest", "Get-Command", "Add-Member", "Set-Location",
    "Get-Process", "New-Object",
    # scheduled task cmdlets
    "Get-ScheduledTask", "Set-ScheduledTask", "Register-ScheduledTask",
    "Unregister-ScheduledTask", "Enable-ScheduledTask", "Disable-ScheduledTask",
    "Get-ScheduledTaskInfo", "New-ScheduledTaskAction", "New-ScheduledTaskTrigger",
    "New-ScheduledTaskPrincipal", "New-ScheduledTaskSettingsSet", "New-TimeSpan",
}


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig")


def _strip(src: str) -> str:
    """Remove block comments, line comments and string literals."""
    src = re.sub(r"<#.*?#>", "", src, flags=re.S)
    src = re.sub(r"@'.*?'@", "", src, flags=re.S)
    src = re.sub(r'@".*?"@', "", src, flags=re.S)
    src = re.sub(r"'[^'\n]*'", "", src)
    src = re.sub(r'"[^"\n]*"', "", src)
    return re.sub(r"(?m)#.*$", "", src)


def _defined_in(path: Path) -> set[str]:
    return set(
        re.findall(r"(?im)^\s*function\s+([A-Za-z]+(?:-[A-Za-z]+)?)", _source(path))
    )


def _dot_sourced(path: Path) -> list[Path]:
    """Scripts this one dot-sources, resolved relative to scripts/.

    Only literal `. (Join-Path $PSScriptRoot 'x.ps1')` is recognised. That is
    deliberately narrow: the original bug this module exists for was a helper
    used across files that were never sourced into one another, so a call is
    only excused by a dot-source the reader can see.
    """
    pattern = r"^\s*\.\s*\(\s*Join-Path\s+\$PSScriptRoot\s+'([^']+\.ps1)'\s*\)"
    out = []
    for name in re.findall(pattern, _source(path), flags=re.M):
        target = SCRIPTS_DIR / name
        assert target.exists(), f"{path.name} dot-sources missing {name}"
        out.append(target)
    return out


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_undefined_functions(path: Path):
    """Every Verb-Noun call resolves to a definition or a known cmdlet.

    A definition counts if it is in this file or in a script this file
    dot-sources - not merely because it exists somewhere in scripts/.
    """
    src = _source(path)
    defined = _defined_in(path)
    for sourced in _dot_sourced(path):
        defined |= _defined_in(sourced)
    body = _strip(src)
    called = set(re.findall(r"(?<![-\w$])([A-Z][a-z]+-[A-Za-z]+)", body))
    unknown = sorted(c for c in called if c not in defined and c not in KNOWN)
    assert not unknown, (
        f"{path.name} calls undefined function(s): {unknown}. "
        f"Either define them in this file or add genuine cmdlets to KNOWN."
    )


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_balanced_blocks(path: Path):
    body = _strip(_source(path))
    assert body.count("{") == body.count("}"), f"{path.name}: unbalanced braces"
    assert body.count("(") == body.count(")"), f"{path.name}: unbalanced parens"


@pytest.mark.skipif(POWERSHELL is None, reason="no PowerShell on this platform")
@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_parses_as_powershell(path: Path):
    """Hand the file to PowerShell's own parser.

    Counting braces above is a proxy; this is the real thing, and it costs a
    subprocess. A script that does not parse is a silently skipped run, which
    is how 2026-08-11 was lost.
    """
    probe = (
        "$e = $null; "
        "$null = [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{path.as_posix()}', [ref]$null, [ref]$e); "
        "if ($e -and $e.Count) { "
        "$e | ForEach-Object { Write-Host \"line $($_.Extent.StartLineNumber): $($_.Message)\" }; "
        "exit 1 }"
    )
    r = subprocess.run(
        [POWERSHELL, "-NoProfile", "-NonInteractive", "-Command", probe],
        capture_output=True, text=True, timeout=120,
    )
    assert r.returncode == 0, f"{path.name} does not parse:\n{r.stdout}{r.stderr}"


def test_no_script_hardcodes_a_shared_logon_delay():
    """Both tasks sharing one delay is what made them start together.

    add-catchup-trigger.ps1 set 'PT3M' on both and would have quietly restored
    the collision if it were ever re-run, so it now delegates to
    register-tasks.ps1 instead of writing triggers of its own.
    """
    offenders = []
    for path in SCRIPTS:
        src = _strip(_source(path))
        if re.search(r"\$logon\.Delay\s*=\s*'PT\d+M'", src):
            offenders.append(path.name)
    assert not offenders, (
        f"{offenders} assign a literal logon delay; it must come from the "
        f"per-task spec so the two loops can be staggered"
    )


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_utf8_bom_and_ascii(path: Path):
    """Windows PowerShell 5.1 reads a BOM-less file as cp1252.

    A UTF-8 em dash then decodes to bytes the parser mistakes for a quote,
    which broke nightly-screener.ps1 during setup.
    """
    raw = path.read_bytes()
    assert raw[:3] == b"\xef\xbb\xbf", f"{path.name}: missing UTF-8 BOM"
    text = raw[3:].decode("utf-8")
    non_ascii = sorted({c for c in text if ord(c) > 127})
    assert not non_ascii, f"{path.name}: non-ASCII characters {non_ascii}"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_helpers_used_are_defined_before_finally(path: Path):
    """Publish-Brief runs from `finally`, so it must exist in that file."""
    src = _source(path)
    if "Publish-Brief" in _strip(src):
        assert re.search(r"(?im)^\s*function\s+Publish-Brief", src), (
            f"{path.name} calls Publish-Brief but does not define it"
        )


# --------------------------------------------------------------------------
# A session that never ran must not report as a session that found nothing.
#
# 2026-08-14: the CLI exited 1 after one second on an API weekly-limit 429.
# nightly-screener.ps1 captured $claudeExit, logged it at INFO, and never
# branched on it. With no commits, all four gates passed trivially, the runner
# logged "Run complete (no changes)", wrote the once-per-day success marker and
# published a normal morning brief. The owner's only status channel reported a
# healthy morning; in fact the day was lost. Found by the 2026-08-21
# retrospective, which also found that 7 of 11 scheduled code sessions to date
# never started - making "did it run?" the single most load-bearing question
# this runner asks.
# --------------------------------------------------------------------------

NIGHTLY = SCRIPTS_DIR / "nightly-screener.ps1"


def _block(src: str, opener: str) -> str:
    """Source text of the brace-block introduced by `opener`."""
    start = src.index(opener)
    depth, i = 0, src.index("{", start)
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i : j + 1]
    raise AssertionError(f"unterminated block after {opener!r}")


def test_session_outcome_is_judged_from_the_transcript():
    """The runner must decide whether the session actually ran."""
    src = _source(NIGHTLY)
    assert re.search(r"(?im)^\s*function\s+Get-SessionOutcome", src), (
        "nightly-screener.ps1 must define Get-SessionOutcome"
    )
    assert "Get-SessionOutcome -TranscriptPath" in src, (
        "Get-SessionOutcome is defined but never called on the transcript"
    )
    for token in ("is_error", "api_error_status", "num_turns"):
        assert token in src, (
            f"the transcript check ignores '{token}', so an API-limit failure "
            f"would still look like a healthy session"
        )


def test_no_success_marker_when_the_session_did_not_run():
    """The once-per-day marker must not be written for a lost day.

    Writing it means the at-logon catch-up trigger skips the retry, so one
    API-limit failure silently costs the whole day.
    """
    src = _source(NIGHTLY)
    block = _block(src, "if ($commitCount -eq 0)")
    assert "$SessionFailed" in block, (
        "the 'no commits' path does not check $SessionFailed - it cannot tell "
        "'nothing to do' from 'never started'"
    )
    assert block.index("$SessionFailed") < block.index("$SuccessMarker"), (
        "$SuccessMarker is written before $SessionFailed is checked"
    )


def test_failed_session_is_logged_at_error_and_exits_nonzero():
    src = _source(NIGHTLY)
    assert "SESSION DID NOT RUN" in src, (
        "a failed session must be logged distinctly; write_brief.py classifies "
        "run logs by their text, and 'Run complete (no changes)' reads as success"
    )
    block = _block(src, "if ($commitCount -eq 0)")
    assert "Stop-Run" in block, "a lost day must exit non-zero, not 0"


def test_success_marker_is_guarded_on_every_path():
    """No path may stamp the day as done when the session did not run."""
    src = _source(NIGHTLY)
    no_commit = _block(src, "if ($commitCount -eq 0)")
    writes = []
    for m in re.finditer(r"WriteAllText\(\$SuccessMarker", src):
        start = src.rfind("\n", 0, m.start()) + 1
        writes.append(src[start : src.index("\n", m.start())])
    assert writes, "the runner no longer writes a success marker at all"

    for line in writes:
        if line in no_commit:
            continue  # covered by test_no_success_marker_when_the_session_did_not_run
        assert "$SessionFailed" in line, (
            f"success-marker write is not guarded by $SessionFailed: {line.strip()}"
        )
