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
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
SCRIPTS = sorted(SCRIPTS_DIR.glob("*.ps1"))

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


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_no_undefined_functions(path: Path):
    """Every Verb-Noun call resolves to a definition or a known cmdlet."""
    src = _source(path)
    defined = set(re.findall(r"(?im)^\s*function\s+([A-Za-z]+(?:-[A-Za-z]+)?)", src))
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
