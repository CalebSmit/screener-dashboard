<#
.SYNOPSIS
    Collect option quotes after the close, so the 02:00 run has some to read.

.DESCRIPTION
    Measured 2026-10-09 (research/measurements/2026-10-09-option-quote-availability.py):
    Yahoo serves the option chain overnight with every bid, ask and implied volatility at
    zero. The 02:00 data loop therefore spent ~1,000 option requests and produced a usable
    reading for 0 of 503 stocks, against 394 of 503 at 21:27 ET the evening before. The
    06:00 code loop is in the same dead window.

    This task fills data/options/quotes.json at an hour when quotes exist; context_fetch
    probes at 02:00, finds none being served, and reads that file instead.

    Deliberately NOT like the other two loops:

      * It runs no git command and writes only under data/options/, which is gitignored.
        So it needs no repo lock and cannot collide with either loop or with an owner
        session editing the tree. tests/test_option_quote_refresh_task.py pins that.
      * A run at an hour with no quotes is a correct no-op, not a failure: options_cache
        probes first and leaves the existing cache alone. Exit code stays 0 so that a real
        fault is not lost among expected red runs.
      * Nothing it collects can reach a score (CLAUDE.md settled row "ctx").

.PARAMETER Budget
    Seconds to spend before keeping what it has. Default 900, matching the context pass.

.PARAMETER Limit
    Only the first N stocks - for a smoke test.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\refresh-option-quotes.ps1 -Limit 5

.NOTES
    Registered by scripts/register-tasks.ps1 as "Screener Option Quotes", weekdays 8:00 PM
    local. Keep this file ASCII-only and saved as UTF-8 with BOM.
#>
[CmdletBinding()]
param(
    [double]$Budget = 900,
    [int]$Limit = 0
)

$ErrorActionPreference = 'Continue'

$RepoPath = Split-Path -Parent $PSScriptRoot
$LogDir   = Join-Path $RepoPath 'logs'
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$LogFile  = Join-Path $LogDir ("options-" + (Get-Date -Format 'yyyy-MM-dd_HHmmss') + ".log")

function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $line = "[{0}] [{1}] {2}" -f (Get-Date -Format 'HH:mm:ss'), $Level, $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Write-Log "=== Option quote refresh ==="
Write-Log "repo: $RepoPath"

Set-Location $RepoPath

$pyArgs = @('options_cache.py', '--budget', $Budget)
if ($Limit -gt 0) { $pyArgs += @('--limit', $Limit) }

try {
    $output = & python @pyArgs 2>&1
    $code = $LASTEXITCODE
} catch {
    Write-Log "python could not be run: $($_.Exception.Message)" 'ERROR'
    Write-Log "=== Refresh ended without fetching ==="
    exit 0
}

foreach ($line in $output) {
    if ($line -and "$line".Trim()) { Write-Log ("    " + "$line".Trim()) }
}

if ($code -ne 0) {
    Write-Log "options_cache.py exited $code. The cache is left as it was." 'WARN'
} else {
    Write-Log "Refresh complete."
}

# The repo must be exactly as it was found: this task publishes nothing.
$dirty = & git status --porcelain 2>$null
if ($dirty) {
    Write-Log "Working tree is dirty after the refresh - it should not be. Left untouched:" 'WARN'
    foreach ($d in $dirty) { Write-Log "    $d" 'WARN' }
}

Write-Log "=== Done ==="
exit 0
