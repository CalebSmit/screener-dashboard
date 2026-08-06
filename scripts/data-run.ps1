<#
.SYNOPSIS
    Data loop: run the screener live, refresh the dashboard, feed the
    improvement engine. Mon/Wed/Fri, 2:00 AM.

.DESCRIPTION
    This is the loop that makes the methodology able to learn. Each run records
    an improvement-engine snapshot; as forward returns mature, those snapshots
    become live information coefficients, which is the evidence
    improvement_engine.py needs before it may adjust any factor weight.

    Without this running, the screener's self-improvement is inert - which is
    exactly the state it was in from 2026-02 to 2026-08 (3 IC observations,
    8 required).

    It runs at 2:00 AM so the 6:00 AM code loop starts from a clean tree with
    fresh evidence already committed.

    NOTE: keep this file ASCII-only and saved as UTF-8 with BOM.

.PARAMETER SkipPush
    Run and commit locally, but do not push to main.

.PARAMETER Tickers
    Comma-separated ticker subset, for testing (e.g. AAPL,MSFT,GOOGL).
#>
[CmdletBinding()]
param(
    [switch]$SkipPush,
    [string]$Tickers
)

$ErrorActionPreference = 'Continue'

$RepoPath = Split-Path -Parent $PSScriptRoot
$LogDir   = Join-Path $RepoPath 'logs'
$LockFile = Join-Path $LogDir '.datarun.lock'
$Date     = Get-Date -Format 'yyyy-MM-dd'
$Stamp    = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$LogFile  = Join-Path $LogDir "datarun-$Stamp.log"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -ErrorAction Stop | Out-Null }

function Write-Log {
    param([string]$Message, [string]$Level = 'INFO')
    $line = "[{0}] [{1}] {2}" -f (Get-Date -Format 'HH:mm:ss'), $Level, $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding utf8
}

function Invoke-Native {
    param([Parameter(Mandatory)][string]$Exe, [string[]]$Arguments = @())
    $out = & $Exe @Arguments 2>&1 | ForEach-Object { [string]$_ }
    return [pscustomobject]@{
        Output   = @($out)
        ExitCode = $LASTEXITCODE
        Text     = (@($out) -join "`n")
    }
}

function Stop-Run {
    param([string]$Message, [int]$Code = 1)
    Write-Log $Message 'ERROR'
    if (Test-Path $LockFile) { Remove-Item $LockFile -Force -ErrorAction SilentlyContinue }
    exit $Code
}

$env:Path = "$([Environment]::GetEnvironmentVariable('Path','Machine'));$([Environment]::GetEnvironmentVariable('Path','User'))"

if (Test-Path $LockFile) {
    $age = (Get-Date) - (Get-Item $LockFile).LastWriteTime
    if ($age.TotalHours -lt 6) { Write-Log "Data run already in progress ($([int]$age.TotalMinutes)m). Exiting." 'WARN'; exit 0 }
    Write-Log "Stale lock. Reclaiming." 'WARN'; Remove-Item $LockFile -Force
}
Set-Content -Path $LockFile -Value $PID -Encoding utf8

try {
    Set-Location $RepoPath
    Write-Log "=== Data loop $Date ==="

    foreach ($tool in @('git', 'python')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { Stop-Run "'$tool' not on PATH." }
    }

    $co = Invoke-Native 'git' @('checkout', 'main')
    if ($co.ExitCode -ne 0) { Stop-Run "Could not check out main." }
    Invoke-Native 'git' @('pull', '--ff-only') | Out-Null

    # --- Run the screener ---------------------------------------------------
    $args = @('run_screener.py')
    if ($Tickers) { $args += @('--tickers', $Tickers) }
    Write-Log "Running: python $($args -join ' ')  (expect this to take a while)"

    $run = Invoke-Native 'python' $args
    foreach ($line in ($run.Output | Select-Object -Last 25)) {
        if ($line -and $line.Trim()) { Write-Log "    $($line.Trim())" }
    }

    if ($run.ExitCode -ne 0) {
        Write-Log "Screener exited $($run.ExitCode). Discarding partial output." 'ERROR'
        Invoke-Native 'git' @('checkout', '--', '.') | Out-Null
        Stop-Run "Data run failed - the code loop will see this in logs/ and should treat it as top priority." 2
    }

    # --- Regenerate the dashboard -------------------------------------------
    Write-Log "Regenerating dashboard..."
    $gd = Invoke-Native 'python' @('generate_dashboard.py')
    if ($gd.ExitCode -ne 0) {
        Write-Log "generate_dashboard.py failed - not publishing." 'ERROR'
        Invoke-Native 'git' @('checkout', '--', '.') | Out-Null
        Stop-Run "Dashboard generation failed." 2
    }

    # index.html is what GitHub Pages serves; keep it in step with dashboard.html
    $dash = Join-Path $RepoPath 'dashboard.html'
    $idx  = Join-Path $RepoPath 'index.html'
    if (Test-Path $dash) { Copy-Item $dash $idx -Force; Write-Log "Copied dashboard.html -> index.html" }

    # --- Sanity-check before publishing --------------------------------------
    foreach ($pair in @(@($idx, 50000), @((Join-Path $RepoPath 'dashboard_data.js'), 100000))) {
        $p = $pair[0]; $min = $pair[1]
        if (-not (Test-Path $p) -or (Get-Item $p).Length -lt $min) {
            Write-Log "Output $p looks wrong (missing or under $min bytes). Not publishing." 'ERROR'
            Invoke-Native 'git' @('checkout', '--', '.') | Out-Null
            Stop-Run "Refusing to publish a broken dashboard." 2
        }
    }

    # --- Commit --------------------------------------------------------------
    $changed = Invoke-Native 'git' @('status', '--porcelain')
    if (-not $changed.Text.Trim()) {
        Write-Log "Run produced no changes. Nothing to publish."
        Write-Log "=== Data loop complete ==="
        exit 0
    }

    Invoke-Native 'git' @('add', '-A') | Out-Null
    $commit = Invoke-Native 'git' @('commit', '-m', "data: screener run $Date")
    if ($commit.ExitCode -ne 0) { Stop-Run "Commit failed." 2 }
    Write-Log "Committed run output."

    if ($SkipPush) { Write-Log "-SkipPush set; stopping before push."; exit 0 }

    $push = Invoke-Native 'git' @('push', 'origin', 'main')
    if ($push.ExitCode -ne 0) { Stop-Run "Push to main failed. Local main is ahead." 2 }
    Write-Log "Published to main - live dashboard refreshed."

    # --- Report evidence accumulation ---------------------------------------
    $icPath = Join-Path $RepoPath 'improvement\live_ic_history.csv'
    if (Test-Path $icPath) {
        $obs = (Get-Content $icPath | Measure-Object -Line).Lines - 1
        Write-Log "Improvement engine now has $obs live IC observation(s); 8 are needed before it may propose a weight change."
    }

    Write-Log "=== Data loop complete ==="
    exit 0
}
catch {
    Write-Log "Unhandled error: $($_.Exception.Message)" 'ERROR'
    Write-Log "$($_.ScriptStackTrace)" 'ERROR'
    exit 1
}
finally {
    if (Test-Path $LockFile) { Remove-Item $LockFile -Force -ErrorAction SilentlyContinue }
}
