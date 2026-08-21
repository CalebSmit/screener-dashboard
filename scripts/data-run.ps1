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
    [string]$Tickers,
    [switch]$Force
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

function Write-NativeOutput {
    param([object]$Result, [string]$Level = 'INFO')
    foreach ($line in $Result.Output) {
        if ($line -and $line.Trim()) { Write-Log "    $($line.Trim())" $Level }
    }
}

function Stop-Run {
    param([string]$Message, [int]$Code = 1)
    Write-Log $Message 'ERROR'
    if (Test-Path $LockFile) { Remove-Item $LockFile -Force -ErrorAction SilentlyContinue }
    exit $Code
}

# Writes MORNING_BRIEF.md and publishes it. Called from `finally`, so a summary
# appears whether the run published, was discarded by the health check, or
# crashed - a discarded run is the one you most want to be told about.
function Publish-Brief {
    param([string]$Label)
    try {
        Write-Log "Writing morning brief..."
        $b = Invoke-Native 'python' @('scripts/write_brief.py')
        Write-NativeOutput $b
        if ($b.ExitCode -ne 0) { return }

        Invoke-Native 'git' @('add', '--', 'MORNING_BRIEF.md') | Out-Null
        $staged = Invoke-Native 'git' @('diff', '--cached', '--name-only', '--', 'MORNING_BRIEF.md')
        if (-not $staged.Text.Trim()) { Write-Log "Brief unchanged; nothing to publish."; return }

        $c = Invoke-Native 'git' @('commit', '-m', "brief: $Label")
        if ($c.ExitCode -ne 0) { Write-Log "Could not commit brief." 'WARN'; return }

        $p = Invoke-Native 'git' @('push', 'origin', 'HEAD:main')
        if ($p.ExitCode -eq 0) { Write-Log "Morning brief published." }
        else { Write-Log "Brief committed locally; push failed." 'WARN' }
    } catch {
        Write-Log "Brief step failed (non-fatal): $($_.Exception.Message)" 'WARN'
    }
}

$env:Path = "$([Environment]::GetEnvironmentVariable('Path','Machine'));$([Environment]::GetEnvironmentVariable('Path','User'))"

if (Test-Path $LockFile) {
    $age = (Get-Date) - (Get-Item $LockFile).LastWriteTime
    if ($age.TotalHours -lt 6) { Write-Log "Data run already in progress ($([int]$age.TotalMinutes)m). Exiting." 'WARN'; exit 0 }
    Write-Log "Stale lock. Reclaiming." 'WARN'; Remove-Item $LockFile -Force
}
Set-Content -Path $LockFile -Value $PID -Encoding utf8

# --- Run-once-per-day guard --------------------------------------------------
# These tasks use InteractiveToken, so they only fire while a user is logged on.
# On 2026-08-12 the machine rebooted at 00:47 (Windows Update) and sat at the
# login screen; both the 02:00 and 06:00 runs were simply lost. A catch-up
# logon trigger fixes that, but then needs this guard so logging in three times
# does not run the loop three times.
$SuccessMarker = Join-Path $LogDir '.datarun-last-success'
if ((Test-Path $SuccessMarker) -and -not $Force) {
    # Strip a UTF-8 BOM: Set-Content -Encoding utf8 writes one in PS 5.1 and
    # .Trim() does not remove it, so this comparison silently never matched.
    $last = (Get-Content $SuccessMarker -TotalCount 1).TrimStart([char]0xFEFF).Trim()
    if ($last -eq $Date) {
        Write-Log "Data loop already completed successfully today ($Date). Nothing to do."
        if (Test-Path $LockFile) { Remove-Item $LockFile -Force -ErrorAction SilentlyContinue }
        exit 0
    }
}

try {
    Set-Location $RepoPath
    Write-Log "=== Data loop $Date ==="

    foreach ($tool in @('git', 'python')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { Stop-Run "'$tool' not on PATH." }
    }

    # --- Wait for the network -----------------------------------------------
    # The machine may have just woken from sleep for this task, in which case
    # the network stack often is not up yet. Without this, the screener runs
    # with no connectivity and silently substitutes synthetic data.
    $netOk = $false
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        try {
            $r = Invoke-WebRequest -Uri 'https://query1.finance.yahoo.com' -UseBasicParsing `
                    -TimeoutSec 15 -ErrorAction Stop
            if ($r.StatusCode -ge 200) { $netOk = $true; break }
        } catch {
            # Any HTTP response at all means DNS and routing work.
            if ($_.Exception.Response) { $netOk = $true; break }
        }
        Write-Log "Network not ready (attempt $attempt/10). Waiting 30s..." 'WARN'
        Start-Sleep -Seconds 30
    }
    if (-not $netOk) {
        Stop-Run "No network connectivity after 5 minutes. Skipping today's run rather than publishing synthetic data." 2
    }
    Write-Log "Network is up."

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

    # --- Data quality gate ---------------------------------------------------
    # The screener falls back to "sector-realistic sample values" when a fetch
    # fails. That is fine for one or two tickers; it is fabricated data if it
    # happens to most of them. Publishing that to a public dashboard would be
    # the single worst thing this system could do, so check before believing
    # the output. File size is NOT a sufficient check - a fully synthetic run
    # produces a perfectly normal-looking 2.6 MB payload.
    $dqLog = Join-Path $RepoPath 'validation\data_quality_log.csv'
    if (Test-Path $dqLog) {
        $rows = @(Import-Csv $dqLog)
        $todayRows = @($rows | Where-Object { $_.Timestamp -like "$Date*" })
        $synthetic = @($todayRows | Where-Object {
            $_.Description -match 'synthetic' -or $_.Action_Taken -match 'sample values'
        })
        $fetchFail = @($todayRows | Where-Object { $_.Issue_Type -eq 'fetch_failure' })

        Write-Log "Data quality: $($fetchFail.Count) fetch failure(s), $($synthetic.Count) synthetic substitution(s) today."

        if ($synthetic.Count -gt 0) {
            Write-Log "Screener substituted SYNTHETIC data for $($synthetic.Count) ticker(s)." 'ERROR'
            Write-Log "Refusing to publish fabricated values. Discarding this run." 'ERROR'
            Invoke-Native 'git' @('checkout', '--', '.') | Out-Null
            Invoke-Native 'git' @('clean', '-fd', 'improvement/snapshots') | Out-Null
            Stop-Run "Run discarded: synthetic data detected. Check connectivity." 2
        }

        # Yahoo rate-limits routinely; a modest failure rate is normal and the
        # affected tickers are excluded rather than faked. A large one is not.
        $maxFailPct = 40
        $universe = 500
        $failPct = [math]::Round(100.0 * $fetchFail.Count / $universe, 1)
        if ($fetchFail.Count -gt ($universe * $maxFailPct / 100)) {
            Write-Log "Fetch failure rate ~$failPct% exceeds $maxFailPct%. Data too thin to publish." 'ERROR'
            Invoke-Native 'git' @('checkout', '--', '.') | Out-Null
            Invoke-Native 'git' @('clean', '-fd', 'improvement/snapshots') | Out-Null
            Stop-Run "Run discarded: fetch failure rate ~$failPct%." 2
        }
    } else {
        Write-Log "No data quality log found - cannot verify the run was real. Not publishing." 'ERROR'
        Invoke-Native 'git' @('checkout', '--', '.') | Out-Null
        Stop-Run "Run discarded: no data quality log." 2
    }

    # --- Run health check ----------------------------------------------------
    # Catches the failures the screener itself reports as success: a cached
    # warm-start that never fetched, missing prices/analyst targets, and a
    # collapse in category dispersion. See scripts/check_run_health.py - every
    # case it guards against actually happened between 08-06 and 08-10.
    Write-Log "Checking run health..."
    $health = Invoke-Native 'python' @('scripts/check_run_health.py')
    Write-NativeOutput $health

    if ($health.ExitCode -eq 1) {
        Write-Log "Run is DEGRADED. Refusing to publish." 'ERROR'
        Invoke-Native 'git' @('checkout', '--', '.') | Out-Null
        Invoke-Native 'git' @('clean', '-fd', 'improvement/snapshots') | Out-Null
        Stop-Run "Run discarded by health check - the live dashboard is unchanged." 2
    }
    elseif ($health.ExitCode -ne 0) {
        Write-Log "Health check could not evaluate this run (exit $($health.ExitCode))." 'ERROR'
        Invoke-Native 'git' @('checkout', '--', '.') | Out-Null
        Stop-Run "Run discarded - could not verify it was healthy." 2
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

    # Stage ONLY the artifacts this loop owns.
    #
    # This used to be `git add -A`, which on 2026-08-10 swept up a code
    # session's uncommitted work - CLAUDE.md, NIGHTLY_LOG.md, a research note,
    # ACTION_REQUIRED.md - and published it straight to main inside a "data:"
    # commit. That work had failed its clean-tree ship gate minutes earlier and
    # was deliberately left on a branch. The data loop must never be a back
    # door around the code loop's gates.
    $DataArtifacts = @(
        'dashboard.html',
        'index.html',
        'dashboard_data.js',
        'factor_output.xlsx',
        'factor_vol_history.csv',
        'sp500_tickers.json',
        'SCREENER_OVERVIEW.md',
        'README.md',
        'improvement',
        'validation'
    )
    foreach ($a in $DataArtifacts) {
        if (Test-Path (Join-Path $RepoPath $a)) {
            Invoke-Native 'git' @('add', '--', $a) | Out-Null
        }
    }

    $staged = Invoke-Native 'git' @('diff', '--cached', '--name-only')
    if (-not $staged.Text.Trim()) {
        Write-Log "Run produced no changes to data artifacts. Nothing to publish."
        Write-Log "=== Data loop complete ==="
        exit 0
    }
    Write-Log "Staging $(@($staged.Output | Where-Object { $_.Trim() }).Count) data artifact(s)."

    # Anything still unstaged belongs to someone else - report it, leave it.
    $unstaged = Invoke-Native 'git' @('status', '--porcelain')
    $foreign = @($unstaged.Output | Where-Object { $_ -and $_ -notmatch '^[AMD] ' -and $_.Trim() })
    if ($foreign.Count -gt 0) {
        Write-Log "Leaving $($foreign.Count) non-data file(s) untouched - not for this loop to publish:" 'WARN'
        foreach ($f in $foreign) { Write-Log "    $($f.Trim())" 'WARN' }
    }

    # A descriptive commit subject means the GitHub notification email is
    # actually informative, rather than just "data: screener run".
    $headline = "data: screener run $Date"
    try {
        $djs = Join-Path $RepoPath 'dashboard_data.js'
        if (Test-Path $djs) {
            $raw = Get-Content $djs -TotalCount 1
            if ($raw -match '"stocks_scored":\s*(\d+)') { $scored = $Matches[1] }
            $tickers = [regex]::Matches($raw, '"ticker":\s*"([A-Z.\-]+)"') |
                       Select-Object -First 5 | ForEach-Object { $_.Groups[1].Value }
            if ($scored) { $headline = "data: screener run $Date - $scored scored" }
            if ($tickers) { $headline += ", top: $($tickers -join ' ')" }
        }
    } catch { }

    $commit = Invoke-Native 'git' @('commit', '-m', $headline)
    if ($commit.ExitCode -ne 0) { Stop-Run "Commit failed." 2 }
    Write-Log "Committed run output: $headline"

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

    # Only a run that actually published counts as today's run. A discarded or
    # failed run leaves no marker, so a catch-up trigger will retry it.
    [System.IO.File]::WriteAllText($SuccessMarker, $Date)

    Write-Log "=== Data loop complete ==="
    exit 0
}
catch {
    Write-Log "Unhandled error: $($_.Exception.Message)" 'ERROR'
    Write-Log "$($_.ScriptStackTrace)" 'ERROR'
    exit 1
}
finally {
    Publish-Brief "data run $Date"
    if (Test-Path $LockFile) { Remove-Item $LockFile -Force -ErrorAction SilentlyContinue }
}
