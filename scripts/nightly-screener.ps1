<#
.SYNOPSIS
    Morning Screener improvement run (the "code loop"). Mon-Fri, 6:00 AM.

.DESCRIPTION
    Cuts a branch, hands Claude Code the day's focus prompt, then independently
    verifies the result before allowing it onto main.

    The session is autonomous and merges its own work. This script does not
    trust it to have checked correctly: it re-runs every ship gate itself and
    refuses the merge if any fails. On success it tags good/<date>, which is
    the rollback point.

    NOTE: keep this file ASCII-only and saved as UTF-8 with BOM. Windows
    PowerShell 5.1 reads a BOM-less script as cp1252, and any multi-byte
    character then decodes into bytes the parser mistakes for quotes.

.PARAMETER Focus
    Override the day's focus.

.PARAMETER DryRun
    Do everything except invoke Claude Code.

.PARAMETER NoMerge
    Run the session but never merge to main, whatever the gates say.
#>
[CmdletBinding()]
param(
    [string]$Focus,
    [switch]$DryRun,
    [switch]$NoMerge
)

# git/gh/claude write progress to stderr routinely. Under 'Stop', PowerShell
# 5.1 promotes that to a terminating error even on exit code 0, so we check
# $LASTEXITCODE explicitly instead.
$ErrorActionPreference = 'Continue'

# --- Configuration ---------------------------------------------------------
$RepoPath = Split-Path -Parent $PSScriptRoot
$LogDir   = Join-Path $RepoPath 'logs'
$LockFile = Join-Path $LogDir '.nightly.lock'
$Date     = Get-Date -Format 'yyyy-MM-dd'
$Stamp    = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$LogFile  = Join-Path $LogDir "nightly-$Stamp.log"

$FocusByDay = @{
    'Monday'    = 'RESEARCH. Pick one open question from the priorities list in CLAUDE.md. Read real literature and peer tools. Produce a dated note in research/ with citations, effect sizes, and a recommendation. Do not write production code today.'
    'Tuesday'   = 'RESEARCH TO DESIGN. Turn the strongest evidence from Monday into a concrete, falsifiable design. State the hypothesis and how a backtest would refute it. Append it to the research note. Still no production code.'
    'Wednesday' = 'BUILD. Implement the highest-value item the week''s research justified. Write tests alongside the code.'
    'Thursday'  = 'VALIDATE. Measure whether this week''s change actually beat baseline - backtest, IC, or regression. Be willing to conclude it did not, and revert it if so. Record the number either way.'
    'Friday'    = 'HARDEN AND TEACH. Tests, docs, error handling, and the investment-club experience. Would a finance student understand what they are looking at?'
    'Saturday'  = 'CATCH-UP. Not normally scheduled. Work the single highest-value item from the priorities list.'
    'Sunday'    = 'CATCH-UP. Not normally scheduled. Work the single highest-value item from the priorities list.'
}

# Published artifacts the test suite historically rewrote. conftest.py now
# guards these, but the check is kept as defence in depth.
$Artifacts = @(
    'validation/data_quality_log.csv',
    'factor_output.xlsx',
    'sp500_tickers.json'
)

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -ErrorAction Stop | Out-Null }

# --- Helpers ---------------------------------------------------------------
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

function Restore-Artifacts {
    param([string]$Context)
    $restored = @()
    foreach ($f in $Artifacts) {
        if ((Invoke-Native 'git' @('status', '--porcelain', '--', $f)).Text.Trim()) {
            Invoke-Native 'git' @('checkout', '--', $f) | Out-Null
            $restored += $f
        }
    }
    if ($restored.Count -gt 0) {
        Write-Log "Reverted uncommitted changes to published artifact(s) ($Context): $($restored -join ', ')"
    }
}

# --- PATH repair -----------------------------------------------------------
$env:Path = "$([Environment]::GetEnvironmentVariable('Path','Machine'));$([Environment]::GetEnvironmentVariable('Path','User'))"

# --- Single-instance lock --------------------------------------------------
if (Test-Path $LockFile) {
    $lockAge = (Get-Date) - (Get-Item $LockFile).LastWriteTime
    if ($lockAge.TotalHours -lt 8) {
        Write-Log "A run started $([int]$lockAge.TotalMinutes) minutes ago still holds the lock. Exiting." 'WARN'
        exit 0
    }
    Write-Log "Stale lock ($([int]$lockAge.TotalHours)h). Reclaiming." 'WARN'
    Remove-Item $LockFile -Force
}
Set-Content -Path $LockFile -Value $PID -Encoding utf8

try {
    Set-Location $RepoPath

    # --- Retrospective every other Friday -----------------------------------
    # The routine periodically evaluates whether it is actually producing value
    # and rewrites its own process. Uses a different prompt template.
    $cal = [System.Globalization.CultureInfo]::InvariantCulture.Calendar
    $weekNo = $cal.GetWeekOfYear(
        (Get-Date),
        [System.Globalization.CalendarWeekRule]::FirstFourDayWeek,
        [System.DayOfWeek]::Monday)
    $IsRetro = ((Get-Date).DayOfWeek -eq [System.DayOfWeek]::Friday) -and ($weekNo % 2 -eq 0)

    # A retrospective with nothing to review is just churn. Require a real
    # history of sessions first - entries in NIGHTLY_LOG.md are the counter.
    $MinSessionsForRetro = 6
    if ($IsRetro) {
        $logPath = Join-Path $RepoPath 'NIGHTLY_LOG.md'
        $entries = 0
        if (Test-Path $logPath) {
            $entries = @(Select-String -Path $logPath -Pattern '^## \d{4}-\d{2}-\d{2}').Count
        }
        if ($entries -lt $MinSessionsForRetro) {
            Write-Log "Retrospective due, but only $entries session(s) logged (need $MinSessionsForRetro). Doing normal Friday work instead."
            $IsRetro = $false
        }
    }

    if (-not $Focus) {
        if ($IsRetro) {
            $Focus = 'RETROSPECTIVE. Evaluate whether this routine is actually producing value, and change the process where it is not.'
        } else {
            $Focus = $FocusByDay[(Get-Date).DayOfWeek.ToString()]
        }
    }

    Write-Log "=== Code loop $Date (ISO week $weekNo) ==="
    if ($IsRetro) { Write-Log "RETROSPECTIVE SESSION - self-evaluation, not normal work" }
    Write-Log "Focus: $Focus"

    foreach ($tool in @('git', 'claude', 'python')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            Stop-Run "Required tool '$tool' is not on PATH. Aborting."
        }
    }

    Restore-Artifacts 'pre-existing'

    # --- Preflight: can we actually reach the remote? -----------------------
    # Deliberately NOT 'gh auth status'. gh keeps its token in the Windows
    # keyring, which a scheduled task cannot reliably read - so gh reports
    # "not authenticated" at 06:00 even though it works fine interactively.
    # That aborted the first real run for no good reason.
    #
    # This loop merges with plain git and pushes via the 'manager' credential
    # helper, so gh is not needed at all. What matters is whether the remote
    # is reachable, which is what this checks.
    $ls = Invoke-Native 'git' @('ls-remote', '--heads', 'origin', 'main')
    if ($ls.ExitCode -ne 0) {
        Write-NativeOutput $ls 'ERROR'
        Stop-Run "Cannot reach the git remote (offline, or credentials unavailable). Skipping today's run."
    }
    Write-Log "Remote reachable."

    $status = Invoke-Native 'git' @('status', '--porcelain')
    if ($status.Text.Trim()) {
        Write-Log "Working tree is not clean. Refusing to run." 'ERROR'
        Write-NativeOutput $status 'ERROR'
        Stop-Run "Resolve by hand; the next scheduled run will proceed."
    }

    # --- Sync main ----------------------------------------------------------
    $co = Invoke-Native 'git' @('checkout', 'main'); Write-NativeOutput $co
    if ($co.ExitCode -ne 0) { Stop-Run "git checkout main failed." }

    $pull = Invoke-Native 'git' @('pull', '--ff-only'); Write-NativeOutput $pull
    if ($pull.ExitCode -ne 0) { Stop-Run "git pull --ff-only failed - main diverged locally. Resolve by hand." }

    $BaseSha = (Invoke-Native 'git' @('rev-parse', 'HEAD')).Text.Trim()
    Write-Log "main is at $($BaseSha.Substring(0,8))"

    # --- Branch -------------------------------------------------------------
    $Branch = "nightly/$Date"
    $suffix = 2
    while ((Invoke-Native 'git' @('rev-parse', '--verify', '--quiet', "refs/heads/$Branch")).ExitCode -eq 0) {
        $Branch = "nightly/$Date-$suffix"; $suffix++
    }
    $br = Invoke-Native 'git' @('checkout', '-b', $Branch); Write-NativeOutput $br
    if ($br.ExitCode -ne 0) { Stop-Run "Failed to create branch $Branch." }
    Write-Log "Working on $Branch"

    # --- Prompt -------------------------------------------------------------
    $TemplateName = if ($IsRetro) { 'retrospective.md' } else { 'nightly.md' }
    $TemplatePath = Join-Path $RepoPath ".claude\prompts\$TemplateName"
    if (-not (Test-Path $TemplatePath)) { Stop-Run "Prompt template missing at $TemplatePath." }
    Write-Log "Using prompt template: $TemplateName"
    $Prompt = (Get-Content $TemplatePath -Raw).
        Replace('{{FOCUS}}', $Focus).Replace('{{DATE}}', $Date).Replace('{{BRANCH}}', $Branch)

    if ($DryRun) {
        Write-Log "DryRun: prompt resolved to $($Prompt.Length) chars. Reverting branch."
        Invoke-Native 'git' @('checkout', 'main') | Out-Null
        Invoke-Native 'git' @('branch', '-D', $Branch) | Out-Null
        Write-Log "=== Dry run complete ==="
        if (Test-Path $LockFile) { Remove-Item $LockFile -Force -ErrorAction SilentlyContinue }
        exit 0
    }

    # --- Run the session ----------------------------------------------------
    Write-Log "Invoking Claude Code..."
    $JsonLog = Join-Path $LogDir "nightly-$Stamp.json"
    $Prompt | claude --print --permission-mode acceptEdits --output-format json |
        Out-File -FilePath $JsonLog -Encoding utf8
    $claudeExit = $LASTEXITCODE
    Write-Log "Claude Code exited $claudeExit. Transcript: $JsonLog"

    Restore-Artifacts 'left by session test runs'

    # =======================================================================
    #  SHIP GATES - re-verified here, independently of what the session
    #  believed. Nothing reaches main unless all of these pass.
    # =======================================================================
    Write-Log "--- Ship gates ---"
    $gateFailures = @()

    # The session may have merged to main itself. Verify whichever branch
    # actually holds the work.
    $head = (Invoke-Native 'git' @('rev-parse', '--abbrev-ref', 'HEAD')).Text.Trim()
    Write-Log "HEAD is on '$head'"

    # Gate 1: tests
    $t = Invoke-Native 'python' @('-m', 'pytest', 'tests/', 'test_screener.py', '-q')
    $tail = ($t.Output | Select-Object -Last 3) -join ' '
    if ($t.ExitCode -eq 0) { Write-Log "GATE 1 tests: PASS ($tail)" }
    else { Write-Log "GATE 1 tests: FAIL ($tail)" 'ERROR'; $gateFailures += 'tests' }

    # Gate 2: pipeline wiring
    $d = Invoke-Native 'python' @('run_screener.py', '--dry-run')
    if ($d.ExitCode -eq 0) { Write-Log "GATE 2 dry-run: PASS" }
    else { Write-Log "GATE 2 dry-run: FAIL" 'ERROR'; Write-NativeOutput $d 'ERROR'; $gateFailures += 'dry-run' }

    # A dry-run refreshes the ticker cache; that is production behaviour, not a change to ship.
    Restore-Artifacts 'refreshed by gate 2'

    # Gate 3: dashboard artifacts still intact
    $indexPath = Join-Path $RepoPath 'index.html'
    $dataPath  = Join-Path $RepoPath 'dashboard_data.js'
    $g3 = $true
    foreach ($pair in @(@($indexPath, 50000), @($dataPath, 100000))) {
        $p = $pair[0]; $min = $pair[1]
        if (-not (Test-Path $p)) { Write-Log "GATE 3: $p missing" 'ERROR'; $g3 = $false; continue }
        $len = (Get-Item $p).Length
        if ($len -lt $min) { Write-Log "GATE 3: $p is only $len bytes (expected >$min)" 'ERROR'; $g3 = $false }
    }
    if ($g3 -and (Test-Path $dataPath)) {
        $firstChars = (Get-Content $dataPath -TotalCount 1) -replace '\s', ''
        if ($firstChars -notmatch '^window\.SCREENER_DATA=') {
            Write-Log "GATE 3: dashboard_data.js no longer starts with the expected assignment" 'ERROR'
            $g3 = $false
        }
    }
    if ($g3) { Write-Log "GATE 3 dashboard artifacts: PASS" } else { $gateFailures += 'dashboard' }

    # Gate 4: nothing stray left behind
    $leftover = Invoke-Native 'git' @('status', '--porcelain')
    if ($leftover.Text.Trim()) {
        Write-Log "GATE 4 clean tree: FAIL - uncommitted changes remain" 'ERROR'
        Write-NativeOutput $leftover 'ERROR'
        $gateFailures += 'clean-tree'
    } else { Write-Log "GATE 4 clean tree: PASS" }

    # --- Decide -------------------------------------------------------------
    $workBranch = if ($head -eq 'main') { 'main' } else { $head }
    $commits = Invoke-Native 'git' @('log', '--oneline', "$BaseSha..HEAD")
    $commitCount = @($commits.Output | Where-Object { $_.Trim() }).Count
    Write-Log "$commitCount new commit(s) since main was at $($BaseSha.Substring(0,8))"
    if ($commitCount -gt 0) { Write-NativeOutput $commits }

    if ($gateFailures.Count -gt 0) {
        Write-Log "SHIP GATES FAILED: $($gateFailures -join ', '). Not merging." 'ERROR'
        if ($workBranch -eq 'main') {
            Write-Log "Session had already committed onto local main. Resetting local main back to $($BaseSha.Substring(0,8)); work preserved on $Branch." 'WARN'
            Invoke-Native 'git' @('branch', '-f', $Branch, 'HEAD') | Out-Null
            Invoke-Native 'git' @('reset', '--hard', $BaseSha) | Out-Null
        }
        $push = Invoke-Native 'git' @('push', '-u', 'origin', $Branch)
        Write-NativeOutput $push
        Write-Log "Work pushed to $Branch for inspection. main is untouched." 'WARN'
        Stop-Run "Run finished with failing gates - see above." 2
    }

    if ($commitCount -eq 0) {
        Write-Log "No commits produced. Nothing to ship." 'WARN'
        Invoke-Native 'git' @('checkout', 'main') | Out-Null
        Invoke-Native 'git' @('branch', '-D', $Branch) | Out-Null
        Write-Log "=== Run complete (no changes) ==="
        exit 0
    }

    if ($NoMerge) {
        Invoke-Native 'git' @('push', '-u', 'origin', $Branch) | Out-Null
        Write-Log "-NoMerge set: pushed $Branch, leaving main alone."
        exit 0
    }

    # --- Merge to main ------------------------------------------------------
    Write-Log "All gates passed. Merging to main."
    if ($workBranch -ne 'main') {
        Invoke-Native 'git' @('checkout', 'main') | Out-Null
        $merge = Invoke-Native 'git' @('merge', '--no-ff', $Branch, '-m', "nightly $Date - $($Focus.Split('.')[0])")
        Write-NativeOutput $merge
        if ($merge.ExitCode -ne 0) {
            Invoke-Native 'git' @('merge', '--abort') | Out-Null
            Invoke-Native 'git' @('push', '-u', 'origin', $Branch) | Out-Null
            Stop-Run "Merge conflict. Left on $Branch; main untouched." 2
        }
    }

    $push = Invoke-Native 'git' @('push', 'origin', 'main')
    Write-NativeOutput $push
    if ($push.ExitCode -ne 0) { Stop-Run "Push to main failed. Local main is ahead; resolve by hand." 2 }

    # --- Tag the rollback point ---------------------------------------------
    $tag = "good/$Date"
    if ((Invoke-Native 'git' @('rev-parse', '--verify', '--quiet', "refs/tags/$tag")).ExitCode -eq 0) {
        $tag = "good/$Date-$([DateTime]::Now.ToString('HHmm'))"
    }
    Invoke-Native 'git' @('tag', '-a', $tag, '-m', "All ship gates passed $Stamp") | Out-Null
    Invoke-Native 'git' @('push', 'origin', $tag) | Out-Null
    Write-Log "Tagged $tag - roll back with: git reset --hard $tag"

    Invoke-Native 'git' @('branch', '-d', $Branch) | Out-Null
    Write-Log "=== Run complete: shipped to main ==="
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
