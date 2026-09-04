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
    [switch]$NoMerge,
    [switch]$Force
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
    'Monday'    = 'RESEARCH. Take one specific thing - a factor, a metric, a threshold, a construction rule - and learn it properly, from the literature AND from documented practice, in this one session. Real citations, effect sizes, the conditions the effect held under, and how quant shops and institutional screens actually handle it. Where academia and practice disagree, say so and say why. A dated note in research/, complete today. No production code.'
    'Tuesday'   = 'PRODUCT. Open the live dashboard as a user would. Does it answer what should I look at / should I buy this / should I sell what I hold / how much? Read plan/dashboard-inventory.md before building anything - the most likely failure is rebuilding what exists. Ship a dashboard change, or write down precisely what it cannot answer and why.'
    'Wednesday' = 'SYNTHESIS. How does this fit the rest of the screener? What does it overlap with, what does it make redundant, what does it imply for the other seven categories? Design the coherent whole, not the isolated tweak. Record any methodology change in METHODOLOGY_CHANGELOG.md with its sources.'
    'Thursday'  = 'BUILD. Implement what the week''s research justified. Write tests alongside the code.'
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

# Writes MORNING_BRIEF.md and publishes it. Called from `finally`, so the owner
# gets a summary whether the run succeeded, was stopped by a gate, or crashed -
# a failed run is the one you most want to be told about.
function Publish-Brief {
    param([string]$Label)
    try {
        Write-Log "Writing morning brief..."
        $b = Invoke-Native 'python' @('scripts/write_brief.py')
        Write-NativeOutput $b
        if ($b.ExitCode -ne 0) { return }

        # Nothing is staged here on purpose. The tree may legitimately be dirty
        # after a failed run, and the publisher below compares the brief against
        # origin/main's own tree rather than against this machine's index.
        #
        # Publish onto origin/main by building a single-file commit there - see
        # scripts/publish-brief.ps1. This used to be `git push origin HEAD:main`,
        # which on a gate failure pushes the whole refused branch to the public
        # site from inside `finally`.
        Publish-BriefToMain -RepoPath $RepoPath -Label $Label -Logger ${function:Write-Log} | Out-Null
    } catch {
        Write-Log "Brief step failed (non-fatal): $($_.Exception.Message)" 'WARN'
    }
}

# Reads the CLI's own JSON transcript and decides whether a session actually
# ran. A session that never started is indistinguishable from one that found
# nothing to do: no commits, clean tree, and all four gates pass trivially.
# On 2026-08-14 the CLI exited 1 after one second on an API weekly-limit 429
# (`"api_error_status": 429`, `num_turns: 1`). This script logged
# "Run complete (no changes)", wrote the success marker, and published a normal
# morning brief - so the owner's only status channel said the morning was fine
# when in fact nothing had happened. Failure must never report as success.
function Get-SessionOutcome {
    param([string]$TranscriptPath, [int]$ExitCode)

    $ok     = $true
    $reason = ''

    if ($ExitCode -ne 0) { $ok = $false; $reason = "Claude Code exited $ExitCode" }

    $len = if (Test-Path $TranscriptPath) { (Get-Item $TranscriptPath).Length } else { -1 }
    if ($len -lt 2) {
        $ok = $false
        if (-not $reason) { $reason = "transcript is empty or missing ($len bytes)" }
    } else {
        try {
            $tr = Get-Content $TranscriptPath -Raw | ConvertFrom-Json
            if ($tr.is_error) {
                $ok = $false
                $detail = if ($tr.api_error_status) { "API error $($tr.api_error_status)" } else { 'session reported an error' }
                $msg = ([string]$tr.result) -replace '\s+', ' '
                if ($msg.Length -gt 160) { $msg = $msg.Substring(0, 160) }
                $reason = ("$detail - $msg").Trim()
            }
            # A healthy session is many turns. One turn means it died on the
            # first request, whatever it claimed in its exit code.
            if ($null -ne $tr.num_turns -and [int]$tr.num_turns -le 1) {
                $ok = $false
                if (-not $reason) { $reason = "session made only $($tr.num_turns) turn(s) - it never started work" }
            }
        } catch {
            $ok = $false
            if (-not $reason) { $reason = "transcript is not valid JSON: $($_.Exception.Message)" }
        }
    }

    return [pscustomobject]@{ Ok = $ok; Reason = $reason }
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

# Mutual exclusion against the data loop. Both tasks carry an at-logon
# catch-up trigger, so on the first logon of the day they start together and
# race for .git/index.lock. On 2026-08-29 this script's Restore-Artifacts was
# holding the index when the data loop tried to check out main, and the data
# run died. See scripts/repo-lock.ps1.
. (Join-Path $PSScriptRoot 'repo-lock.ps1')
$RepoLock = $null

# The brief is published from `finally`, so it runs after a gate failure too.
# It must therefore never be able to carry the refused work with it.
. (Join-Path $PSScriptRoot 'publish-brief.ps1')

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

# Set once the session has been invoked and its transcript judged. Declared
# here so `finally` can read them however the run ends.
$SessionFailed = $false
$SessionError  = ''

# --- Run-once-per-day guard --------------------------------------------------
# These tasks use InteractiveToken, so they only fire while a user is logged on.
# On 2026-08-12 the machine rebooted at 00:47 (Windows Update) and sat at the
# login screen; both the 02:00 and 06:00 runs were simply lost. A catch-up
# logon trigger fixes that, but then needs this guard so logging in three times
# does not run a session three times.
$SuccessMarker = Join-Path $LogDir '.nightly-last-success'
if ((Test-Path $SuccessMarker) -and -not $Force -and -not $DryRun) {
    # Strip a UTF-8 BOM: Set-Content -Encoding utf8 writes one in PS 5.1 and
    # .Trim() does not remove it, so this comparison silently never matched.
    $last = (Get-Content $SuccessMarker -TotalCount 1).TrimStart([char]0xFEFF).Trim()
    if ($last -eq $Date) {
        Write-Log "Code loop already completed today ($Date). Nothing to do."
        if (Test-Path $LockFile) { Remove-Item $LockFile -Force -ErrorAction SilentlyContinue }
        exit 0
    }
}

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

    # --- Take the repo before the first git command -------------------------
    # Restore-Artifacts, immediately below, runs `git status` - which takes
    # .git/index.lock to refresh the index. That is the exact command that
    # killed the 2026-08-29 data run.
    #
    # A code session is the scarce resource (it draws on a weekly Claude usage
    # ceiling; the data loop is pure Python and git), so it waits for the data
    # loop rather than giving up the day. The wait costs at most one data run,
    # measured at 11.8-13.6 minutes, out of a 4-hour execution limit.
    $RepoLock = Enter-RepoLock -LogDir $LogDir -Holder 'code loop' `
                    -Logger { param($m, $l) Write-Log $m $l }
    if (-not $RepoLock) {
        Stop-Run "Could not get exclusive use of the repo. Not running git alongside another loop."
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

    # --- Preflight: is this folder trusted by the CLI? ----------------------
    # Claude Code keys folder trust by path in %USERPROFILE%\.claude.json. The
    # desktop app writes backslashes; the CLI reads forward slashes. If the
    # CLI's key is not trusted, the session starts but every permission in
    # .claude/settings.json is ignored - python, pytest and git are all denied
    # at runtime. On 2026-08-10 that burned a full session: it researched for
    # 12 minutes, could not commit, and failed the clean-tree gate.
    #
    # Checked here so a jammed run costs seconds instead of a whole session.
    # NOTE: patching this flag from a script does NOT persist - Claude Code
    # rewrites the file from its own state. Trust must be granted through the
    # CLI's own dialog, in an interactive session started inside this folder.
    $cfgPath = Join-Path $env:USERPROFILE '.claude.json'
    if (Test-Path $cfgPath) {
        try {
            $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
            $fwdKey = $RepoPath.Replace('\', '/')
            $entry = $cfg.projects.PSObject.Properties | Where-Object { $_.Name -eq $fwdKey }
            if ($entry -and -not $entry.Value.hasTrustDialogAccepted) {
                Write-Log "This folder is NOT trusted by the Claude Code CLI." 'ERROR'
                Write-Log "The session would run but be denied python, pytest and git." 'ERROR'
                Write-Log "Fix (interactively, with no other Claude session open):" 'ERROR'
                Write-Log "    cd `"$RepoPath`"" 'ERROR'
                Write-Log "    & `"`$env:APPDATA\npm\claude.cmd`"" 'ERROR'
                Write-Log "  then answer YES to the trust prompt and /exit." 'ERROR'
                Stop-Run "Skipping today's run rather than burning a session on a jammed workspace."
            }
        } catch {
            Write-Log "Could not read .claude.json to verify trust: $($_.Exception.Message)" 'WARN'
        }
    }
    Write-Log "Workspace trust OK."

    # --- Preflight: recover a dirty tree ------------------------------------
    # A session that cannot commit leaves the tree dirty. Refusing to run on a
    # dirty tree is right, but refusing *forever* turns one bad night into a
    # permanently jammed loop - which is exactly what happened after
    # 2026-08-10. Stash the leftovers instead: nothing is lost, and tomorrow
    # starts clean.
    $status = Invoke-Native 'git' @('status', '--porcelain')
    if ($status.Text.Trim()) {
        Write-Log "Working tree is dirty - probably leftovers from a failed session:" 'WARN'
        Write-NativeOutput $status 'WARN'
        $stash = Invoke-Native 'git' @('stash', 'push', '-u', '-m', "auto-rescue $Stamp")
        Write-NativeOutput $stash 'WARN'
        if ($stash.ExitCode -ne 0) {
            Stop-Run "Could not stash the leftovers. Resolve by hand: git status"
        }
        Write-Log "Stashed as 'auto-rescue $Stamp'. Recover with: git stash list / git stash pop" 'WARN'
    }

    $status = Invoke-Native 'git' @('status', '--porcelain')
    if ($status.Text.Trim()) {
        Write-Log "Working tree still not clean after stashing. Refusing to run." 'ERROR'
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

    # --- Sweep stray merged nightly/* branches -------------------------------
    # The branch this run creates below is deleted on success (see the tag
    # step). The exit code of that delete was never checked, so a rare failure -
    # a transient git lock, an interrupted run - left debris with no log line
    # and no visibility; nightly/2026-08-10 and nightly/2026-08-27 were found
    # this way on 2026-09-01, both fully merged, neither cleaned up. Sweeping
    # here makes that self-healing instead of something an interactive session
    # has to notice: any local nightly/* branch already merged into main is
    # stale by definition and safe to remove with 'git branch -d' (the safe
    # form - it refuses anything not actually merged).
    $stray = (Invoke-Native 'git' @('branch', '--merged', 'main')).Output |
              ForEach-Object { $_.ToString().Trim(' *') } |
              Where-Object { $_ -like 'nightly/*' }
    foreach ($b in $stray) {
        $del = Invoke-Native 'git' @('branch', '-d', $b)
        if ($del.ExitCode -eq 0) { Write-Log "Swept stale merged branch: $b" }
    }

    # --- Branch -------------------------------------------------------------
    $Branch = "nightly/$Date"
    $suffix = 2
    while ((Invoke-Native 'git' @('rev-parse', '--verify', '--quiet', "refs/heads/$Branch")).ExitCode -eq 0) {
        $Branch = "nightly/$Date-$suffix"; $suffix++
    }
    $br = Invoke-Native 'git' @('checkout', '-b', $Branch); Write-NativeOutput $br
    if ($br.ExitCode -ne 0) { Stop-Run "Failed to create branch $Branch." }
    Write-Log "Working on $Branch"

    # --- Sweep stray merged nightly/* branches on the REMOTE -----------------
    # The local sweep above only cleans this machine. Each session pushes its
    # working branch to origin before merging, and nothing ever deleted it
    # there, so origin accumulated one dead branch per session indefinitely -
    # 11 of them (nightly/2026-08-10 .. nightly/2026-09-02) by 2026-09-03, all
    # fully merged. Noticed by the 2026-09-02 evening session and left as
    # something to do by hand, which is exactly the pattern rule 11 exists to
    # stop; sweeping here makes it self-healing.
    #
    # Safety: '--merged origin/main' means every commit on the branch is
    # already reachable from main, so deleting the ref discards no history and
    # is not a rewrite (rule 2). $Branch is skipped - an earlier same-day run
    # may have pushed a merged branch of that name, and this run is about to
    # push its own. A failed delete is a WARN, never fatal: dead refs on origin
    # are untidy, not dangerous, and are swept again next run.
    $fetchPrune = Invoke-Native 'git' @('fetch', '--prune', 'origin')
    if ($fetchPrune.ExitCode -ne 0) {
        Write-Log "WARN: git fetch --prune failed; skipping remote branch sweep."
    } else {
        $strayRemote = (Invoke-Native 'git' @('branch', '-r', '--merged', 'origin/main')).Output |
                        ForEach-Object { $_.ToString().Trim(' *') } |
                        Where-Object { $_ -like 'origin/nightly/*' } |
                        ForEach-Object { $_.Substring('origin/'.Length) } |
                        Where-Object { $_ -ne $Branch }
        foreach ($rb in $strayRemote) {
            $rdel = Invoke-Native 'git' @('push', 'origin', '--delete', $rb)
            if ($rdel.ExitCode -eq 0) {
                Write-Log "Swept stale merged remote branch: origin/$rb"
            } else {
                Write-Log "WARN: could not delete remote branch origin/$rb - will retry next run."
            }
        }
    }

    # --- Prompt -------------------------------------------------------------
    $TemplateName = if ($IsRetro) { 'retrospective.md' } else { 'nightly.md' }
    $TemplatePath = Join-Path $RepoPath "prompts\$TemplateName"
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

    $session = Get-SessionOutcome -TranscriptPath $JsonLog -ExitCode $claudeExit
    $SessionFailed = -not $session.Ok
    $SessionError  = $session.Reason
    if ($SessionFailed) {
        Write-Log "SESSION DID NOT RUN: $SessionError" 'ERROR'
        Write-Log "Treating this as a failure, not as 'nothing to do'. No success marker will be written," 'ERROR'
        Write-Log "so the catch-up trigger will retry rather than skipping the day." 'ERROR'
    }

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
    # NOTE (2026-08-21 retrospective): CLAUDE.md and the nightly prompt both
    # describe this gate as "dashboard_data.js parses", but the check above only
    # regex-matches the first line - a truncated 3 MB payload sails through it.
    # A node-based parse was written and then deliberately NOT shipped: neither
    # PowerShell nor node could be executed in that session, and an unverified
    # change here can only fail *closed*, refusing every future merge. Jamming
    # the loop is worse than the weaker check. See NIGHTLY_LOG.md 2026-08-21.
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

        # 2026-09-02: prompts/nightly.md has the session push main itself as
        # its own last step, on the happy path, before this independent
        # re-check ever runs. A local reset cannot undo a push the session
        # already made - "main is untouched" used to be printed unconditionally
        # here, which was simply false the one time it mattered. Check what is
        # actually live on origin and, if the session already pushed before
        # this gate failure was caught, revert it there - a new commit, never
        # a rewrite (rule 2). scripts/revert-bad-merge.ps1 is deliberately
        # conservative: it only pushes if the reverted tree verifies as
        # byte-identical to $BaseSha, and does nothing (safely) if origin/main
        # was never touched, which is the common case this whole block exists
        # for.
        $revertScript = Join-Path $RepoPath 'scripts\revert-bad-merge.ps1'
        $rv = Invoke-Native 'powershell' @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $revertScript,
                                            '-BaseSha', $BaseSha, '-RepoPath', $RepoPath)
        Write-NativeOutput $rv
        if ($rv.ExitCode -eq 0) {
            Write-Log "Work pushed to $Branch for inspection. origin/main verified clean (reverted if the earlier push had reached it)." 'WARN'
        } else {
            Write-Log "origin/main may still carry the failing commits and the automatic revert could not verify a clean fix - this needs a human right now." 'ERROR'
        }
        Stop-Run "Run finished with failing gates - see above." 2
    }

    if ($commitCount -eq 0) {
        Invoke-Native 'git' @('checkout', 'main') | Out-Null
        Invoke-Native 'git' @('branch', '-D', $Branch) | Out-Null
        if ($SessionFailed) {
            Write-Log "No commits, because the session never ran. This is a lost day, not a quiet one." 'ERROR'
            Stop-Run "=== Run FAILED: $SessionError ===" 2
        }
        Write-Log "No commits produced. Nothing to ship." 'WARN'
        [System.IO.File]::WriteAllText($SuccessMarker, $Date)
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

    $delBranch = Invoke-Native 'git' @('branch', '-d', $Branch)
    if ($delBranch.ExitCode -ne 0) {
        # Not fatal - the sweep at the top of the next run picks it up - but
        # this used to fail silently into Out-Null, which is exactly how
        # nightly/2026-08-10 and nightly/2026-08-27 went unnoticed for weeks.
        Write-Log "Could not delete $Branch after merge (the next run sweep picks it up)." 'WARN'
    }
    if (-not $SessionFailed) { [System.IO.File]::WriteAllText($SuccessMarker, $Date) }
    Write-Log "=== Run complete: shipped to main ==="
    exit 0
}
catch {
    Write-Log "Unhandled error: $($_.Exception.Message)" 'ERROR'
    Write-Log "$($_.ScriptStackTrace)" 'ERROR'
    exit 1
}
finally {
    $briefLabel = if ($SessionFailed) { "code session $Date - SESSION DID NOT RUN" } else { "code session $Date" }
    # Publish-Brief runs git, so the repo lock is released after it, not before.
    Publish-Brief $briefLabel
    Exit-RepoLock $RepoLock
    if (Test-Path $LockFile) { Remove-Item $LockFile -Force -ErrorAction SilentlyContinue }
}
