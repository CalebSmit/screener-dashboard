<#
.SYNOPSIS
    Mutual exclusion between the data loop and the code loop.

.DESCRIPTION
    Dot-sourced by data-run.ps1 and nightly-screener.ps1. Both operate on the
    same working tree, and git serialises nothing for them: a second git
    process that wants the index simply fails.

    WHY THIS EXISTS - 2026-08-29.

    register-tasks.ps1 gives both tasks an at-logon catch-up trigger with the
    same PT3M delay, so on the first logon of the day they start in the same
    second. On 2026-08-29 they did:

        logs/datarun-2026-08-29_121115.log   [12:11:15] === Data loop ===
        logs/nightly-2026-08-29_121115.log   [12:11:15] === Code loop ===

    At 12:11:16 the data loop ran `git checkout main` while the code loop was
    inside Restore-Artifacts running `git status`, which takes .git/index.lock
    to refresh the index. git exits 128:

        fatal: Unable to create '.../.git/index.lock': File exists.

    The data loop treated that as fatal and stopped before running the
    screener at all. The catch-up trigger added to stop days being lost had
    become a way to lose one.

    Each script already had a single-instance lock of its own (.datarun.lock,
    .nightly.lock). Those stop a loop racing *itself*; nothing stopped the two
    loops racing *each other*. This is that missing lock.

    ORDERING. Whoever asks first wins, and the loser waits rather than dying.
    Waiting is right because both loops are short - measured over 2026-08-21 to
    2026-08-28, data runs took 11.8-13.6 min and code sessions 16.9-25.9 min -
    against task execution limits of 3h and 4h. The default 60-minute wait is
    over twice the longest observed hold, and giving up loses the day, which is
    the outcome this whole file exists to prevent.

    RECLAIM. A lock whose owning process is gone is reclaimed immediately: a
    crashed run must not jam tomorrow. PID reuse is guarded by a hard age
    ceiling beyond either task's execution limit.

    NOTE: keep this file ASCII-only and saved as UTF-8 with BOM.
#>

function Get-RepoLockPath {
    param([Parameter(Mandatory)][string]$LogDir)
    Join-Path $LogDir '.repo.lock'
}

function Write-RepoLockLog {
    param([string]$Message, [string]$Level = 'INFO', [scriptblock]$Logger)
    if ($Logger) { & $Logger $Message $Level }
    else { Write-Host ("[{0}] {1}" -f $Level, $Message) }
}

function Read-RepoLockInfo {
    param([Parameter(Mandatory)][string]$Path)
    $info = [pscustomobject]@{ OwnerPid = 0; Holder = 'unknown'; Age = $null }
    try {
        $lines = @(Get-Content -Path $Path -TotalCount 2 -ErrorAction Stop)
        if ($lines.Count -ge 1) {
            $parsed = 0
            if ([int]::TryParse($lines[0].TrimStart([char]0xFEFF).Trim(), [ref]$parsed)) {
                $info.OwnerPid = $parsed
            }
        }
        if ($lines.Count -ge 2 -and $lines[1].Trim()) { $info.Holder = $lines[1].Trim() }
    } catch {
        # An unreadable lock is handled by the age ceiling, not by guessing.
    }
    try { $info.Age = (Get-Date) - (Get-Item $Path -ErrorAction Stop).LastWriteTime } catch { }
    return $info
}

function Test-RepoLockOwnerAlive {
    param([int]$OwnerPid)
    if ($OwnerPid -le 0) { return $false }
    return [bool](Get-Process -Id $OwnerPid -ErrorAction SilentlyContinue)
}

<#
Acquire the shared repo lock. Returns the lock path on success, or $null if
the wait expired - the caller decides what a timeout means for it.
#>
function Enter-RepoLock {
    param(
        [Parameter(Mandatory)][string]$LogDir,
        [Parameter(Mandatory)][string]$Holder,
        [int]$MaxWaitMinutes = 60,
        [int]$PollSeconds = 20,
        [int]$MaxHoldHours = 6,
        [scriptblock]$Logger
    )

    $path     = Get-RepoLockPath $LogDir
    $deadline = (Get-Date).AddMinutes($MaxWaitMinutes)
    $waited   = $false

    while ($true) {
        $stream = $null
        try {
            # CreateNew is atomic and fails if the file exists. Test-Path then
            # Set-Content is not: two processes three seconds apart both pass
            # the test, which is precisely the case this guards.
            $stream = [System.IO.File]::Open(
                $path,
                [System.IO.FileMode]::CreateNew,
                [System.IO.FileAccess]::Write,
                [System.IO.FileShare]::None)
        } catch {
            $stream = $null
        }

        if ($stream) {
            try {
                $text  = "{0}`n{1}`n{2}`n" -f $PID, $Holder, (Get-Date -Format 'o')
                $bytes = [System.Text.Encoding]::ASCII.GetBytes($text)
                $stream.Write($bytes, 0, $bytes.Length)
            } finally {
                $stream.Dispose()
            }
            if ($waited) { Write-RepoLockLog "Repo lock acquired." 'INFO' $Logger }
            return $path
        }

        if (-not (Test-Path $path)) {
            # The create failed for some reason other than contention - a
            # missing directory, or permissions. Looping would spin forever.
            Write-RepoLockLog "Cannot create the repo lock at $path." 'ERROR' $Logger
            return $null
        }

        $info = Read-RepoLockInfo $path
        $tooOld = ($null -ne $info.Age) -and ($info.Age.TotalHours -ge $MaxHoldHours)

        if (-not (Test-RepoLockOwnerAlive $info.OwnerPid)) {
            Write-RepoLockLog ("Repo lock held by PID {0} ({1}), which is gone. Reclaiming." -f `
                $info.OwnerPid, $info.Holder) 'WARN' $Logger
            Remove-Item $path -Force -ErrorAction SilentlyContinue
            continue
        }

        if ($tooOld) {
            Write-RepoLockLog ("Repo lock held by {0} for {1:N1}h, beyond the {2}h ceiling. Reclaiming." -f `
                $info.Holder, $info.Age.TotalHours, $MaxHoldHours) 'WARN' $Logger
            Remove-Item $path -Force -ErrorAction SilentlyContinue
            continue
        }

        if ((Get-Date) -ge $deadline) {
            Write-RepoLockLog ("Gave up waiting for {0} (PID {1}) after {2} minutes." -f `
                $info.Holder, $info.OwnerPid, $MaxWaitMinutes) 'ERROR' $Logger
            return $null
        }

        if (-not $waited) {
            Write-RepoLockLog ("Repo is held by {0} (PID {1}). Waiting up to {2} min - running both loops at once corrupts git's index." -f `
                $info.Holder, $info.OwnerPid, $MaxWaitMinutes) 'WARN' $Logger
            $waited = $true
        }
        Start-Sleep -Seconds $PollSeconds
    }
}

<#
Release the lock, but only if this process still owns it. Checking ownership
matters: a reclaimed lock now belongs to someone else, and deleting theirs
would let a third process in alongside them.
#>
function Exit-RepoLock {
    param([string]$Path)
    if (-not $Path) { return }
    if (-not (Test-Path $Path)) { return }
    $info = Read-RepoLockInfo $Path
    if ($info.OwnerPid -ne $PID) { return }
    Remove-Item $Path -Force -ErrorAction SilentlyContinue
}
