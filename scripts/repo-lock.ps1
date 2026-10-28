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
    crashed run must not jam tomorrow. Since 2026-10-27 a lock is published whole
    (Publish-RepoLockFile) and reclaimed only under a token after a re-read
    (Invoke-RepoLockReclaim); a lock that names no owner counts as held until
    it is $UnreadableGraceMinutes old. PID reuse is guarded by a hard age
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
Publish a lock file in one step. The contents are written to a private temp file
in the same directory and moved onto the target: File.Move fails if the target
exists and is atomic on one volume, so the lock carries its owner's PID from the
moment it exists. (2026-10-27: CreateNew-then-write left the file empty for
70-800 ms, and a contender reading it then took the live owner for a dead one -
research/2026-10-27-flaky-tests.md.) Returns $true if this process now owns the
target, $false if someone else does. Throws if the temp file cannot be written.
#>
function Publish-RepoLockFile {
    param([Parameter(Mandatory)][string]$Path, [Parameter(Mandatory)][string]$Text)
    $tmp = "{0}.{1}.{2}.tmp" -f $Path, $PID, ([guid]::NewGuid().ToString('N'))
    [System.IO.File]::WriteAllText($tmp, $Text, [System.Text.Encoding]::ASCII)
    try {
        [System.IO.File]::Move($tmp, $Path)
        return $true
    } catch {
        return $false
    } finally {
        if (Test-Path -LiteralPath $tmp) { Remove-Item -LiteralPath $tmp -Force -ErrorAction SilentlyContinue }
    }
}

<#
Remove a lock judged dead or over-age, without ever removing a live one.
Only the holder of a short-lived reclaim token may delete the lock, and it
re-reads the lock under the token first: a lock whose owner is dead cannot
change except by deletion, so the re-read is still true when it deletes. Two
racers that both read the same dead PID used to delete each other's fresh locks
(2-5 holders in 6 of 16 six-racer trials, 2026-10-27). Returns nothing; the
caller loops and tries to acquire again.
#>
function Invoke-RepoLockReclaim {
    param([Parameter(Mandatory)][string]$Path, [int]$JudgedPid, [string]$Why, [int]$MaxHoldHours = 6,
          [int]$UnreadableGraceMinutes = 10, [scriptblock]$Logger)
    $token = "$Path.reclaim"
    $text  = "{0}`nreclaim`n{1}`n" -f $PID, (Get-Date -Format 'o')
    if (-not (Publish-RepoLockFile $token $text)) {
        # Someone else is reclaiming. A token is held for milliseconds, so one
        # older than two minutes was left by a process that died holding it.
        try {
            $age = (Get-Date) - (Get-Item -LiteralPath $token -ErrorAction Stop).LastWriteTime
            if ($age.TotalMinutes -ge 2) { Remove-Item -LiteralPath $token -Force -ErrorAction SilentlyContinue }
        } catch { }
        Start-Sleep -Milliseconds 200
        return
    }
    try {
        if (-not (Test-Path -LiteralPath $Path)) { return }
        $now = Read-RepoLockInfo $Path
        $stillStale = $false
        if ($now.OwnerPid -eq $JudgedPid) {
            if ($now.OwnerPid -le 0) {
                $stillStale = ($null -ne $now.Age) -and ($now.Age.TotalMinutes -ge $UnreadableGraceMinutes)
            } elseif (-not (Test-RepoLockOwnerAlive $now.OwnerPid)) {
                $stillStale = $true
            } else {
                $stillStale = ($null -ne $now.Age) -and ($now.Age.TotalHours -ge $MaxHoldHours)
            }
        }
        if ($stillStale) {
            Write-RepoLockLog $Why 'WARN' $Logger
            Remove-Item -LiteralPath $Path -Force -ErrorAction SilentlyContinue
        }
    } finally {
        Remove-Item -LiteralPath $token -Force -ErrorAction SilentlyContinue
    }
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
        [int]$UnreadableGraceMinutes = 10,
        [scriptblock]$Logger
    )

    $path     = Get-RepoLockPath $LogDir
    $deadline = (Get-Date).AddMinutes($MaxWaitMinutes)
    $waited   = $false

    $failedCreates = 0
    while ($true) {
        $text = "{0}`n{1}`n{2}`n" -f $PID, $Holder, (Get-Date -Format 'o')
        try {
            $got = Publish-RepoLockFile $path $text
        } catch {
            # The temp file could not be written - a missing directory, or
            # permissions. Looping would spin forever.
            Write-RepoLockLog "Cannot create the repo lock at $path." 'ERROR' $Logger
            return $null
        }
        if ($got) {
            if ($waited) { Write-RepoLockLog "Repo lock acquired." 'INFO' $Logger }
            return $path
        }

        if (-not (Test-Path -LiteralPath $path)) {
            # The holder released between our move and this check. Try again at
            # once, but never spin: a move that keeps failing onto an absent
            # path is a fault, not contention.
            $failedCreates++
            if ($failedCreates -ge 20) {
                Write-RepoLockLog "Cannot create the repo lock at $path." 'ERROR' $Logger
                return $null
            }
            continue
        }
        $failedCreates = 0

        $info = Read-RepoLockInfo $path
        $tooOld = ($null -ne $info.Age) -and ($info.Age.TotalHours -ge $MaxHoldHours)

        if ($info.OwnerPid -le 0) {
            # Unreadable, or no PID. Since locks are published whole this is not
            # a lock mid-write; but it is not proof of a dead owner either, so it
            # counts as held until it is older than the grace period.
            $unreadableOld = ($null -ne $info.Age) -and ($info.Age.TotalMinutes -ge $UnreadableGraceMinutes)
            if ($unreadableOld) {
                Invoke-RepoLockReclaim -Path $path -JudgedPid $info.OwnerPid -MaxHoldHours $MaxHoldHours `
                    -UnreadableGraceMinutes $UnreadableGraceMinutes -Logger $Logger `
                    -Why ("Repo lock at {0} names no owner and is {1:N0} min old. Reclaiming." -f $path, $info.Age.TotalMinutes)
                continue
            }
        } elseif (-not (Test-RepoLockOwnerAlive $info.OwnerPid)) {
            Invoke-RepoLockReclaim -Path $path -JudgedPid $info.OwnerPid -MaxHoldHours $MaxHoldHours `
                -UnreadableGraceMinutes $UnreadableGraceMinutes -Logger $Logger `
                -Why ("Repo lock held by PID {0} ({1}), which is gone. Reclaiming." -f $info.OwnerPid, $info.Holder)
            continue
        } elseif ($tooOld) {
            Invoke-RepoLockReclaim -Path $path -JudgedPid $info.OwnerPid -MaxHoldHours $MaxHoldHours `
                -UnreadableGraceMinutes $UnreadableGraceMinutes -Logger $Logger `
                -Why ("Repo lock held by {0} for {1:N1}h, beyond the {2}h ceiling. Reclaiming." -f `
                    $info.Holder, $info.Age.TotalHours, $MaxHoldHours)
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
