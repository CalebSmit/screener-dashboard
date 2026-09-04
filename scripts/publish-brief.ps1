<#
.SYNOPSIS
    Publish MORNING_BRIEF.md to origin/main without ever pushing local HEAD.

.DESCRIPTION
    Dot-sourced by data-run.ps1 and nightly-screener.ps1. Both call it from a
    `finally` block, because the brief is the watchdog heartbeat
    (.github/workflows/loop-watchdog.yml) and must land whether the run
    succeeded or failed.

    WHY THIS EXISTS - 2026-09-04 retrospective.

    Both runners used to publish the brief with:

        git push origin HEAD:main

    On the happy path HEAD is main, so that was correct and it is what ran on
    every successful day. On a *ship-gate failure* it is not. Then HEAD is the
    nightly branch carrying the work the gates just refused, and
    `push HEAD:main` fast-forwards origin/main onto every commit on it -
    publishing exactly that work to the branch GitHub Pages serves to the
    public, from a `finally` block, after the gates said no.

    Reproduced in a real origin+clone sandbox before this file was written; the
    reproduction is tests/test_brief_publish_safety.py::test_pushing_head_to_main_
    publishes_the_whole_branch, which drives the old command and asserts the
    broken file arrives on origin/main.

    It had not fired yet only by luck of ordering. The one gate failure in this
    project's history (2026-09-02) happened while the session had already
    merged onto local main, so the recovery path reset local main to the base
    commit first and the subsequent HEAD:main push was rejected as a
    non-fast-forward - the log reads "Brief committed locally; push failed."
    prompts/nightly.md also tells a session that fails its own gates to leave
    the work on the branch, which is the shape that does publish.

    THE REPLACEMENT builds the brief commit on top of origin/main with git
    plumbing. It never checks out, never merges local work, and never pushes a
    local ref, so the only file it can possibly change on main is
    MORNING_BRIEF.md - by construction, not by convention.

    NOTE: keep this file ASCII-only and saved as UTF-8 with BOM.
#>

$BriefFile = 'MORNING_BRIEF.md'

function Write-BriefLog {
    param([string]$Message, [string]$Level = 'INFO', [scriptblock]$Logger)
    if ($Logger) { & $Logger $Message $Level }
    else { Write-Host ("[{0}] {1}" -f $Level, $Message) }
}

function Invoke-BriefGit {
    param([Parameter(Mandatory)][string]$RepoPath, [string[]]$GitArgs)
    $out = & git -C $RepoPath @GitArgs 2>&1
    return [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Text     = (($out | ForEach-Object { $_.ToString() }) -join "`n").Trim()
    }
}

<#
Publish the working tree's MORNING_BRIEF.md as a single-file commit on top of
origin/main.

Returns $true if a brief reached origin/main (or was already identical there),
$false otherwise. Never throws, never fatal: a missing heartbeat is a watchdog
alarm, but a crashed `finally` would also lose the repo lock release.
#>
function Publish-BriefToMain {
    param(
        [Parameter(Mandatory)][string]$RepoPath,
        [Parameter(Mandatory)][string]$Label,
        [int]$Attempts = 3,
        [scriptblock]$Logger
    )

    $briefPath = Join-Path $RepoPath $BriefFile
    if (-not (Test-Path $briefPath)) {
        Write-BriefLog "No $BriefFile to publish." 'WARN' $Logger
        return $false
    }

    for ($attempt = 1; $attempt -le $Attempts; $attempt++) {
        # Always re-read origin/main. On a retry it has moved, which is the
        # whole reason the previous push was rejected.
        $fetch = Invoke-BriefGit $RepoPath @('fetch', '--quiet', 'origin', 'main')
        if ($fetch.ExitCode -ne 0) {
            Write-BriefLog "Could not fetch origin/main to publish the brief: $($fetch.Text)" 'WARN' $Logger
            return $false
        }

        $base = (Invoke-BriefGit $RepoPath @('rev-parse', 'FETCH_HEAD')).Text
        if (-not $base) {
            Write-BriefLog "Could not resolve origin/main." 'WARN' $Logger
            return $false
        }

        $blobResult = Invoke-BriefGit $RepoPath @('hash-object', '-w', '--', $briefPath)
        if ($blobResult.ExitCode -ne 0 -or -not $blobResult.Text) {
            Write-BriefLog "Could not hash $BriefFile : $($blobResult.Text)" 'WARN' $Logger
            return $false
        }
        $blob = $blobResult.Text

        # Build the new tree in a throwaway index. The repository's real index
        # is never touched, so this cannot disturb a run that is mid-flight or
        # leave anything staged behind us.
        $tmpIndex = Join-Path ([System.IO.Path]::GetTempPath()) ("screener-brief-{0}-{1}.index" -f $PID, $attempt)
        $tree = ''
        $previousIndex = $env:GIT_INDEX_FILE
        try {
            $env:GIT_INDEX_FILE = $tmpIndex
            $read = Invoke-BriefGit $RepoPath @('read-tree', $base)
            if ($read.ExitCode -ne 0) {
                Write-BriefLog "Could not read origin/main's tree: $($read.Text)" 'WARN' $Logger
                return $false
            }
            $upd = Invoke-BriefGit $RepoPath @('update-index', '--add', '--cacheinfo', "100644,$blob,$BriefFile")
            if ($upd.ExitCode -ne 0) {
                Write-BriefLog "Could not stage the brief: $($upd.Text)" 'WARN' $Logger
                return $false
            }
            $treeResult = Invoke-BriefGit $RepoPath @('write-tree')
            if ($treeResult.ExitCode -ne 0) {
                Write-BriefLog "Could not write the brief tree: $($treeResult.Text)" 'WARN' $Logger
                return $false
            }
            $tree = $treeResult.Text
        } finally {
            if ($null -ne $previousIndex) { $env:GIT_INDEX_FILE = $previousIndex }
            else { Remove-Item Env:GIT_INDEX_FILE -ErrorAction SilentlyContinue }
            Remove-Item $tmpIndex -Force -ErrorAction SilentlyContinue
        }

        $baseTree = (Invoke-BriefGit $RepoPath @('rev-parse', "$base^{tree}")).Text
        if ($tree -eq $baseTree) {
            Write-BriefLog "Brief unchanged on main; nothing to publish." 'INFO' $Logger
            Sync-BriefWorkingTree $RepoPath $base $Logger
            return $true
        }

        $commitResult = Invoke-BriefGit $RepoPath @('commit-tree', $tree, '-p', $base, '-m', "brief: $Label")
        if ($commitResult.ExitCode -ne 0 -or -not $commitResult.Text) {
            Write-BriefLog "Could not create the brief commit: $($commitResult.Text)" 'WARN' $Logger
            return $false
        }
        $commit = $commitResult.Text

        # Push the commit object directly. No local ref is named on either side
        # of the refspec, so nothing on this machine can ride along.
        $push = Invoke-BriefGit $RepoPath @('push', 'origin', "${commit}:refs/heads/main")
        if ($push.ExitCode -eq 0) {
            Write-BriefLog "Morning brief published." 'INFO' $Logger
            Sync-BriefWorkingTree $RepoPath $commit $Logger
            return $true
        }

        if ($attempt -lt $Attempts) {
            Write-BriefLog "Brief push rejected (origin/main moved); retrying." 'WARN' $Logger
            Start-Sleep -Seconds 3
        } else {
            Write-BriefLog "Brief not published after $Attempts attempts: $($push.Text)" 'WARN' $Logger
        }
    }

    return $false
}

<#
Leave the working tree clean.

The brief was published without committing locally, so MORNING_BRIEF.md is
still modified here. The next run starts with `git checkout main` and
`git pull`, and a dirty tracked file that the pull also changes aborts both -
losing the following day to tidiness. If local main is exactly the commit we
built on, fast-forward it; otherwise just restore the file from HEAD. Either
way the brief itself is safe on origin.
#>
function Sync-BriefWorkingTree {
    param([string]$RepoPath, [string]$Target, [scriptblock]$Logger)

    # Restore first, then fast-forward. A merge refuses to overwrite a locally
    # modified tracked file, and MORNING_BRIEF.md is exactly that here - so
    # trying the merge first fails on the one path where it matters.
    $restore = Invoke-BriefGit $RepoPath @('checkout', '--', $BriefFile)
    if ($restore.ExitCode -ne 0) {
        Write-BriefLog "Left $BriefFile modified in the working tree: $($restore.Text)" 'WARN' $Logger
        return
    }

    $head = (Invoke-BriefGit $RepoPath @('rev-parse', '--abbrev-ref', 'HEAD')).Text
    if ($head -ne 'main') { return }

    $ff = Invoke-BriefGit $RepoPath @('merge', '--ff-only', $Target)
    if ($ff.ExitCode -ne 0) {
        Write-BriefLog "Could not fast-forward local main to the published brief: $($ff.Text)" 'WARN' $Logger
    }
}
