<#
.SYNOPSIS
    Undo a bad merge that already reached origin/main, by reverting - never
    by rewriting history.

.DESCRIPTION
    Found 2026-09-02. The nightly loop's failure-recovery path assumed IT was
    the thing pushing to main, so on a gate failure it reset the LOCAL copy
    of main and logged "main is untouched." But prompts/nightly.md has the
    Claude Code session merge and push main itself as its own final step, on
    the happy path, before the wrapper's independent re-check ever runs. On
    2026-09-02 the session's own gate check passed and it pushed a merge to
    origin/main at 06:24:22; the wrapper's fresh re-run of the suite at
    06:26:53 hit a flaky `test_parquet_roundtrip` failure and "reset" a local
    main that origin had already moved past. Resetting a ref nobody is
    tracking accomplishes nothing - the bad commit stayed live on the branch
    GitHub Pages serves, and the log's own claim was wrong.

    This script is the actual fix: if origin/main is still sitting on
    $BaseSha, there is nothing to do (the common case - the session's push
    never landed, so the existing local-reset path was already sufficient).
    If origin/main has moved past it, revert whatever it gained back to
    $BaseSha's tree, verify the reverted tree is byte-identical to $BaseSha
    (not just "close"), and only then push. `git revert` adds new commits;
    it never rewrites what is already public (CLAUDE.md rule 2).

    Deliberately conservative: if the revert cannot be made to produce a tree
    matching $BaseSha exactly - a conflict, a merge revert needing manual
    resolution, anything unexpected - this refuses to push and exits non-zero
    rather than guess. A human then has exactly what a diff between
    $BaseSha and origin/main shows them; that is a smaller, clearer problem
    than the one this script would create by pushing something unverified to
    the same branch that failed verification in the first place.

.PARAMETER BaseSha
    The last known-good commit - what local main was (or would be) reset to
    on gate failure.

.PARAMETER RepoPath
    Path to the git working tree. Operations happen on a detached, temporary
    ref so the caller's own checked-out branch is left alone.

.OUTPUTS
    Exit 0: origin/main already matched $BaseSha (nothing to revert) OR the
            revert was verified and pushed.
    Exit 1: origin/main had diverged and the revert could not be verified as
            exact. Nothing was pushed. $RepoPath's original branch/HEAD is
            restored either way.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$BaseSha,
    [Parameter(Mandatory)][string]$RepoPath
)

$ErrorActionPreference = 'Continue'

function Invoke-Git {
    param([string[]]$Arguments)
    $out = & git -C $RepoPath @Arguments 2>&1
    [pscustomobject]@{
        ExitCode = $LASTEXITCODE
        Text     = ($out | ForEach-Object { if ($null -ne $_) { $_.ToString() } }) -join "`n"
    }
}

$origBranch = (Invoke-Git @('rev-parse', '--abbrev-ref', 'HEAD')).Text.Trim()

try {
    $fetch = Invoke-Git @('fetch', 'origin', 'main')
    if ($fetch.ExitCode -ne 0) {
        Write-Output "REVERT-BAD-MERGE: could not fetch origin/main: $($fetch.Text)"
        exit 1
    }

    $originHead = (Invoke-Git @('rev-parse', 'origin/main')).Text.Trim()
    if ($originHead -eq $BaseSha) {
        Write-Output "REVERT-BAD-MERGE: origin/main is already at $($BaseSha.Substring(0,8)) - nothing to revert."
        exit 0
    }

    Write-Output "REVERT-BAD-MERGE: origin/main ($($originHead.Substring(0,8))) is ahead of the last known-good $($BaseSha.Substring(0,8)) - the session's own push landed before the failure was caught. Reverting."

    $tempRef = "revert-bad-merge-$([DateTime]::Now.ToString('yyyyMMddHHmmss'))"
    $co = Invoke-Git @('checkout', '-B', $tempRef, 'origin/main')
    if ($co.ExitCode -ne 0) {
        Write-Output "REVERT-BAD-MERGE: could not check out origin/main: $($co.Text)"
        exit 1
    }

    $rv = Invoke-Git @('revert', '--no-edit', "$BaseSha..HEAD")
    if ($rv.ExitCode -ne 0) {
        Invoke-Git @('revert', '--abort') | Out-Null
        # A merge commit in the range needs -m explicitly; retried alone
        # because -m only makes sense applied to the merge commit itself,
        # not a whole range that may also contain ordinary commits.
        $rv = Invoke-Git @('revert', '--no-edit', '-m', '1', 'HEAD')
    }
    if ($rv.ExitCode -ne 0) {
        Write-Output "REVERT-BAD-MERGE: revert failed and needs a human: $($rv.Text)"
        Invoke-Git @('revert', '--abort') | Out-Null
        exit 1
    }

    # The one check this whole script exists to make: the reverted tree must
    # be byte-identical to the last known-good state, not just "no conflict
    # markers". A clean revert that still leaves the tree different from
    # $BaseSha is exactly the unverified-guess this script must refuse.
    $diff = Invoke-Git @('diff', '--quiet', $BaseSha, 'HEAD')
    if ($diff.ExitCode -ne 0) {
        Write-Output "REVERT-BAD-MERGE: reverted tree does not match $($BaseSha.Substring(0,8)) exactly - refusing to push. A human needs to look at this directly."
        exit 1
    }

    $push = Invoke-Git @('push', 'origin', "${tempRef}:main")
    if ($push.ExitCode -ne 0) {
        Write-Output "REVERT-BAD-MERGE: revert verified locally but push to origin/main failed: $($push.Text)"
        exit 1
    }

    Write-Output "REVERT-BAD-MERGE: pushed a verified revert to origin/main. Public site is back to $($BaseSha.Substring(0,8))."
    exit 0
}
finally {
    if ($origBranch -and $origBranch -ne 'HEAD') {
        Invoke-Git @('checkout', $origBranch) | Out-Null
    }
    if ($tempRef) {
        Invoke-Git @('branch', '-D', $tempRef) | Out-Null
    }
}
