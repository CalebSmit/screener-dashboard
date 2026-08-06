<#
.SYNOPSIS
    Mark this project folder as trusted for the command-line Claude Code.

.DESCRIPTION
    Claude Code records folder trust in %USERPROFILE%\.claude.json, keyed by
    folder path. The desktop app writes the path with backslashes; the
    command-line version used by the scheduled runs looks it up with forward
    slashes. Accepting the trust prompt in one does not satisfy the other.

    If only the backslash entry is trusted, every scheduled run starts with
    "this workspace has not been trusted", ignores its permission settings, and
    fails.

    This script sets the forward-slash entry to trusted, so the unattended runs
    work. It changes exactly one true/false value and backs the file up first.

    It does not touch credentials, tokens, or any other setting.

    NOTE: keep this file ASCII-only and saved as UTF-8 with BOM.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RepoPath = Split-Path -Parent $PSScriptRoot
$FwdKey   = $RepoPath.Replace('\', '/')
$Cfg      = Join-Path $env:USERPROFILE '.claude.json'

Write-Host ""
Write-Host "Trusting this folder for unattended Claude Code runs" -ForegroundColor Cyan
Write-Host "  folder: $RepoPath"
Write-Host "  key   : $FwdKey"
Write-Host ""

if (-not (Test-Path $Cfg)) {
    Write-Host "  Could not find $Cfg" -ForegroundColor Red
    Write-Host "  Start 'claude' once, then run this again." -ForegroundColor Red
    exit 1
}

# --- Back up first ----------------------------------------------------------
$stamp  = Get-Date -Format 'yyyyMMdd-HHmmss'
$backup = "$Cfg.bak-$stamp"
Copy-Item $Cfg $backup -Force
$before = (Get-Item $Cfg).Length
Write-Host "  Backup saved: $backup" -ForegroundColor Green

try {
    $json = Get-Content $Cfg -Raw | ConvertFrom-Json

    if (-not $json.projects) {
        Write-Host "  No 'projects' section found - nothing to do." -ForegroundColor Red
        exit 1
    }

    $existing = $json.projects.PSObject.Properties | Where-Object { $_.Name -eq $FwdKey }

    if ($existing) {
        if ($existing.Value.hasTrustDialogAccepted -eq $true) {
            Write-Host "  Already trusted. Nothing to change." -ForegroundColor Green
            Remove-Item $backup -Force
            exit 0
        }
        $existing.Value.hasTrustDialogAccepted = $true
        Write-Host "  Updated existing entry to trusted." -ForegroundColor Green
    } else {
        # Clone the backslash entry if there is one, so we keep its settings.
        $bsKey  = $RepoPath
        $source = $json.projects.PSObject.Properties | Where-Object { $_.Name -eq $bsKey }
        if ($source) {
            $clone = $source.Value | ConvertTo-Json -Depth 100 | ConvertFrom-Json
            $clone.hasTrustDialogAccepted = $true
        } else {
            $clone = [pscustomobject]@{ hasTrustDialogAccepted = $true }
        }
        $json.projects | Add-Member -NotePropertyName $FwdKey -NotePropertyValue $clone
        Write-Host "  Added a trusted entry." -ForegroundColor Green
    }

    # Depth matters: the default of 2 would silently mangle this file.
    $json | ConvertTo-Json -Depth 100 | Set-Content $Cfg -Encoding utf8

    $after = (Get-Item $Cfg).Length
    Write-Host "  File size: $before -> $after bytes"

    # --- Verify -------------------------------------------------------------
    $check = Get-Content $Cfg -Raw | ConvertFrom-Json
    $ok = ($check.projects.PSObject.Properties |
           Where-Object { $_.Name -eq $FwdKey }).Value.hasTrustDialogAccepted

    Write-Host ""
    if ($ok -eq $true) {
        Write-Host "  Verified: folder is now trusted." -ForegroundColor Green
        Write-Host ""
        Write-Host "  Next: re-run scripts\finish-setup.ps1 to confirm an" -ForegroundColor Cyan
        Write-Host "  unattended run works end to end." -ForegroundColor Cyan
    } else {
        Write-Host "  Verification failed. Restoring your original file." -ForegroundColor Red
        Copy-Item $backup $Cfg -Force
        exit 1
    }
}
catch {
    Write-Host ""
    Write-Host "  Something went wrong: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "  Restoring your original file from the backup." -ForegroundColor Yellow
    Copy-Item $backup $Cfg -Force
    Write-Host "  Restored. Nothing was changed." -ForegroundColor Green
    exit 1
}

Write-Host ""
