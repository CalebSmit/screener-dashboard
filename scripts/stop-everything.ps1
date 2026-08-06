<#
.SYNOPSIS
    Panic button. Switches both morning routines off. Nothing else changes.

.DESCRIPTION
    Run this if the project is doing something you don't like. It stops future
    runs immediately. Everything already done stays as it is - see ROLLBACK.md
    if you also want to undo recent changes.

    Turn things back on with scripts\finish-setup.ps1.

    NOTE: keep this file ASCII-only and saved as UTF-8 with BOM.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'

Write-Host ""
Write-Host "Stopping the automatic morning runs..." -ForegroundColor Yellow
Write-Host ""

foreach ($t in @('Screener Data Run', 'Nightly Screener Improvement')) {
    $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    if (-not $task) {
        Write-Host "  Not found: $t" -ForegroundColor Red
        continue
    }
    if ($task.State -eq 'Disabled') {
        Write-Host "  Already off: $t" -ForegroundColor Green
    } else {
        Disable-ScheduledTask -TaskName $t | Out-Null
        Write-Host "  Switched OFF: $t" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "Done. No further automatic runs will happen." -ForegroundColor Cyan
Write-Host "The live dashboard stays exactly as it is now."
Write-Host ""
Write-Host "To undo recent changes as well, see ROLLBACK.md"
Write-Host "To turn everything back on, run:  scripts\finish-setup.ps1"
Write-Host ""
