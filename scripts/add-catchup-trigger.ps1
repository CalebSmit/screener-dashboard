<#
.SYNOPSIS
    Add a catch-up "at logon" trigger to both scheduled loops.

.DESCRIPTION
    The two tasks run with LogonType InteractiveToken - i.e. only while a user
    is logged on. On 2026-08-12 the machine rebooted at 00:47 for a Windows
    update and sat at the login screen; nobody was logged in at 02:00 or 06:00,
    so both runs were simply lost with no error anywhere.

    This adds a second trigger to each task: at logon, delayed 3 minutes. If a
    scheduled run was missed, logging in picks it up.

    That is safe because both scripts now write a once-per-day success marker
    (logs/.datarun-last-success, logs/.nightly-last-success) and exit
    immediately if today's run already succeeded. So logging in five times does
    not run the loop five times - and a run that *failed* leaves no marker, so
    it is correctly retried.

    Idempotent: re-running just rewrites the same two triggers.

    NOTE: keep this file ASCII-only and saved as UTF-8 with BOM.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\add-catchup-trigger.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$Repo = Split-Path -Parent $PSScriptRoot

$specs = @(
    @{ Name = 'Screener Data Run';            Script = 'data-run.ps1';         At = '2:00AM' },
    @{ Name = 'Nightly Screener Improvement'; Script = 'nightly-screener.ps1'; At = '6:00AM' }
)

Write-Host ""
Write-Host "Adding catch-up logon triggers" -ForegroundColor Cyan
Write-Host "  repo: $Repo"
Write-Host ""

foreach ($s in $specs) {
    if (-not (Get-ScheduledTask -TaskName $s.Name -ErrorAction SilentlyContinue)) {
        Write-Host "  SKIP (task not found): $($s.Name)" -ForegroundColor Red
        continue
    }

    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$Repo\scripts\$($s.Script)`"" `
        -WorkingDirectory $Repo

    $daily = New-ScheduledTaskTrigger -Weekly `
        -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $s.At
    $logon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
    $logon.Delay = 'PT3M'

    Set-ScheduledTask -TaskName $s.Name -Action $action -Trigger @($daily, $logon) | Out-Null
    Write-Host "  updated: $($s.Name)" -ForegroundColor Green
}

Write-Host ""
foreach ($s in $specs) {
    $t = Get-ScheduledTask -TaskName $s.Name -ErrorAction SilentlyContinue
    if ($t) {
        $i = Get-ScheduledTaskInfo -TaskName $s.Name
        Write-Host ("  {0,-30} {1,-8} triggers={2}  next={3}" -f `
            $s.Name, $t.State, $t.Triggers.Count, $i.NextRunTime)
    }
}
Write-Host ""
Write-Host "Done. A missed run will now be picked up 3 minutes after you log in." -ForegroundColor Cyan
Write-Host ""
