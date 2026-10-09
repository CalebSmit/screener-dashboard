<#
.SYNOPSIS
    Create (or repair) the two scheduled loops. The definitions live here, in
    version control, not only in Task Scheduler on one machine.

.DESCRIPTION
    Open since 2026-08-10 and called the single highest-value infrastructure
    item by the 2026-08-21 retrospective: `grep -rn "Register-ScheduledTask" .`
    found nothing, so the 02:00 and 06:00 triggers existed solely as hand-made
    Task Scheduler entries. If that machine were rebuilt, the routine would be
    gone with no record of how it had been configured.

    Creates three tasks:
      Screener Data Run             02:00 Mon-Fri  -> scripts/data-run.ps1
      Nightly Screener Improvement  06:00 Mon-Fri  -> scripts/nightly-screener.ps1
      Screener Option Quotes        20:00 Mon-Fri  -> scripts/refresh-option-quotes.ps1

    The two loops also get an at-logon trigger so a run missed while the machine
    sat at the login screen is picked up when the owner next signs in. That is
    safe because both scripts write a once-per-day success marker and exit early
    if the day already succeeded; a failed run leaves no marker and is correctly
    retried.

    The option-quote refresh (added 2026-10-09) deliberately has neither a logon
    trigger nor StartWhenAvailable: it is only useful between the close and
    midnight ET, because the source serves the option chain with every bid, ask
    and implied volatility at zero overnight - which is why the 02:00 loop got a
    usable reading for 0 of 503 stocks. Running it late would do nothing but
    write a log line. It touches no tracked file and takes no repo lock, so it
    cannot collide with either loop.

    The logon delays are staggered - data 3 min, code 20 min. They were both
    3 min until 2026-08-29, when the two loops started in the same second and
    fought over .git/index.lock; the data run exited before running the
    screener. scripts/repo-lock.ps1 makes a collision safe; the stagger makes
    the order deterministic.

    Idempotent: re-running replaces the definitions with these.

.PARAMETER NoLogonCatchup
    Skip the at-logon triggers, leaving only the fixed schedule.

.PARAMETER Disabled
    Register the tasks but leave them switched off.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\register-tasks.ps1

.NOTES
    LogonType is InteractiveToken deliberately: the alternative that survives a
    logged-out machine requires storing the account password, which this project
    will not do. The consequence - runs are lost while nobody is logged on - is
    why the at-logon catch-up exists. Three of the first four outages had this
    cause. Turning on "Use my sign-in info to automatically finish setting up
    after an update or restart" (Settings > Accounts > Sign-in options) is the
    complementary fix and lives outside this repo.

    Keep this file ASCII-only and saved as UTF-8 with BOM.
#>
[CmdletBinding()]
param(
    [switch]$NoLogonCatchup,
    [switch]$Disabled
)

$ErrorActionPreference = 'Stop'

$Repo = Split-Path -Parent $PSScriptRoot

$Specs = @(
    @{
        Name        = 'Screener Data Run'
        Script      = 'data-run.ps1'
        At          = '2:00AM'
        LogonDelay  = 'PT3M'
        Limit       = (New-TimeSpan -Hours 3)
        Description = 'Runs the screener live, health-gates the result, refreshes the dashboard and feeds the improvement engine. Pure Python and git - uses no Claude quota.'
    },
    @{
        Name        = 'Nightly Screener Improvement'
        Script      = 'nightly-screener.ps1'
        At          = '6:00AM'
        # Deliberately later than the data run's PT3M. Both used to be PT3M, so
        # at logon they started in the same second and fought over
        # .git/index.lock; on 2026-08-29 that killed the data run outright.
        # scripts/repo-lock.ps1 is the actual fix - it makes a collision safe -
        # and this makes the order deterministic: evidence first, then the
        # session that reads it. 20 minutes clears a data run, measured at
        # 11.8-13.6 min over 2026-08-21..28.
        LogonDelay  = 'PT20M'
        Limit       = (New-TimeSpan -Hours 4)
        Description = 'Autonomous Claude Code session. Re-verifies all four ship gates itself, merges to main on success and tags good/<date>. Never merges on a failing gate.'
    }
    @{
        Name        = 'Screener Option Quotes'
        Script      = 'refresh-option-quotes.ps1'
        At          = '8:00PM'
        # No logon trigger and no StartWhenAvailable, unlike the two loops above. This task is
        # only useful inside a window: measured 2026-10-09, the option chain carries quotes at
        # 21:27 ET and comes back with every bid, ask and implied volatility at zero by 03:00 ET,
        # so the 02:00 data loop got a usable reading for 0 of 503 stocks. A catch-up firing at
        # an arbitrary hour would correctly do nothing, which is just noise in the log - and a
        # missed evening is already covered, because options_cache serves a reading for up to
        # three days and so carries a Friday close to Monday.
        NoLogon     = $true
        NoCatchUp   = $true
        Limit       = (New-TimeSpan -Minutes 30)
        Description = 'Collects option quotes after the close into data/options/quotes.json, which the 02:00 run reads because the source serves no quotes at that hour. Touches no tracked file, runs no git command, takes no repo lock, and can never change a score.'
    }
)

Write-Host ""
Write-Host "Registering the Screener scheduled loops" -ForegroundColor Cyan
Write-Host "  repo: $Repo"
Write-Host ""

foreach ($s in $Specs) {
    $scriptPath = Join-Path $Repo "scripts\$($s.Script)"
    if (-not (Test-Path $scriptPath)) {
        Write-Host "  SKIP - not found: $scriptPath" -ForegroundColor Red
        continue
    }

    $action = New-ScheduledTaskAction -Execute 'powershell.exe' `
        -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`"" `
        -WorkingDirectory $Repo

    $triggers = @(
        New-ScheduledTaskTrigger -Weekly `
            -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $s.At
    )
    if (-not $NoLogonCatchup -and -not $s.NoLogon) {
        $logon = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
        $logon.Delay = $s.LogonDelay
        $triggers += $logon
    }

    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited

    # StartWhenAvailable + WakeToRun let a sleeping machine still run; neither
    # helps when nobody is logged on, which is what the logon trigger covers.
    # A task marked NoCatchUp opts out: it is only useful inside a time window,
    # so running it late is worse than not running it.
    if ($s.NoCatchUp) {
        $settings = New-ScheduledTaskSettingsSet -WakeToRun `
            -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
            -ExecutionTimeLimit $s.Limit -MultipleInstances IgnoreNew
    } else {
        $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -WakeToRun `
            -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries `
            -ExecutionTimeLimit $s.Limit -MultipleInstances IgnoreNew
    }

    if (Get-ScheduledTask -TaskName $s.Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $s.Name -Confirm:$false
    }

    Register-ScheduledTask -TaskName $s.Name -Action $action -Trigger $triggers `
        -Principal $principal -Settings $settings -Description $s.Description | Out-Null

    if ($Disabled) { Disable-ScheduledTask -TaskName $s.Name | Out-Null }

    Write-Host "  registered: $($s.Name)" -ForegroundColor Green
}

Write-Host ""
foreach ($s in $Specs) {
    $t = Get-ScheduledTask -TaskName $s.Name -ErrorAction SilentlyContinue
    if ($t) {
        $i = Get-ScheduledTaskInfo -TaskName $s.Name
        Write-Host ("  {0,-30} {1,-8} triggers={2}  next={3}" -f `
            $s.Name, $t.State, $t.Triggers.Count, $i.NextRunTime)
    }
}
Write-Host ""
Write-Host "Leave the PC on or asleep overnight - not shut down - and stay logged in." -ForegroundColor Cyan
Write-Host ""
