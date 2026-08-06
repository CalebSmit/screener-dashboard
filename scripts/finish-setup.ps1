<#
.SYNOPSIS
    One-time setup walkthrough. Run this once, follow the prompts, done.

.DESCRIPTION
    Walks through the four things that need a human: signing in to GitHub,
    signing in to Claude Code (and trusting this folder), saving the setup to
    GitHub, and switching the two scheduled tasks on.

    Safe to run more than once - it skips anything already done.

    NOTE: keep this file ASCII-only and saved as UTF-8 with BOM.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Continue'
$RepoPath = Split-Path -Parent $PSScriptRoot

function Say {
    param([string]$Text, [string]$Colour = 'White')
    Write-Host $Text -ForegroundColor $Colour
}
function Header {
    param([string]$Text)
    Write-Host ""
    Write-Host "==============================================================" -ForegroundColor Cyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host "==============================================================" -ForegroundColor Cyan
}

$env:Path = "$([Environment]::GetEnvironmentVariable('Path','Machine'));$([Environment]::GetEnvironmentVariable('Path','User'))"
Set-Location $RepoPath

Header "Screener setup - 5 steps, about 5 minutes"
Say "Working folder: $RepoPath"

# ---------------------------------------------------------------- STEP 0 ----
Header "STEP 0 of 5 - Check git knows who you are"

$gitName  = (& git config --global user.name)  2>$null
$gitEmail = (& git config --global user.email) 2>$null

if ($gitName -and $gitEmail) {
    Say "  Git identity set: $gitName <$gitEmail>" Green
} else {
    Say "  Git does not know your name yet. Every commit needs one." Yellow
    Say ""
    $n = Read-Host "  Your name (press Enter for 'Caleb Smit')"
    if (-not $n) { $n = 'Caleb Smit' }
    $e = Read-Host "  Your email (press Enter for 'caleb.smit@icloud.com')"
    if (-not $e) { $e = 'caleb.smit@icloud.com' }

    & git config --global user.name  $n
    & git config --global user.email $e
    Say ""
    Say "  Set to: $n <$e>" Green
}

# ---------------------------------------------------------------- STEP 1 ----
Header "STEP 1 of 5 - Sign in to GitHub"

& gh auth status *>$null
if ($LASTEXITCODE -eq 0) {
    Say "  Already signed in to GitHub. Skipping." Green
} else {
    Say "  A sign-in wizard will now open in this window."
    Say ""
    Say "  Answer the questions like this:" Yellow
    Say "    - What account do you want to log into?  ->  GitHub.com"
    Say "    - Preferred protocol?                    ->  HTTPS"
    Say "    - Authenticate Git with your GitHub credentials?  ->  Yes"
    Say "    - How would you like to authenticate?    ->  Login with a web browser"
    Say ""
    Say "  It will show you a short code, then open your browser."
    Say "  Paste the code in the browser and approve it."
    Say ""
    Read-Host "  Press Enter when you are ready to start"

    & gh auth login

    & gh auth status *>$null
    if ($LASTEXITCODE -ne 0) {
        Say ""
        Say "  GitHub sign-in did not complete. Re-run this script to try again." Red
        exit 1
    }
    Say ""
    Say "  GitHub sign-in complete." Green
}

# ---------------------------------------------------------------- STEP 2 ----
Header "STEP 2 of 5 - Sign in to Claude Code and trust this folder"

Say "  Claude Code will now open in this window."
Say ""
Say "  Two things to do inside it:" Yellow
Say "    1. If it asks you to log in, type:  /login   and follow the prompts."
Say "    2. If it asks whether you trust this folder, choose YES."
Say ""
Say "  Trusting the folder is essential. Without it the morning runs"
Say "  ignore their permissions and fail every time." Yellow
Say ""
Say "  When you are done, type  /exit  to come back here."
Say ""
Read-Host "  Press Enter to open Claude Code"

& claude

Say ""
Say "  Back from Claude Code. Checking the unattended runs will work..."

# The scheduled runs use the command-line Claude, which records trust under a
# forward-slash path. The desktop app uses backslashes. If only the backslash
# entry is trusted, every 6 AM run fails on permissions - so verify explicitly
# rather than finding out tomorrow morning.
$trustOk = $true
$cfgPath = "$env:USERPROFILE\.claude.json"
if (Test-Path $cfgPath) {
    try {
        $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
        $fwdKey = $RepoPath.Replace('\', '/')
        $prop = $cfg.projects.PSObject.Properties | Where-Object { $_.Name -eq $fwdKey }
        if ($prop -and -not $prop.Value.hasTrustDialogAccepted) { $trustOk = $false }
    } catch { }
}

if (-not $trustOk) {
    Say ""
    Say "  PROBLEM: this folder is not trusted by the command-line Claude." Red
    Say "  The morning runs would fail every time." Red
    Say ""
    Say "  Fix it by running this one line, then re-run this setup script:" Yellow
    Say ""
    Say "    claude --add-dir `"$RepoPath`"" Cyan
    Say ""
    Say "  Or start 'claude' in this folder and answer YES to the trust question."
    Say ""
    $go = Read-Host "  Continue anyway? (y/N)"
    if ($go -ne 'y') { Say "  Stopped. Fix the trust issue and re-run." Yellow; exit 1 }
} else {
    Say "  Trust check passed." Green
}

# ---------------------------------------------------------------- STEP 3 ----
Header "STEP 3 of 5 - Save the setup to GitHub"

$changes = & git status --porcelain 2>&1 | ForEach-Object { [string]$_ }
if (-not ($changes -join '').Trim()) {
    Say "  Nothing new to save. Skipping." Green
} else {
    Say "  Saving these files to GitHub:"
    $changes | Where-Object { $_.Trim() } | ForEach-Object { Say "     $($_.Trim())" }
    Say ""

    & git add -A
    & git commit -m "setup: autonomous improvement routine (data loop + code loop)" | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Say "  Could not save the changes. Stopping so nothing is half-done." Red
        exit 1
    }

    & git push origin main
    if ($LASTEXITCODE -ne 0) {
        Say ""
        Say "  Saved on this computer, but could not upload to GitHub." Red
        Say "  Check your internet connection and re-run this script." Red
        exit 1
    }
    Say ""
    Say "  Saved and uploaded to GitHub." Green
}

# ---------------------------------------------------------------- STEP 4 ----
Header "STEP 4 of 5 - Switch the two morning routines on"

$tasks = @('Screener Data Run', 'Nightly Screener Improvement')
foreach ($t in $tasks) {
    $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    if (-not $task) { Say "  Could not find scheduled task '$t'." Red; continue }
    if ($task.State -eq 'Disabled') {
        Enable-ScheduledTask -TaskName $t | Out-Null
        Say "  Switched on: $t" Green
    } else {
        Say "  Already on: $t" Green
    }
}

# ---------------------------------------------------------------- STEP 5 ----
Header "STEP 5 of 5 - Prove an unattended run can actually work"

Say "  Running the same headless check the 6 AM job uses..."
Say ""

$probe = "Reply with exactly: HEADLESS_OK" | & claude --print --output-format text 2>&1 |
    ForEach-Object { [string]$_ }
$probeText = ($probe -join ' ')

if ($probeText -match 'HEADLESS_OK') {
    Say "  Unattended runs work." Green
} else {
    Say "  The unattended check did NOT succeed. It said:" Red
    ($probe | Select-Object -First 6) | ForEach-Object { Say "     $_" Red }
    Say ""
    Say "  The scheduled runs are switched on but will likely fail." Yellow
    Say "  Most common cause: this folder is not trusted by the command-line" Yellow
    Say "  Claude. Try:  claude --add-dir `"$RepoPath`"" Yellow
    Say ""
}

# ------------------------------------------------------------------- DONE ----
Header "All set"

foreach ($t in $tasks) {
    $task = Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue
    if ($task) {
        $info = Get-ScheduledTaskInfo -TaskName $t
        Say ("  {0,-30} {1,-8} next run: {2}" -f $t, $task.State, $info.NextRunTime)
    }
}

Say ""
Say "  What happens from now on:" Cyan
Say "    2:00 AM (Mon-Fri)  the screener runs and refreshes the live dashboard"
Say "    6:00 AM (Mon-Fri)  Claude spends the session improving the project"
Say ""
Say "  Leave this PC on or asleep overnight - not shut down."
Say "  Sleep is fine; it wakes itself up."
Say ""
Say "  To see what happened, open NIGHTLY_LOG.md in the project folder."
Say "  To stop everything, run:  scripts\stop-everything.ps1"
Say ""
