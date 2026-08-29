<#
.SYNOPSIS
    Superseded by register-tasks.ps1. Delegates to it.

.DESCRIPTION
    This script used to write the two tasks' triggers itself, giving both an
    at-logon trigger delayed 'PT3M'. register-tasks.ps1 (shipped 2026-08-21)
    does everything this did and more, idempotently, and is now the single
    definition of both tasks.

    Two scripts writing the same triggers is not merely redundant, it is a
    live hazard: on 2026-08-29 the identical PT3M delays made both loops start
    in the same second and fight over .git/index.lock, and the data run died
    before it ran the screener. register-tasks.ps1 now staggers them (data
    PT3M, code PT20M). Had this file kept its own copy, running it once would
    have quietly put the collision back.

    Kept as a delegating shim rather than deleted, because the filename appears
    in NIGHTLY_LOG.md and in the owner's setup notes.

    NOTE: keep this file ASCII-only and saved as UTF-8 with BOM.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\add-catchup-trigger.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

Write-Host ""
Write-Host "add-catchup-trigger.ps1 is superseded by register-tasks.ps1." -ForegroundColor Yellow
Write-Host "Running that instead - it registers both tasks, with the catch-up"
Write-Host "logon triggers, and staggers them so the two loops cannot collide."
Write-Host ""

& (Join-Path $PSScriptRoot 'register-tasks.ps1')
