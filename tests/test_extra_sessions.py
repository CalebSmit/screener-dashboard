"""The extra code sessions: a second session every day at 13:00, and weekend mornings.

WHY THIS EXISTS - 2026-10-14, owner: *"i want you to be doing more, scheduling more runs. I want
to speed up the development of these tools please."* Until then the code loop ran once a weekday.

They are the same runner as the 06:00 session, so they inherit its four ship gates, its repo lock,
its merge-and-tag and its failure reporting. What is pinned here is what makes running it more
often safe:

* **Each slot has its own once-a-day marker**, so the morning's success does not cancel the
  afternoon, and a slot cannot run twice in a day.
* **An extra slot never stashes and never switches branches under someone.** At 13:00 or on a
  weekend, a dirty tree or a tree off `main` is work in progress, not a failed session's leftovers;
  the slot stands down and leaves it exactly as it was. The check comes before the runner changes
  anything (`Restore-Artifacts` would otherwise revert three files first).
* **An owner session marked in progress stands every slot down**, before any git command.
* **A usage limit stands the extra slots down for a day**, so the 06:00 session keeps first call on
  the weekly quota; the runner records one when a session dies on a 429.
* **No logon trigger and no late catch-up**, like the quote refresh: a late start is most likely to
  land on the owner working in the repo, and the staggered logon pair stays a pair.
* Retrospectives stay with the 06:00 slot.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTER = ROOT / "scripts" / "register-tasks.ps1"
NIGHTLY = ROOT / "scripts" / "nightly-screener.ps1"
TASKS = {"Screener Midday Session": ("-Slot midday", "1:00PM"),
         "Screener Weekend Session": ("-Slot weekend", "6:00AM")}


def _src(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig")


def _spec(name: str) -> str:
    src = _src(REGISTER)
    m = re.search(r"Name\s*=\s*'" + re.escape(name) + r"'", src)
    assert m, f"{name} has no entry in $Specs"
    block = src[m.start():]
    return block[:block.index("Description")]


def test_both_extra_sessions_are_registered_from_version_control():
    for name, (args, at) in TASKS.items():
        block = _spec(name)
        assert "Script      = 'nightly-screener.ps1'" in block, name
        assert f"Arguments   = '{args}'" in block and f"At          = '{at}'" in block, name


def test_the_midday_session_runs_every_day_and_the_weekend_one_on_weekends():
    mid, wkd = _spec("Screener Midday Session"), _spec("Screener Weekend Session")
    for day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"):
        assert f"'{day}'" in mid, day
    assert "'Saturday'" in wkd and "'Sunday'" in wkd and "'Monday'" not in wkd


def test_extra_sessions_neither_catch_up_late_nor_join_the_logon_pair():
    for name in TASKS:
        block = _spec(name)
        assert re.search(r"NoLogon\s*=\s*\$true", block) and re.search(r"NoCatchUp\s*=\s*\$true", block), name
        assert "LogonDelay" not in block, name


def test_the_registration_actually_passes_the_arguments_and_the_days():
    src = _src(REGISTER)
    assert "$s.Arguments" in src and '$extraArgs' in src and "`\"$scriptPath`\"$extraArgs" in src
    assert "if ($s.Days)" in src and "-DaysOfWeek $days" in src


def test_each_slot_keeps_its_own_once_a_day_marker_and_log():
    src = _src(NIGHTLY)
    assert '".nightly-last-success$SlotTag"' in src
    assert '"nightly-$Stamp$SlotTag.log"' in src and '"nightly-$Stamp$SlotTag.json"' in src
    assert "$Slots = @('', 'midday', 'weekend')" in src
    assert "if ($Slots -notcontains $Slot)" in src                      # an unknown slot does not run


def test_an_extra_slot_stands_down_on_work_in_progress_before_touching_anything():
    src = _src(NIGHTLY)
    check = src.index("    if ($IsExtra) {\n        $here")
    assert src.index("Enter-RepoLock -LogDir") < check                 # git only under the repo lock
    assert check < src.index("Restore-Artifacts 'pre-existing'")      # before anything is reverted
    assert check < src.index("'stash', 'push'")                         # never reaches the stash
    assert check < src.index("@('checkout', 'main')")                   # nor a branch switch
    block = src[check:src.index("Restore-Artifacts 'pre-existing'")]
    assert "$dirty -or $here -ne 'main'" in block and "exit 0" in block and "$SkipBrief = $true" in block


def test_an_owner_session_marker_stands_every_slot_down_before_any_git_command():
    src = _src(NIGHTLY)
    stand = src.index("$ownerAge = Get-MarkerAgeHours $OwnerSessionFile")
    assert stand < src.index("Enter-RepoLock -LogDir") and stand < src.index("Invoke-Native 'git'", stand)
    assert "$ownerAge -lt 8" in src                                      # a forgotten marker expires
    assert "Join-Path $LogDir '.owner-session'" in src


def test_a_usage_limit_is_recorded_and_stands_only_the_extra_slots_down():
    src = _src(NIGHTLY)
    assert "$IsExtra -and $null -ne $limitAge -and $limitAge -lt 24" in src
    record = src.index("[System.IO.File]::WriteAllText($UsageLimitFile")
    assert src.index('Write-Log "SESSION DID NOT RUN: $SessionError"') < record
    assert "'429|rate.?limit|usage limit'" in src


def test_retrospectives_stay_with_the_morning_slot_and_extra_slots_get_their_own_focus():
    src = _src(NIGHTLY)
    assert "if ($IsExtra) { $IsRetro = $false }" in src
    assert "'midday' = 'QUEUE." in src and "$FocusBySlot[$Slot]" in src
    assert "Not normally scheduled" not in src                          # weekends are scheduled now


def test_a_slot_that_stood_down_publishes_no_brief_and_releases_only_a_held_lock():
    src = _src(NIGHTLY)
    fin = src[src.index("finally {"):]
    assert "if (-not $SkipBrief) { Publish-Brief $briefLabel }" in fin
    assert "if ($RepoLock) { Exit-RepoLock $RepoLock }" in fin
    assert fin.index("Publish-Brief") < fin.index("Exit-RepoLock")      # the brief still runs under the lock


def test_the_ascii_rule_still_holds():
    """Windows PowerShell 5.1 reads these as cp1252 without a BOM; non-ASCII breaks the parser."""
    for p in (NIGHTLY, REGISTER):
        raw = p.read_bytes()
        assert raw.startswith(b"\xef\xbb\xbf"), p.name
        assert all(b < 128 for b in raw[3:]), p.name
