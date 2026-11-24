"""A push that fails must not jam both loops for a day (2026-11-23).

WHY THIS EXISTS. On 2026-11-23 the 02:00 data run passed every gate, committed, and its push died on
an HTTP 408. The runner stopped with "Push to main failed. Local main is ahead." Seventeen seconds
later it published the morning brief - a commit built on origin/main - so local main, still holding
the unpushed data commit, had diverged. The 06:00 code session, the 13:00 session and the 18:00 data
retry each stopped on "Not possible to fast-forward": one dropped connection cost the day's page and
three sessions, and nothing recovered until an owner session merged and pushed by hand that evening.

What is pinned here:

* **The push is tried several times**, with a wait between, and still names the data commit alone.
* **A push that never lands leaves local main level with origin/main.** The runner first asks origin
  whether the commit arrived after all; if not it takes its own commit back and discards the run the
  way every gate does (`Undo-RunArtifacts`), so the next run can fast-forward.
* **No success marker is written on that path**, so the 18:00 retry runs again on the day's close.
* The old ending, which left main ahead, is gone.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RUN = ROOT / "scripts" / "data-run.ps1"
PUSH = "'push', 'origin', \"${dataCommit}:refs/heads/main\""


def _src() -> str:
    return DATA_RUN.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


def _push_block() -> str:
    src = _src()
    start = src.index("$PushTries =")
    return src[start:src.index("Published to main", start)]


def test_the_push_is_tried_more_than_once_with_a_wait_between():
    block = _push_block()
    tries = int(re.search(r"\$PushTries = (\d+)", block).group(1))
    wait = int(re.search(r"\$PushWaitSeconds = (\d+)", block).group(1))
    assert tries >= 3, "one try is what lost 2026-11-23"
    assert 20 <= wait and tries * wait <= 600, "long enough for a blip to pass, short enough for an unattended run"
    loop = block[block.index("foreach ($try in 1..$PushTries)"):block.index("if ($push.ExitCode -ne 0) {")]
    assert PUSH in loop and "break" in loop and "Start-Sleep -Seconds $PushWaitSeconds" in loop


def test_the_push_still_names_the_data_commit_and_nothing_else():
    src = _src()
    assert src.count(PUSH) == 1
    assert "'push', 'origin', 'main'" not in src and "'push', 'origin', 'HEAD" not in src


def test_a_push_that_never_lands_leaves_main_level_with_origin():
    block = _push_block()
    failed = block[block.index("if ($push.ExitCode -ne 0) {"):]
    fetch = failed.index("'fetch', 'origin', 'main'")
    arrived = failed.index("if ($remoteHead -eq $dataCommit)")
    own = failed.index("if ($localHead -eq $dataCommit)")
    reset = failed.index("'reset', '-q', 'HEAD~1'")
    undo = failed.index("Undo-RunArtifacts")
    stop = failed.index("Stop-Run")
    assert fetch < arrived < own < reset < undo < stop, \
        "ask origin first; take back only this run's own commit; discard; then stop"
    assert "--hard" not in failed, "the discard is the one every gate uses, not a hard reset"


def test_the_ending_that_left_main_ahead_is_gone():
    assert "Local main is ahead." not in _src()


def test_a_run_that_did_not_publish_writes_no_success_marker():
    src = _src()
    stop = src.index("Push to main failed $PushTries times")
    marker = src.index("[System.IO.File]::WriteAllText($SuccessMarker, $Date)")
    assert stop < marker, "Stop-Run exits before the marker, so the 18:00 retry runs"
