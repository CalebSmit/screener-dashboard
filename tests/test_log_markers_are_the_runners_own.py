"""A run's outcome is read from the lines its runner wrote, never from text it echoed.

WHY THIS EXISTS - 2026-11-03. The midday session found that ``health_numbers`` read
"SESSION DID NOT RUN" back out of the health line the runner echoes into every log, so one real
failure made every later session look dead. Its Next asked for a sweep of every other reader. The
brief's ``outcome()`` and ``newest_session()`` and ``health_numbers.data_loop()`` all asked the
same kind of question - is this phrase anywhere in the text? - while the runners echo git output,
the merge stat, the commit subjects they are about to merge and the health line into the same log.
Measured on four echo cases built from real lines (``research/measurements/2026-11-03-log-marker-echoes.py``):
a failed run was called "completed" twice and "stopped deliberately" once, and an echoed
"Standing down" subject made a real session invisible. On all 39 real logs the new reading agrees
with the old one exactly.

The same sweep found the health line's "finished" test keyed to "Morning brief published", which
a run that cannot reach GitHub never writes: the 06:00 run skipped that morning for a DNS failure
was passed over, and the line reported the day before's login failure in its place.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import health_numbers as hn  # noqa: E402
import write_brief as wb  # noqa: E402

FAILED_RUN = "[06:00:01] [ERROR] Cannot reach the git remote (offline, or credentials unavailable). Skipping today's run."

ECHOES = [
    "[13:53:55] [INFO]     0043f03 Log: the fix shipped to main at last",              # a commit subject
    "[13:53:55] [INFO]     1a2b3c4 Explain why a run is discarded",
    "[13:53:55] [INFO]     9f8e7d6 === Run complete: shipped to main === in the log",
    "[08:07:01] [INFO] Health: **Health (rule 8, all five):** last code session ran? yes - "
    "`logs/x.log` (completed). Data loop published? yes - (completed; HEALTH: PASS)",  # the health line
    "[13:53:56] [INFO]      SHIP GATES FAILED: in a merge stat | 2 +-",
]


@pytest.mark.parametrize("echo", ECHOES)
def test_an_echo_cannot_change_a_failed_runs_verdict(echo):
    assert wb.outcome([FAILED_RUN, echo])[0] == "failed"


def test_an_echoed_stand_down_does_not_hide_a_real_session(tmp_path, monkeypatch):
    monkeypatch.setattr(wb, "LOGS", tmp_path)
    p = tmp_path / "nightly-2026-11-03_060000.log"
    p.write_text(FAILED_RUN + "\n[13:53:55] [INFO]     5d6e7f8 Standing down the extra slots on a 429\n",
                 encoding="utf-8")
    assert wb.newest_session() == p


@pytest.mark.parametrize("line, state", [
    ("[02:23:19] [INFO] === Data loop complete ===", "completed"),
    ("[13:54:25] [INFO] === Run complete: shipped to main ===", "completed"),
    ("[06:40:00] [INFO] === Run complete (no changes) ===", "completed, nothing to change"),
    ("[02:20:00] [ERROR] Run discarded by health check - the live dashboard is unchanged.", "stopped deliberately"),
    ("[06:30:00] [ERROR] SHIP GATES FAILED: tests. Not merging.", "stopped deliberately"),
    ("﻿[02:23:19] [INFO] === Data loop complete ===", "completed"),     # first line carries the BOM
])
def test_the_runners_own_lines_still_classify(line, state):
    assert wb.outcome([FAILED_RUN, line])[0] == state


def _code(path: Path) -> str:
    return "\n".join(ln for ln in path.read_text(encoding="utf-8-sig").splitlines()
                     if not ln.lstrip().startswith("#"))


def test_every_marker_is_a_line_a_runner_writes():
    """The constants are the runners' own words; reword a runner line and this fails, rather than
    the brief quietly classifying every run as failed."""
    runners = _code(ROOT / "scripts" / "data-run.ps1") + _code(ROOT / "scripts" / "nightly-screener.ps1")
    written = re.findall(r"(?:Write-Log|Stop-Run)\s+\"([^\"]*)\"", runners)
    for marker in (*wb.COMPLETED, *wb.NO_CHANGES, *wb.STOPPED, *wb.FINISHED, *wb.NOT_A_SESSION):
        assert any(w.startswith(marker) for w in written), f"no runner writes a line starting {marker!r}"


def _log(folder: Path, name: str, body: str, mtime: float):
    p = folder / name
    p.write_text(body, encoding="utf-8")
    os.utime(p, (mtime, mtime))


def test_a_run_that_could_not_publish_its_brief_is_still_the_last_session(tmp_path, monkeypatch):
    """2026-11-03 06:00: skipped for DNS, so no "Morning brief published" - but it ran its finally."""
    monkeypatch.setattr(hn.wb, "LOGS", tmp_path)
    _log(tmp_path, "nightly-2026-11-02_130000-midday.log",
         "[13:04:04] [ERROR] SESSION DID NOT RUN: session reported an error - Failed to authenticate\n"
         "[13:08:07] [INFO] Writing morning brief...\n[13:08:13] [INFO] Morning brief published.\n", 1000)
    _log(tmp_path, "nightly-2026-11-03_060000.log",
         FAILED_RUN + "\n[06:00:01] [INFO] Writing morning brief...\n"
         "[06:00:03] [WARN] Could not fetch origin/main to publish the brief: fatal: unable to access\n", 2000)
    got = hn.last_code_session()
    assert got.startswith("**failed** - `logs/nightly-2026-11-03_060000.log`"), got
    assert "Cannot reach the git remote" in got and "Failed to authenticate" not in got


def test_the_data_loop_reads_its_own_verdicts(tmp_path, monkeypatch):
    monkeypatch.setattr(hn.wb, "LOGS", tmp_path)
    _log(tmp_path, "datarun-2026-11-03_020000.log",
         "[02:19:15] [INFO]     HEALTH: PASS - safe to publish\n"
         "[02:23:19] [INFO] === Data loop complete ===\n", 1000)
    assert hn.data_loop().startswith("yes") and "HEALTH: PASS" in hn.data_loop()
    _log(tmp_path, "datarun-2026-11-04_020000.log",
         "[02:19:15] [INFO]     HEALTH: DEGRADED - discard\n"
         "[02:19:16] [INFO]     abc1234 data: the run where the Data loop complete line moved\n"
         "[02:19:17] [ERROR] Run discarded by health check - the live dashboard is unchanged.\n", 2000)
    got = hn.data_loop()
    assert got.startswith("**no**") and "no HEALTH: PASS line" in got and "stopped deliberately" in got
