"""``scripts/health_numbers.py`` prints rule 8's five numbers from the brief's own functions.

Pinned: the last code session is the newest *finished* one (the session running it is inside an
unfinished log), a dry run or a stand-down is never "the last session", the top roadmap item is the
first dated item under "### Open" that its own heading does not mark closed, ages are computed
rather than copied, and the script never fails its caller.
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import health_numbers as hn  # noqa: E402

DAY = datetime(2026, 11, 2, 19, 0)

CLAUDE_FIXTURE = """# x
### Settled
**0.9. Old thing (opened 2026-09-01)** - not under Open, ignored.
### Open
**0.8 and 0.9 are CLOSED** (2026-10-07).
**0.11. The context layer - CLOSED 2026-10-20** (record ...).
**0.13. The Factor Lab (opened 2026-10-12).** Owner item.
**0.10. Keep the inputs (opened 2026-10-07).**
**0.14. The inputs against the filings (opened 2026-10-14, owner-run).**
**0.6. One observation per run date - CLOSED 2026-10-08 (opened 2026-09-30).**
"""


def test_open_items_are_read_in_order_with_computed_ages():
    items = hn.open_items(CLAUDE_FIXTURE, today=DAY)
    assert items == [("0.13", "The Factor Lab", 21), ("0.10", "Keep the inputs", 26),
                     ("0.14", "The inputs against the filings", 19)]


def test_the_real_file_has_a_dated_open_item():
    """Asserts the premise, not which item: the list changes as items close."""
    items = hn.open_items(today=DAY)
    assert items, "no dated item under ### Open - rule 8's roadmap line has nothing to read"
    assert all(age >= 0 for _, _, age in items)


def _log(folder: Path, name: str, body: str, mtime: float):
    p = folder / name
    p.write_text(body, encoding="utf-8")
    os.utime(p, (mtime, mtime))


def test_the_last_session_is_the_newest_finished_one(tmp_path, monkeypatch):
    monkeypatch.setattr(hn.wb, "LOGS", tmp_path)
    _log(tmp_path, "nightly-a.log", "[06:00:00] [INFO] shipped to main\n[06:40:00] [INFO] Morning brief published.\n", 1000)
    _log(tmp_path, "nightly-b.log", "[13:00:00] [INFO] === Dry run complete ===\n[13:00:01] [INFO] Morning brief published.\n", 2000)
    _log(tmp_path, "nightly-c.log", "[13:00:00] [INFO] Invoking Claude Code...\n", 3000)   # running now
    got = hn.last_code_session()
    assert got.startswith("yes - `logs/nightly-a.log`")


def test_a_session_that_did_not_run_says_so_and_why(tmp_path, monkeypatch):
    monkeypatch.setattr(hn.wb, "LOGS", tmp_path)
    _log(tmp_path, "nightly-a.log",
         "[06:00:04] [ERROR] SESSION DID NOT RUN: session reported an error - Failed to authenticate\n"
         "[06:08:00] [INFO] Morning brief published.\n", 1000)
    got = hn.last_code_session()
    assert got.startswith("**no**") and "Failed to authenticate" in got


def test_the_top_item_line_names_the_first_and_counts_the_rest(monkeypatch):
    monkeypatch.setattr(hn, "open_items", lambda claude_md=None, today=None:
                        [("0.13", "The Factor Lab", 21), ("0.10", "Keep the inputs", 26)])
    assert hn.top_roadmap(DAY) == "**0.13, The Factor Lab** (age **21 days**); also open: 0.10 26d"


def test_the_script_prints_one_line_and_never_fails_its_caller():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "health_numbers.py")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0
    out = r.stdout.strip()
    assert out.count("\n") == 0 and out.startswith("**Health (rule 8, all five):**")
    for part in ("last code session ran?", "Data loop published?", "Evidence base at `1m`",
                 "Priority 0", "Top open roadmap item"):
        assert part in out


def test_the_nightly_prompt_points_at_it():
    assert "scripts/health_numbers.py" in (ROOT / "prompts" / "nightly.md").read_text(encoding="utf-8")
