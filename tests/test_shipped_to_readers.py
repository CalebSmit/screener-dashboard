"""The brief's and the health line's one number for product progress (2026-11-19).

`write_brief.shipped_to_readers`: commits in the last week that changed a reader-facing file, and
methodology changelog entries in the same week. 17 scheduled sessions shipped no reader-visible change
between 2026-11-05 and 11-19 and nothing said so (`research/2026-11-19-fresh-eyes-review.md` 4).
"""
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import write_brief as wb  # noqa: E402

DAY = datetime(2026, 11, 20, 6, 0)


def _log(*commits):
    return "".join("\x00" + subj + "\n\n" + "\n".join(files) + "\n" for subj, files in commits)


def test_counts_commits_that_touch_what_a_reader_sees():
    log = _log(("Fix the sheet's caveat", ["stock_summary.py", "tests/test_x.py"]),
               ("Runner hardening", ["scripts/data-run.ps1", "tests/test_y.py"]),
               ("data: screener run 2026-11-19", ["dashboard_data.js", "index.html"]),
               ("brief: code session", ["MORNING_BRIEF.md"]),
               ("Lab experiment 20", ["factor_lab.py", "lab.html"]))
    out = wb.shipped_to_readers(log_text=log, changelog="", today=DAY)
    assert out.startswith("2 commit(s) changed what a reader sees, 0 methodology change(s)")


def test_counts_changelog_entries_inside_the_window_only():
    cl = "## 2026-11-19 - a\n\n## 2026-11-14 - b\n\n## 2026-11-12 - c\n"
    out = wb.shipped_to_readers(log_text="", changelog=cl, today=DAY)
    assert "0 commit(s)" in out and "2 methodology change(s)" in out


def test_a_week_with_nothing_says_so():
    out = wb.shipped_to_readers(log_text=_log(("Tests only", ["tests/test_z.py"])), changelog="", today=DAY)
    assert "nothing reached a reader this week" in out


def test_it_is_in_the_brief_and_the_health_line():
    src = (ROOT / "scripts" / "write_brief.py").read_text(encoding="utf-8")
    assert '"| Shipped to readers | {shipped_to_readers()} |"' in src.replace("f\"", "\"")
    hn = (ROOT / "scripts" / "health_numbers.py").read_text(encoding="utf-8")
    assert "wb.shipped_to_readers(" in hn
