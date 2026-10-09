"""No test may pin the size of the S&P 500.

WHY THIS EXISTS - 2026-10-09.

The 02:00 data run scored **501** stocks instead of the 502 of the days before - an ordinary
change in index membership, and a correct run. Three tests in ``tests/test_dashboard_browser.py``
asserted the literal ``502`` / ``503`` against the published payload and failed.

That is worse than three red tests, because of how the ship gates are wired: the runner's gate 1
is ``python -m pytest tests/ test_screener.py -q`` and it merges only on **exit code 0**
(``scripts/nightly-screener.ps1``, "Gate 1: tests"). It takes no baseline, deliberately - a gate
that tolerates "the same failures as yesterday" is not a gate. So a literal that the *data* can
move on its own does not fail one test, it **blocks every merge, on every branch, until a human
edits the number**. The code loop would have stopped improving the tool for a reason that has
nothing to do with any session's work. The S&P 500 changes constituents several times a year.

The fix in that module was to assert the invariant (the table's row count, last rank and count
text all agree with the payload's own universe) within a plausibility band. This module is the
tripwire that keeps the literal from coming back somewhere else: any test that reads a published
artifact must not compare anything to a number in the band a universe size falls in.

The band is the one ``universe_history.validate_membership`` already refuses outside of, widened a
little. If a test legitimately needs a number in it - a budget, a byte count, a pixel width - give
it a name from ``ALLOWED_CONTEXT`` on the same line, or put it in a module that reads no published
artifact.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# The range a plausible S&P 500 universe size falls in. A literal in here, asserted against
# something read out of a published artifact, is the defect.
BAND = range(460, 521)

# Reads one of the generated artifacts, so its assertions see live, moving data.
ARTIFACT_MARKERS = ("dashboard_data.js", "dashboard_context.js", "SCREENER_DATA",
                    "SCREENER_CONTEXT", "index.html")

# Numbers in the band that are plainly not a universe count. Any of these words on the same
# line makes the literal fine.
ALLOWED_CONTEXT = ("px", "pixel", "width", "height", "scrollWidth", "bytes", "byte", "KB", "ms",
                   "millisecond", "timeout", "wait_for", "port", "seconds", "viewport")


def _test_sources() -> list[Path]:
    files = sorted(ROOT.joinpath("tests").glob("test_*.py"))
    extra = ROOT / "test_screener.py"
    if extra.exists():
        files.append(extra)
    return [f for f in files if f.name != Path(__file__).name]


def _reads_a_published_artifact(text: str) -> bool:
    return any(m in text for m in ARTIFACT_MARKERS)


def test_no_test_that_reads_the_published_payload_pins_a_universe_size():
    offenders: list[str] = []
    for path in _test_sources():
        text = path.read_text(encoding="utf-8")
        if not _reads_a_published_artifact(text):
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            stripped = line.strip()
            if not stripped.startswith("assert"):
                continue
            if any(word in line for word in ALLOWED_CONTEXT):
                continue
            # a comparison against a bare literal, e.g. `== 502`, `== "503"`, `== 501,`
            for m in re.finditer(r'(==|!=)\s*["\']?(\d{3})["\']?(?!\d)', stripped):
                if int(m.group(2)) in BAND:
                    offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {stripped[:120]}")
    assert not offenders, (
        "These assertions pin a number in the S&P 500 universe band against live payload data. "
        "Index membership moves on its own, gate 1 merges only on pytest exit 0, so one of these "
        "blocks every merge on the day the count changes. Assert the invariant instead:\n  "
        + "\n  ".join(offenders)
    )


def test_the_tripwire_can_actually_fire():
    """A tripwire wired to something that cannot happen is decoration (CLAUDE.md rule 8)."""
    sample = 'assert page.evaluate("window.SCREENER_DATA.table_data.length") == 502'
    hits = [m for m in re.finditer(r'(==|!=)\s*["\']?(\d{3})["\']?(?!\d)', sample)
            if int(m.group(2)) in BAND]
    assert hits, "the pattern no longer matches the exact line that failed on 2026-10-09"
    assert not any(w in sample for w in ALLOWED_CONTEXT)


def test_the_band_matches_the_membership_validator():
    """Same band the universe validator already enforces, so there is one definition of
    'a plausible S&P 500', not two that can drift apart."""
    import universe_history

    src = Path(universe_history.__file__).read_text(encoding="utf-8")
    assert "495" in src and "515" in src, (
        "universe_history no longer names the 495-515 band this module was widened from; "
        "re-derive BAND from whatever replaced it."
    )
    assert BAND.start <= 495 and BAND.stop - 1 >= 515


def test_the_browser_module_is_covered_by_this_scan():
    """The module whose failure prompted this must be one the scan actually looks at."""
    browser = ROOT / "tests" / "test_dashboard_browser.py"
    assert browser.exists()
    assert _reads_a_published_artifact(browser.read_text(encoding="utf-8"))
    assert browser in _test_sources()
