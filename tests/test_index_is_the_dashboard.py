"""`index.html` is the dashboard, not a redirect to it.

WHY THIS EXISTS - 2026-10-08.

`index.html` is what GitHub Pages serves. Step 12 of `run_screener.py` wrote a ~350-byte
`<meta http-equiv="refresh">` stub there, and `scripts/data-run.ps1` then copied
`dashboard.html` over it. So the scheduled path published the real page and never showed the
problem, while a plain `python run_screener.py` left the published file as a redirect and the
tree **failing ship gate 3**, which requires `index.html` to be larger than 50,000 bytes
(`scripts/nightly-screener.ps1`). Found by looking at the file after a rescoring run, which
is the only reason it was not committed.

This is the same asymmetry CLAUDE.md warns about under the data loop's gates: two paths do
the same job, and the weaker one is the bug.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GATE_3_FLOOR = 50_000  # scripts/nightly-screener.ps1, gate 3


def test_the_published_file_is_not_a_redirect():
    html = (ROOT / "index.html").read_text(encoding="utf-8", errors="replace")
    assert "http-equiv" not in html.lower() or "refresh" not in html.lower(), (
        "index.html is a <meta refresh> stub; GitHub Pages serves this file, and ship gate 3 "
        "requires a real page")


def test_the_published_file_clears_the_gate_3_floor():
    size = (ROOT / "index.html").stat().st_size
    assert size > GATE_3_FLOOR, (
        f"index.html is {size} bytes; ship gate 3 requires more than {GATE_3_FLOOR}")


def test_the_published_file_is_the_dashboard():
    index = (ROOT / "index.html").read_bytes()
    dash = (ROOT / "dashboard.html").read_bytes()
    assert index == dash, (
        "index.html and dashboard.html have diverged; both publish paths copy one to the "
        "other, so a difference means one of them is stale")


def test_the_screener_copies_the_dashboard_rather_than_writing_a_stub():
    """Pinned against the generator, not the artifact - a hand-fixed `index.html` would hide
    the next run putting the stub back (CLAUDE.md rule 10)."""
    src = (ROOT / "run_screener.py").read_text(encoding="utf-8")
    step12 = src[src.index("# ---- 12. Generate interactive dashboard"):][:3000]
    assert 'ROOT / "index.html"' in step12, "step 12 no longer writes index.html at all"
    assert "http-equiv" not in step12, (
        "run_screener.py step 12 writes a redirect stub to index.html again; it must copy the "
        "generated dashboard, which is what data-run.ps1 does after it")
    assert re.search(r'copy2\(\s*dash_path\s*,\s*ROOT\s*/\s*"index\.html"\s*\)', step12), \
        "step 12 should copy the generated dashboard to index.html"


def test_both_publish_paths_still_put_the_dashboard_in_index():
    """The PowerShell loop's copy stays too: belt and braces on the file the public reads."""
    ps = (ROOT / "scripts" / "data-run.ps1").read_text(encoding="utf-8", errors="replace")
    assert re.search(r"Copy-Item\s+\$dash\s+\$idx", ps), (
        "data-run.ps1 no longer copies dashboard.html -> index.html")
