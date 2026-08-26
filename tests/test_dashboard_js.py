"""The emitted dashboard script must be syntactically valid JavaScript.

`generate_dashboard.generate_html()` builds the entire front end - roughly two
thousand lines of JS - inside a Python f-string, so every literal brace in that
JS has to be doubled. Get one wrong and the browser fails to parse the whole
script: the page renders as an empty shell with no table, no charts and no
error message, while every existing ship gate still passes. Gate 3 checks that
`dashboard_data.js` parses; nothing checked the script that consumes it.

This is not hypothetical. NIGHTLY_LOG.md 2026-08-24 records a near-miss of the
same shape (a change that "looked right, passed a shallow check, and would have
broken the unattended run"), caught only because `--help` happened to be
inspected. `main` is served live to the public by GitHub Pages with nobody
watching, so a blank dashboard would stay blank until someone looked.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import generate_dashboard as g  # noqa: E402

NODE = shutil.which("node")


def _script_blocks(html: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r"<script>(.*?)</script>", html, re.S)]


@pytest.fixture(scope="module")
def html() -> str:
    # No run directory needed: generate_html renders the template and loads its
    # data from the companion file at runtime.
    return g.generate_html()


def test_html_contains_exactly_one_inline_script(html):
    assert len(_script_blocks(html)) == 1


@pytest.mark.skipif(NODE is None, reason="node not available")
def test_inline_script_is_valid_javascript(html, tmp_path):
    """A doubled-brace mistake in the f-string template shows up here."""
    script = _script_blocks(html)[0]
    path = tmp_path / "dashboard-inline.js"
    path.write_text(script, encoding="utf-8")

    result = subprocess.run([NODE, "--check", str(path)],
                            capture_output=True, text=True)

    assert result.returncode == 0, (
        "emitted dashboard JS does not parse:\n" + result.stderr[:2000])


def test_no_unescaped_format_placeholders_survived(html):
    """A single brace in the template silently eats its contents.

    `{foo}` in an f-string is evaluated, not emitted, so the failure is a
    *missing* fragment rather than a stray one. Catching the inverse - a
    literal `{{` reaching the output because it was over-escaped - is the
    cheap half of the check.
    """
    script = _script_blocks(html)[0]
    assert "{{" not in script
    assert "}}" not in script


@pytest.mark.parametrize("hook", [
    "sec-changed",          # the What Changed section
    "movers-up",
    "movers-down",
    "changed-caption",
    "changed-footnote",
    "th-rank-delta",        # the universe-table delta column
    "section-history",      # per-stock rank history in the drill-down
    "modal-history",
])
def test_history_ui_hooks_are_present(html, hook):
    """The time dimension is the dashboard's biggest documented gap
    (plan/dashboard-north-star.md). If a refactor drops one of these
    ids the corresponding panel silently renders nothing."""
    assert hook in html


def test_history_render_is_wired_into_startup(html):
    script = _script_blocks(html)[0]
    assert "renderChanged();" in script
    assert "renderStockHistory(ticker);" in script
