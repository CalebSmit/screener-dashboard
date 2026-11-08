"""The track record says its last holding period is still running (2026-11-06).

Each period runs from one month's ranking to the next, so the last has no successor: it ends on
the record's own last day. That night the Nov 3 period - three trading days old - was drawn and
counted like the month-long ones, with nothing to say so. The bar is now hatched and labelled
"so far", the caption says the count includes it, and the table's Held column says so. Checked on
the generator's own page with the published context; the period dates are swapped in and restored,
so the test asserts the rule, not which month is open.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None


@pytest.fixture(scope="module", params=[1280, 375])
def page(request, tmp_path_factory):
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("trackopen")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js", "dashboard_detail.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        pg = b.new_page(viewport={"width": request.param, "height": 900})
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.goto((d / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof renderTrack === 'function' && typeof D !== 'undefined' && D.track && "
                             "D.track.available && (D.track.periods || []).length >= 2", timeout=60000)
        pg.evaluate("toggleSection('sec-track')")
        # Measure in the page's own typeface: the fallback is narrower and clips nothing (the same
        # trap as the lab's phone budget, 2026-11-06 - fonts.status reads 'loaded' too early).
        pg.evaluate("document.fonts.ready.then(() => document.fonts.load('600 11px Inter')).then(() => 1)")
        pg.wait_for_function("[...document.fonts].some(f => f.family.replace(/\"/g, '') === 'Inter' && f.status === 'loaded')", timeout=30000)
        yield pg
        b.close()
        assert not errors, errors


# Render with the last period's exit set to `exit` (None keeps it), then read what the panel says.
_JS = """(exit) => {
    const T = D.track, last = T.periods[T.periods.length - 1], keep = last.exit;
    try {
        if (exit) last.exit = exit;
        renderTrack();
        const bars = [...document.querySelectorAll('#track-body .tr-lb')];
        const rows = [...document.querySelectorAll('#track-body .tr-periods tbody tr')];
        return {
            open: bars.map(b => b.classList.contains('open')),
            labels: bars.map(b => b.querySelector('.tr-lb-d').textContent.trim()),
            clipped: bars.map(b => { const e = b.querySelector('.tr-lb-d'), box = e.getBoundingClientRect();
                // each laid-out line of text, measured unrounded: an ellipsis clips a line wider than its box
                const r = document.createRange(); r.selectNodeContents(e);
                return [...r.getClientRects()].some(q => q.width > box.width + 0.25); }),
            caption: document.querySelector('#track-body .tr-lead-h span').textContent,
            held: rows.map(r => r.children[1].textContent.trim()),
            end: T.end, lastDate: last.date,
        };
    } finally { last.exit = keep; renderTrack(); } }"""


def test_the_last_period_is_marked_when_it_ends_on_the_records_last_day(page):
    end = page.evaluate("D.track.end")
    got = page.evaluate(_JS, end)
    assert got["open"][-1] is True and not any(got["open"][:-1]), got["open"]
    assert got["labels"][-1].endswith("so far") and not any(l.endswith("so far") for l in got["labels"][:-1])
    assert "which is still running" in got["caption"], got["caption"]
    assert got["held"][-1].endswith(", so far") and not any(h.endswith("so far") for h in got["held"][:-1])


def test_a_last_period_that_has_ended_is_not_marked(page):
    got = page.evaluate(_JS, "1999-01-01")          # an exit before the record's end: a closed period
    assert not any(got["open"]), got["open"]
    assert "still running" not in got["caption"]
    assert not any(h.endswith("so far") for h in got["held"])


def test_no_period_label_is_cut_off(page):
    got = page.evaluate(_JS, None)
    assert not any(got["clipped"]), list(zip(got["labels"], got["clipped"]))
