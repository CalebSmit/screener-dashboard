"""Nothing on the page is cut short or leaves a hole, at the widths people actually use.

WHY THIS EXISTS - owner request, 2026-11-05: *"make sure it is looking professional and
expensive."* An audit of the generator's page at 1440, 1280, 1100, 1000 and 375px found the kind of
flaw that reads as unfinished rather than wrong:

* the five leader cards sat three-and-two with an empty slot from 761 to 1100px;
* three leader cards cut the company name ("Valero Energy Cor...") at 1280px;
* five sector names in the trap-rate bars were cut ("Consumer Discr...") at every desktop width,
  and below ~1240px the sector table scrolled sideways inside its own panel;
* on a phone, three sector names in the Market panel's rate chart were cut.

Each is measured here in a real browser on the generator's own page and the published payloads -
geometry only, no stock, value or date named (CLAUDE.md "data->red", "clock->red").
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


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    if sync_api is None:
        pytest.skip("playwright not installed")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        if not (ROOT / f).exists():
            pytest.skip(f"MISSING DATA: no published {f}")
    d = tmp_path_factory.mktemp("polish")
    (d / "index.html").write_text(gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t",
                                                   data_version="v"), encoding="utf-8")
    for f in ("dashboard_data.js", "dashboard_context.js"):
        shutil.copy(ROOT / f, d / f)
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b, (d / "index.html").as_uri()
        b.close()


def _open(site, width: int, height: int = 900):
    b, url = site
    pg = b.new_page(viewport={"width": width, "height": height})
    errors: list[str] = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(url, wait_until="load", timeout=60000)
    pg.wait_for_function("typeof toggleSection === 'function' && window.SCREENER_CONTEXT !== undefined "
                         "&& document.querySelectorAll('.top5-card:not(.sk)').length > 0", timeout=60000)
    return pg, errors


_CLIPPED = """sel => [...document.querySelectorAll(sel)].filter(e => e.getBoundingClientRect().width > 0 &&
    (e.scrollWidth > e.clientWidth + 1 || e.scrollHeight > e.clientHeight + 1)).map(e => e.textContent.trim())"""


@pytest.mark.parametrize("width", [800, 1000, 1100])
def test_the_leader_cards_leave_no_empty_slot_on_a_tablet_or_small_laptop(site, width):
    pg, errors = _open(site, width)
    try:
        rows = pg.evaluate("""() => { const by = {};
            document.querySelectorAll('#top5-row > .top5-card').forEach(c => { const r = c.getBoundingClientRect();
                const k = Math.round(r.top); (by[k] = by[k] || []).push([r.left, r.right]); });
            const row = document.getElementById('top5-row').getBoundingClientRect();
            return { row: [row.left, row.right], lines: Object.values(by) }; }""")
        if sum(len(x) for x in rows["lines"]) < 5:
            pytest.skip("MISSING DATA: fewer than five leaders published")
        left, right = rows["row"]
        for line in rows["lines"]:
            assert min(a for a, _ in line) == pytest.approx(left, abs=1.5)
            assert max(z for _, z in line) == pytest.approx(right, abs=1.5), (width, rows)
        assert errors == []
    finally:
        pg.close()


@pytest.mark.parametrize("width", [1100, 1280, 1440])
def test_no_leader_name_or_trap_sector_is_cut_short_and_the_sector_table_fits(site, width):
    pg, errors = _open(site, width)
    try:
        assert pg.evaluate(_CLIPPED, ".top5-company") == []
        pg.evaluate("goToSection('sec-analytics')")
        pg.wait_for_selector("#trap-bars .trap-row", state="attached", timeout=15000)
        assert pg.evaluate(_CLIPPED, ".trap-name") == []
        overflow = pg.evaluate("(() => { const m = document.getElementById('sector-matrix'); return m.scrollWidth - m.clientWidth; })()")
        assert overflow <= 1, f"the sector table scrolls sideways by {overflow}px at {width}px"
        assert errors == []
    finally:
        pg.close()


def test_the_leaders_bars_start_level_whatever_the_name_length(site):
    pg, _ = _open(site, 1280)
    try:
        tops = pg.evaluate("[...document.querySelectorAll('#top5-row .t5-fp')].map(e => Math.round(e.getBoundingClientRect().top))")
        assert len(set(tops)) <= 1, tops
    finally:
        pg.close()


def test_on_a_phone_no_market_sector_name_is_cut_short(site):
    pg, errors = _open(site, 375, 812)
    try:
        pg.evaluate("goToSection('sec-market')")
        pg.wait_for_selector("#market-body .mk-grid", state="attached", timeout=15000)
        if not pg.evaluate("document.querySelectorAll('.mk-rate-n').length"):
            pytest.skip("MISSING DATA: no sector rate medians published")
        assert pg.evaluate(_CLIPPED, ".mk-rate-n") == []
        assert pg.evaluate("document.documentElement.scrollWidth") <= 375
        assert errors == []
    finally:
        pg.close()


@pytest.mark.parametrize("width,height", [(1440, 900), (1280, 900), (375, 812)])
def test_each_leaders_run_line_fits_on_one_line_and_the_columns_start_level(site, width, height):
    """The rank-by-run columns under each leader (2026-11-05): their caption on one line, and the five
    sets of columns level with one another, so the cards still read as a row."""
    pg, errors = _open(site, width, height)
    try:
        caps = pg.evaluate("[...document.querySelectorAll('#top5-row .t5-runs-cap')].map(e => [e.getBoundingClientRect().height, parseFloat(getComputedStyle(e).lineHeight) || 0])")
        if not caps:
            pytest.skip("MISSING DATA: no comparable history to draw under the leaders")
        assert all(h <= (lh or 18) * 1.5 for h, lh in caps), caps
        if width > 760:
            tops = pg.evaluate("[...document.querySelectorAll('#top5-row .t5-runs')].map(e => Math.round(e.getBoundingClientRect().top))")
            assert len(set(tops)) <= 1 or width < 1240, tops
        assert errors == []
    finally:
        pg.close()


@pytest.mark.parametrize("width,height", [(1440, 900), (1100, 900), (800, 800), (375, 812)])
def test_a_holding_cards_line_change_and_score_never_split_and_nothing_overflows(site, width, height):
    """The rank line added to My holdings (2026-11-05) pushed the composite onto a row of its own at
    800px. The line, the change, the score and the remove button now wrap as one group: on a desktop
    they share one row; on a phone the change and the score share one and the line spans the card."""
    pg, errors = _open(site, width, height)
    try:
        picks = pg.evaluate("D.table_data.slice().sort((a, b) => a.Rank - b.Rank).filter((r, i) => i % 97 === 3).slice(0, 4).map(r => r.Ticker)")
        pg.evaluate("ts => { ts.forEach(t => addHolding(t)); }", picks)
        pg.wait_for_function("document.querySelectorAll('#holdings-body .holding-card .holding-end').length >= 1", timeout=8000)
        if not pg.evaluate("document.querySelectorAll('#holdings-body .holding-path').length"):
            pytest.skip("MISSING DATA: no comparable history to draw")
        rows = pg.evaluate("""() => [...document.querySelectorAll('#holdings-body .holding-end')].map(g => {
            const top = e => e ? Math.round(e.getBoundingClientRect().top + e.getBoundingClientRect().height / 2) : null;
            return [top(g.querySelector('.holding-delta')), top(g.querySelector('.holding-composite')), top(g.querySelector('.holding-remove')),
                    top(g.querySelector('.holding-path')), g.querySelector('.holding-path') ? g.querySelector('.holding-path').getBoundingClientRect().width : 0,
                    g.closest('.holding-card').getBoundingClientRect().width]; })""")
        for delta, comp, rm, path, pw, cw in rows:
            assert abs(delta - comp) <= 3 and abs(comp - rm) <= 3, (width, rows)
            if path is None:
                continue
            if width > 560:
                assert abs(path - comp) <= 3, (width, rows)
            else:
                assert path > comp and pw > cw * 0.7, (width, rows)
        assert pg.evaluate("document.documentElement.scrollWidth") <= width
        assert errors == []
    finally:
        pg.evaluate("() => { try { localStorage.clear(); } catch (e) {} }")
        pg.close()

