"""The page's second improvement round (2026-10-27, owner-run; ``plan/ui-elevation.md``, "Round two"),
in a real browser.

Measured first, then fixed; every check asserts a behaviour on the committed page
(``tests/test_rebuild_page.py`` holds it equal to the generator's output):

1. **An axe-core audit** (run by hand, not here - the suite never touches the network) found no
   ``main`` landmark, so every section sat outside a landmark (``landmark-one-main``, ``region``);
   the rankings' column headers at 4.16:1 against their raised background; links inside sentences
   told apart by hue alone; scrolling boxes a keyboard could not reach (the methodology's formulas,
   the factor-correlation grid on a phone); and two empty table headers.
2. **Scrolling the rankings on a 4x-throttled phone** rebuilt all ~37 rows on every row scrolled:
   p95 frame 183 ms. The window now adds and drops only the rows that enter and leave it.
3. **The leftovers of the first pass**: the sector-by-factor table on a phone showed two of nine
   columns behind a sideways scroll; Diagnostics' chevron sat a line below its title; the rank-change
   column was headed by a bare Greek capital.
4. **The methodology's own cross-references** ("see bank-specific scoring") changed the address bar
   and nothing else, because the markdown renders without heading ids.

Skips loudly where Playwright or a browser is not installed. Waits are conditions with 8 s timeouts,
never fixed sleeps.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"

pytestmark = pytest.mark.skipif(not PAGE.exists() or not PAYLOAD.exists(),
                                reason="generated dashboard not present")

READY = "document.querySelectorAll('#universe-tbody tr[data-t]').length > 0"
WAIT = 8000


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _open(browser, width=1280, height=900, touch=False, reduced_motion=None):
    opts = {"viewport": {"width": width, "height": height}}
    if touch:
        opts.update(is_mobile=True, has_touch=True)
    if reduced_motion:
        opts["reduced_motion"] = reduced_motion
    ctx = browser.new_context(**opts)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri(), wait_until="load", timeout=60000)
    page.wait_for_function(READY, timeout=30000)
    return ctx, page, errors


def _lum(rgb: str) -> float:
    vals = [int(v) / 255 for v in re.findall(r"\d+(?:\.\d+)?", rgb)[:3]]
    lin = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in vals]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(a: str, b: str) -> float:
    la, lb = _lum(a), _lum(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# --- 1. the audit's findings ---------------------------------------------------------------------

def test_one_main_landmark_holds_the_sections_and_nothing_else(browser):
    ctx, page, errors = _open(browser)
    try:
        got = page.evaluate("""() => {
            const mains = document.querySelectorAll('main, [role=main]');
            const m = mains[0];
            const inside = id => !!(m && document.getElementById(id) && m.contains(document.getElementById(id)));
            return { n: mains.length,
                     sections: ['kpi-row', 'dist', 'sec-top5', 'sec-holdings', 'sec-analytics', 'sec-defensibility', 'sec-universe'].map(inside),
                     outside: ['top-bar', 'stock-modal', 'methodology-modal', 'compare-modal'].map(inside),
                     footer: !!(m && m.contains(document.querySelector('.dashboard-footer'))),
                     loose: [...document.querySelectorAll('.dashboard-container > section')].map(s => s.id) };
        }""")
        assert got["n"] == 1, got
        assert all(got["sections"]), got
        assert not any(got["outside"]) and not got["footer"], got
        assert got["loose"] == [], got
        assert errors == []
    finally:
        ctx.close()


def test_wide_layout_keeps_the_statement_beside_its_figures(browser):
    """The >= 1200px grid moved from the container to <main>; the hero and the four figures still
    share the first row."""
    ctx, page, _ = _open(browser, 1440, 900)
    try:
        r = page.evaluate("""() => { const h = document.querySelector('.hero').getBoundingClientRect(),
                                      k = document.getElementById('kpi-row').getBoundingClientRect();
                                      return [h.left, h.right, h.top, h.bottom, k.left, k.top, k.bottom]; }""")
        hl, hr, ht, hb, kl, kt, kb = r
        assert kl >= hr, r                      # the figures sit to the right of the statement
        assert kt < hb and kb > ht, r           # on the same band
    finally:
        ctx.close()


def test_rankings_column_headers_meet_aa_contrast(browser):
    ctx, page, _ = _open(browser)
    try:
        pairs = page.evaluate("""() => [...document.querySelectorAll('#universe-table thead th')]
            .filter(th => th.offsetWidth && !th.classList.contains('sorted'))
            .map(th => [getComputedStyle(th).color, getComputedStyle(th).backgroundColor, th.textContent.trim()])""")
        assert len(pairs) >= 10
        low = [(t, round(_contrast(c, b), 2)) for c, b, t in pairs if _contrast(c, b) < 4.5]
        assert low == [], low
    finally:
        ctx.close()


def test_links_inside_sentences_are_underlined(browser):
    ctx, page, _ = _open(browser)
    try:
        page.wait_for_function("window.SCREENER_CONTEXT !== undefined", timeout=30000)
        page.evaluate("document.getElementById('score-key').open = true; toggleSection('sec-track')")
        page.wait_for_function("document.querySelector('.tr-caveats a') !== null", timeout=WAIT)
        deco = page.evaluate("""() => [...document.querySelectorAll('.tr-caveats a, .score-key .sk-foot a')]
            .map(a => [a.textContent.trim(), getComputedStyle(a).textDecorationLine])""")
        assert len(deco) >= 2, deco
        assert all("underline" in d for _, d in deco), deco
    finally:
        ctx.close()


@pytest.mark.parametrize("width", [375, 1280])
def test_methodology_formulas_wrap_instead_of_scrolling(browser, width):
    ctx, page, _ = _open(browser, width, 812)
    try:
        page.evaluate("openMethodology()")
        page.wait_for_function("document.querySelectorAll('#methodology-modal .methodology-body pre').length > 3", timeout=WAIT)
        over = page.evaluate("""() => [...document.querySelectorAll('#methodology-modal .methodology-body pre')]
            .filter(p => p.scrollWidth > p.clientWidth + 1).map(p => p.textContent.slice(0, 40))""")
        assert over == [], over
        # wrapping is visual only: the formula's text is unchanged, one line in the source
        txt = page.evaluate("""() => [...document.querySelectorAll('#methodology-modal pre')]
            .map(p => p.textContent).find(t => t.startsWith('Raw Composite ='))""")
        assert txt and "\n" not in txt.strip(), txt
    finally:
        ctx.close()


def test_scrolling_and_header_cells_have_names(browser):
    ctx, page, _ = _open(browser, 375, 812, touch=True)
    try:
        hm = page.evaluate("""() => { const h = document.getElementById('correlation-heatmap');
            return [h.getAttribute('role'), h.tabIndex, h.getAttribute('aria-label')]; }""")
        assert hm[0] == "region" and hm[1] == 0 and hm[2], hm
        page.evaluate("openStockDetail(D.table_data[0].Ticker)")
        page.wait_for_function("document.querySelector('#stock-modal .peer-th-action') !== null", timeout=WAIT)
        assert page.evaluate("document.querySelector('#stock-modal .peer-th-action').textContent.trim()") != ""
        page.evaluate("closeModal(); UX.compare = D.table_data.slice(0, 3).map(r => r.Ticker); openCompare()")
        page.wait_for_selector("#cmp-body .cmp-table", state="visible", timeout=WAIT)
        assert page.evaluate("document.querySelector('#cmp-body .cmp-table thead th').textContent.trim()") != ""
    finally:
        ctx.close()


# --- 2. the rankings window shifts instead of rebuilding ------------------------------------------

WINDOW_STATE = """() => {
    const tb = document.getElementById('universe-tbody');
    const kids = [...tb.children];
    const rows = kids.filter(r => r.dataset.t);
    const sp = kids.filter(r => r.classList.contains('vt-spacer'));
    const first = rows.length ? +rows[0].getAttribute('aria-rowindex') - 2 : -1;
    const H = rowHeight();
    return {
        tickers: rows.map(r => r.dataset.t),
        want: tableState.filtered.slice(first, first + rows.length).map(r => r.Ticker),
        idx: rows.map(r => +r.getAttribute('aria-rowindex')),
        first: first, n: rows.length, total: shownRows(), H: H,
        top: kids[0] && kids[0].classList.contains('vt-spacer') ? parseFloat(kids[0].firstElementChild.style.height) : 0,
        bot: kids.length && kids[kids.length - 1].classList.contains('vt-spacer') && kids.length > 1 ? parseFloat(kids[kids.length - 1].firstElementChild.style.height) : 0,
        spacers: sp.length, tabStops: rows.filter(r => r.tabIndex === 0).length,
    };
}"""


def _check_window(st):
    assert st["n"] > 0, st
    assert st["tickers"] == st["want"], "rows in the DOM are not the filtered list in order"
    assert st["idx"] == list(range(st["idx"][0], st["idx"][0] + st["n"])), "aria-rowindex not consecutive"
    assert abs(st["top"] - st["first"] * st["H"]) < 0.5, st
    assert abs(st["bot"] - (st["total"] - st["first"] - st["n"]) * st["H"]) < 0.5, st
    assert st["spacers"] <= 2 and st["tabStops"] == 1, st


@pytest.mark.parametrize("width,touch", [(375, True), (1280, False)])
def test_scrolling_the_rankings_keeps_rows_that_stay(browser, width, touch):
    ctx, page, errors = _open(browser, width, 812, touch=touch)
    try:
        page.evaluate("showMoreRows(true)")
        page.evaluate("document.getElementById('universe-table').scrollIntoView()")
        page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 5", timeout=WAIT)
        _check_window(page.evaluate(WINDOW_STATE))
        # mark every row on screen, scroll by about three rows, and wait for the window to move
        page.evaluate("""() => { window.__a0 = winRange.a; window.__b0 = winRange.b;
            document.querySelectorAll('#universe-tbody tr[data-t]').forEach(r => r.__kept = true);
            window.scrollBy(0, 3 * rowHeight() + 600); }""")
        page.wait_for_function("winRange.a !== window.__a0 || winRange.b !== window.__b0", timeout=WAIT)
        st = page.evaluate(WINDOW_STATE)
        _check_window(st)
        kept = page.evaluate("[...document.querySelectorAll('#universe-tbody tr[data-t]')].filter(r => r.__kept).length")
        assert kept > 0, "every row was rebuilt for a scroll of a few rows"
        # many steps down and back up: the window is always exactly the list
        for dy in (1700, 2500, -900, 4000, -6000, 120, -120):
            page.evaluate("dy => window.scrollBy(0, dy)", dy)
            page.wait_for_function("!winQueued", timeout=WAIT)
            page.evaluate("() => renderWindow(false)")
            _check_window(page.evaluate(WINDOW_STATE))
        # a sort still rebuilds from scratch and the window is still exact
        page.evaluate("sortTable('momentum_score')")
        _check_window(page.evaluate(WINDOW_STATE))
        assert errors == []
    finally:
        ctx.close()


def test_a_focused_row_stays_focused_while_the_window_shifts(browser):
    ctx, page, _ = _open(browser, 1280, 900)
    try:
        page.evaluate("document.getElementById('universe-table').scrollIntoView(); focusTableRow(60)")
        page.wait_for_function("document.activeElement && document.activeElement.dataset && document.activeElement.dataset.t", timeout=WAIT)
        t = page.evaluate("document.activeElement.dataset.t")
        page.evaluate("window.__a0 = winRange.a; window.scrollBy(0, 3 * rowHeight())")
        page.wait_for_function("winRange.a !== window.__a0", timeout=WAIT)
        assert page.evaluate("document.activeElement.dataset.t") == t
    finally:
        ctx.close()


# --- 3. the leftovers ----------------------------------------------------------------------------

def test_phone_sector_table_is_cards_with_every_figure_labelled(browser):
    ctx, page, _ = _open(browser, 375, 812, touch=True)
    try:
        page.evaluate("toggleSection('sec-analytics')")
        page.wait_for_function("document.querySelectorAll('#sector-matrix tbody tr').length > 5", timeout=WAIT)
        got = page.evaluate("""() => {
            const host = document.getElementById('sector-matrix');
            const rows = [...host.querySelectorAll('tbody tr')];
            return { scroll: host.scrollWidth - host.clientWidth,
                     rows: rows.map(tr => { const b = tr.getBoundingClientRect();
                        return { cells: [...tr.cells].map(td => { const r = td.getBoundingClientRect();
                                    return [td.dataset.l || '', getComputedStyle(td, '::before').content, r.left >= b.left - 1 && r.right <= b.right + 1 && r.width > 0]; }) }; }) };
        }""")
        assert got["scroll"] <= 1, "the sector table still scrolls sideways on a phone"
        for row in got["rows"]:
            assert len(row["cells"]) == 11, row
            assert all(inside for *_, inside in row["cells"]), row
            # every figure but the sector name carries its column's name
            assert all(lbl and content not in ("none", "normal") for lbl, content, _ in row["cells"][1:]), row
        # the same figures as the desktop table, in the same order
        phone = page.evaluate("[...document.querySelectorAll('#sector-matrix tbody tr')].map(tr => [...tr.cells].map(td => td.textContent.trim()))")
        page.set_viewport_size({"width": 1280, "height": 900})
        page.wait_for_function("getComputedStyle(document.querySelector('#sector-matrix thead')).display !== 'none'", timeout=WAIT)
        desk = page.evaluate("[...document.querySelectorAll('#sector-matrix tbody tr')].map(tr => [...tr.cells].map(td => td.textContent.trim()))")
        assert phone == desk
        heads = page.evaluate("[...document.querySelectorAll('#sector-matrix thead th')].map(th => th.textContent.trim())")
        labels = page.evaluate("[...document.querySelectorAll('#sector-matrix tbody tr:first-child td')].slice(1).map(td => td.dataset.l)")
        assert labels == heads[1:], (labels, heads)
    finally:
        ctx.close()


@pytest.mark.parametrize("width", [375, 1280])
def test_diagnostics_chevron_sits_on_its_title_line(browser, width):
    ctx, page, _ = _open(browser, width, 900)
    try:
        page.wait_for_function("document.querySelector('#defensibility-summary').children.length > 0", timeout=WAIT)
        for state in ("collapsed", "open"):
            if state == "open":
                page.evaluate("toggleSection('sec-defensibility')")
            r = page.evaluate("""() => { const t = document.querySelector('#sec-defensibility .section-title').getBoundingClientRect(),
                                          c = document.querySelector('#sec-defensibility > .section-header > .section-chevron').getBoundingClientRect();
                                          return [t.top, t.bottom, (c.top + c.bottom) / 2]; }""")
            assert r[0] <= r[2] <= r[1], (state, r)
    finally:
        ctx.close()


@pytest.mark.parametrize("width", [1024, 1280, 1440])
def test_rank_change_column_is_named_in_words_and_fits(browser, width):
    ctx, page, _ = _open(browser, width, 900)
    try:
        th = "document.getElementById('th-rank-delta')"
        assert page.evaluate(th + ".textContent.trim()") == "Move"
        assert page.evaluate(th + ".title").startswith("Change in rank")
        for sort in (False, True):
            if sort:
                page.evaluate("sortTable('_rank_delta')")
                page.wait_for_function(th + ".classList.contains('sorted')", timeout=WAIT)
            fit = page.evaluate("""() => { const th = document.getElementById('th-rank-delta');
                return [th.scrollWidth, th.clientWidth]; }""")
            assert fit[0] <= fit[1] + 1, (sort, fit)   # the word and its sort arrow fit the column
        assert "Move" in page.evaluate("document.querySelector('#sk-delta-row dt').textContent")
    finally:
        ctx.close()


# --- 4. the methodology's cross-references -------------------------------------------------------

def test_methodology_cross_references_land_on_their_heading(browser):
    """Generic over the text: every in-page link the methodology writes names a heading that exists,
    and following one scrolls the dialog to that heading and moves focus to it."""
    # Run with reduced motion so the jump is immediate: the smooth scroll's timing under a loaded
    # parallel suite is not what this test is about (two unexplained full-suite failures, 2026-10-27).
    # The smooth path is pinned separately below.
    ctx, page, _ = _open(browser, 375, 812, touch=True, reduced_motion="reduce")
    try:
        page.evaluate("openMethodology()")
        page.wait_for_function("document.querySelectorAll('#methodology-modal .methodology-body h2').length > 3", timeout=WAIT)
        links = page.evaluate("""() => {
            const body = document.querySelector('#methodology-modal .methodology-body');
            const heads = [...body.querySelectorAll('h1,h2,h3,h4,h5,h6')];
            return [...body.querySelectorAll('a[href^="#"]:not([data-m])')].map(a => {
                const f = decodeURIComponent(a.getAttribute('href').slice(1));
                return [f, heads.some(h => h.id === f || methodSlug(h.textContent) === methodSlug(f))]; });
        }""")
        assert links, "the methodology has no in-page links to test"
        assert all(ok for _, ok in links), [f for f, ok in links if not ok]
        frag = links[0][0]
        href_before = page.evaluate("location.hash")
        page.evaluate("""f => { const a = [...document.querySelectorAll('#methodology-modal .methodology-body a[href^="#"]:not([data-m])')]
                                 .find(x => decodeURIComponent(x.getAttribute('href').slice(1)) === f);
                                 a.scrollIntoView(); a.click(); }""", frag)
        page.wait_for_function("""f => { const h = document.activeElement;
            return h && /^H[1-6]$/.test(h.tagName) && methodSlug(h.textContent) === methodSlug(f)
                && Math.abs(h.getBoundingClientRect().top - document.querySelector('#methodology-modal .modal-body, #methodology-modal .methodology-body').getBoundingClientRect().top) < 160; }""",
                               arg=frag, timeout=WAIT)
        assert page.evaluate("location.hash") == href_before, "the link changed the address instead of moving"
        assert page.evaluate("document.getElementById('methodology-modal').style.display") == "flex"
    finally:
        ctx.close()


def test_a_cross_reference_scrolls_smoothly_unless_the_reader_asked_for_less_motion(browser):
    """The handler asks for a smooth scroll by default and an immediate one under
    prefers-reduced-motion; read from the call itself, so no animation timing is involved."""
    for motion, want in (("no-preference", "smooth"), ("reduce", "auto")):
        ctx, page, _ = _open(browser, 1280, 900, reduced_motion=motion)
        try:
            page.evaluate("openMethodology()")
            page.wait_for_function("document.querySelectorAll('#methodology-modal .methodology-body h2').length > 3", timeout=WAIT)
            got = page.evaluate("""() => {
                const calls = [];
                const orig = Element.prototype.scrollIntoView;
                Element.prototype.scrollIntoView = function (o) { calls.push(o && o.behavior); };
                const a = document.querySelector('#methodology-modal .methodology-body a[href^="#"]:not([data-m])');
                a.click();
                Element.prototype.scrollIntoView = orig;
                return calls; }""")
            assert got == [want], (motion, got)
        finally:
            ctx.close()


def test_slugs_match_the_usual_markdown_rule(browser):
    ctx, page, _ = _open(browser)
    try:
        got = page.evaluate("""() => [methodSlug('Bank-specific scoring'), methodSlug('  What it cannot do?  '),
                                      methodSlug("Words this page uses"), methodSlug('If you read nothing else')]""")
        assert got == ["bank-specific-scoring", "what-it-cannot-do", "words-this-page-uses", "if-you-read-nothing-else"]
    finally:
        ctx.close()
