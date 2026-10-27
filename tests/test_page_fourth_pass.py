"""The dashboard page's fourth UI pass (2026-10-27, owner-run; ``plan/ui-elevation.md``), in a real browser.

Every check opens the committed page (``tests/test_rebuild_page.py`` holds it equal to the generator's
output) and asserts a behaviour, not a class name:

1. **Reporting soon on a phone** hid its "Last quarter" and "Insiders" columns with nothing in their
   place. The same two figures now ride under the company name below 640px.
2. **The phone Weighting select** cut "Balanced (published)" to "Balanced (publisl".
3. **Side by side on a phone** put the third stock off the right edge, in a sideways scroll.
4. **The dot picture** could not be reached from the keyboard.
5. **The phone rankings** rendered all ~500 cards: a ~46,000px page with the footer at the end of it.
   They now show 50 at a time, with nothing withheld from the filters, search or deep links.
6. **The rankings table** was ~500 tab stops between the filters and the footer; it is one (roving).
7. **Tablet widths**: at 768px the company column measured 0px and Inv and the flags were cut off;
   below 1180px a sixteenth, empty column took the company name's width.
8. **A trap flag on a phone card** took half the company name's width ("Delta Air ...").
9. **The phone Menu pill** sat over a card's composite on every screen.
10. **The Track record tile** showed an 8-month return with no word on how short that is; it now says
    so in the panel's own figure (claim ``context.track_luck``).
11. **My holdings' remove button** was a 15x17px "x" read aloud as "times".

Skips loudly where Playwright or a browser is not installed, like the other browser modules. Waits
are conditions, never fixed sleeps (fixed sleeps flaked under the parallel suite on 2026-10-27).
"""

from __future__ import annotations

from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"

pytestmark = pytest.mark.skipif(not PAGE.exists() or not PAYLOAD.exists(),
                                reason="generated dashboard not present")

READY = "document.querySelectorAll('#universe-tbody tr[data-t]').length > 0"


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _open(browser, width=1440, height=900, touch=False, hash_="", holdings=None):
    opts = {"viewport": {"width": width, "height": height}}
    if touch:
        opts.update(is_mobile=True, has_touch=True)
    ctx = browser.new_context(**opts)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    if holdings is not None:
        page.add_init_script("try { localStorage.setItem('screener_holdings_v1', JSON.stringify(%s)); } catch (e) {}"
                             % repr(list(holdings)).replace("'", '"'))
    page.goto(PAGE.as_uri() + hash_, wait_until="load", timeout=60000)
    page.wait_for_function(READY, timeout=30000)
    return ctx, page, errors


def _ctx_loaded(page):
    page.wait_for_function("window.SCREENER_CONTEXT !== undefined", timeout=30000)


# --- 1. Reporting soon on a phone ------------------------------------------------------------------

def test_phone_reporting_soon_keeps_last_quarter_and_insiders(browser):
    ctx, page, errors = _open(browser, 375, 812)
    try:
        _ctx_loaded(page)
        page.evaluate("goToSection('sec-reporting'); setReporting('days', 14)")
        page.wait_for_function("document.querySelectorAll('#reporting-body tr.rp-row').length > 0", timeout=8000)
        rows = page.evaluate("""() => [...document.querySelectorAll('#reporting-body tr.rp-row')].map(tr => {
            const td = tr.querySelectorAll('td'), m = tr.querySelector('.rp-m');
            const vis = el => !!el && getComputedStyle(el).display !== 'none' && el.getBoundingClientRect().height > 0;
            return { lq: td[4].textContent.trim(), ins: td[5].textContent.trim(),
                     lqShown: vis(td[4]), insShown: vis(td[5]),
                     m: m ? m.textContent : '', mShown: vis(m) }; })""")
        assert rows
        with_figures = [r for r in rows if r["lq"] not in ("", "–") or r["ins"]]
        assert with_figures, "no reporting row carries a last-quarter or insider figure"
        for r in rows:
            assert not r["lqShown"] and not r["insShown"], r          # the columns are still off at 375
            if r["lq"] not in ("", "–"):
                assert r["mShown"] and r["lq"] in r["m"], r               # ...but the figure is there
            if r["ins"]:
                assert r["mShown"] and r["ins"] in r["m"], r
        assert page.evaluate("document.documentElement.scrollWidth") <= 376
        assert errors == []
    finally:
        ctx.close()


def test_desktop_reporting_soon_shows_the_columns_not_the_phone_line(browser):
    ctx, page, errors = _open(browser)
    try:
        _ctx_loaded(page)
        page.evaluate("goToSection('sec-reporting')")
        page.wait_for_function("document.querySelectorAll('#reporting-body tr.rp-row').length > 0", timeout=8000)
        shown = page.evaluate("""() => [...document.querySelectorAll('#reporting-body .rp-m')]
            .filter(m => getComputedStyle(m).display !== 'none').length""")
        assert shown == 0
        assert errors == []
    finally:
        ctx.close()


# --- 2. The phone Weighting select ------------------------------------------------------------------

def test_phone_weighting_select_shows_its_whole_label(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        fit = page.evaluate("""() => {
            const s = document.getElementById('filter-profile'), cs = getComputedStyle(s);
            const c = document.createElement('canvas').getContext('2d');
            c.font = cs.fontWeight + ' ' + cs.fontSize + ' ' + cs.fontFamily;
            const need = Math.max(...[...s.options].map(o => c.measureText(o.text).width));
            const room = s.clientWidth - parseFloat(cs.paddingLeft) - parseFloat(cs.paddingRight);
            return { need: need, room: room, right: s.getBoundingClientRect().right }; }""")
        assert fit["need"] <= fit["room"], fit
        assert fit["right"] <= 375, fit
        assert errors == []
    finally:
        ctx.close()


# --- 3. Side by side on a phone ---------------------------------------------------------------------

@pytest.mark.parametrize("n", [3, 4])
def test_phone_side_by_side_fits_three_and_four_stocks(browser, n):
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        ts = page.evaluate(f"D.table_data.slice(0, {n}).map(r => r.Ticker)")
        page.evaluate(f"UX.compare = {ts!r}; openCompare()".replace("'", '"'))
        page.wait_for_selector(".cmp-scroll", state="visible", timeout=8000)
        m = page.evaluate("""() => { const s = document.querySelector('.cmp-scroll'), b = document.querySelector('#compare-modal .cmp-content').getBoundingClientRect();
            const cells = [...document.querySelectorAll('.cmp-table td, .cmp-table thead th')].filter(e => e.offsetWidth);
            const out = cells.filter(e => { const r = e.getBoundingClientRect(); return r.right > b.right + 1 || r.left < b.left - 1; }).length;
            const heads = [...document.querySelectorAll('.cmp-table thead .cmp-t')].map(e => e.textContent.trim());
            return { sw: s.scrollWidth, cw: s.clientWidth, doc: document.documentElement.scrollWidth, out: out, heads: heads }; }""")
        assert m["sw"] <= m["cw"] + 1, m
        assert m["doc"] <= 376 and m["out"] == 0, m
        assert sorted(m["heads"]) == sorted(ts), m
        assert errors == []
    finally:
        page.evaluate("clearCompare()")
        ctx.close()


# --- 4. The dot picture from the keyboard -----------------------------------------------------------

def test_the_dot_picture_is_operable_from_the_keyboard(browser):
    ctx, page, errors = _open(browser)
    try:
        page.wait_for_function("typeof DIST !== 'undefined' && DIST.order && DIST.order.length > 0", timeout=8000)
        attrs = page.evaluate("""() => { const h = document.getElementById('dist-plot');
            return { tab: h.tabIndex, role: h.getAttribute('role'), max: h.getAttribute('aria-valuemax'), n: DIST.order.length }; }""")
        assert attrs["tab"] == 0 and attrs["role"] == "slider" and attrs["max"] == str(attrs["n"]), attrs
        # reach it with Tab, the way a keyboard reader does
        page.focus("#dist-plot")
        page.keyboard.press("Shift+Tab")
        page.keyboard.press("Tab")
        page.wait_for_function("document.activeElement && document.activeElement.id === 'dist-plot'", timeout=8000)
        page.keyboard.press("End")
        top = page.evaluate("""() => { const x = DIST.order[DIST.order.length - 1];
            return { t: x.Ticker, c: x.Composite.toFixed(1), r: x.Rank, best: Math.max(...DIST.order.map(o => o.Composite)) === x.Composite,
                     vt: document.getElementById('dist-plot').getAttribute('aria-valuetext'), tip: DIST.tip.hidden ? '' : DIST.tip.textContent }; }""")
        assert top["best"], top
        assert top["t"] in top["vt"] and f"rank {top['r']}" in top["vt"] and top["c"] in top["vt"], top
        assert top["t"] in top["tip"], top                                 # the tooltip follows the keyboard
        page.keyboard.press("ArrowLeft")
        nxt = page.evaluate("DIST.order[DIST.order.length - 2].Ticker")
        assert nxt in page.get_attribute("#dist-plot", "aria-valuetext")
        page.keyboard.press("Enter")
        page.wait_for_selector("#stock-modal .modal-body", state="visible", timeout=8000)
        assert nxt in page.inner_text("#modal-ticker")
        assert errors == []
    finally:
        ctx.close()


# --- 5. The phone rankings, fifty at a time ----------------------------------------------------------

def test_phone_rankings_show_fifty_then_more_then_all(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        n = page.evaluate("D.table_data.length")
        state = """() => { const t = document.getElementById('universe-table'), b = document.getElementById('table-more');
            return { rows: Number(t.getAttribute('aria-rowcount')) - 1, more: !b.hidden,
                     count: b.querySelector('.tm-count').textContent, h: t.getBoundingClientRect().height,
                     doc: document.documentElement.scrollHeight }; }"""
        s0 = page.evaluate(state)
        assert s0["rows"] == 50 and s0["more"] and s0["count"] == f"Showing 50 of {n}", s0
        assert s0["doc"] < 20000, s0                                       # was ~46,000px
        page.click("#table-more .tm-next")
        page.wait_for_function("Number(document.getElementById('universe-table').getAttribute('aria-rowcount')) === 101", timeout=8000)
        page.click("#table-more .tm-all")
        page.wait_for_function(f"Number(document.getElementById('universe-table').getAttribute('aria-rowcount')) === {n + 1}", timeout=8000)
        assert page.evaluate("document.getElementById('table-more').hidden")
        # a new filter starts again from the top, and still reaches past the first fifty
        last = page.evaluate("D.table_data.slice().sort((a, b) => b.Rank - a.Rank)[0]")
        page.fill("#filter-search", last["Ticker"])
        page.wait_for_function(f"tableState.filtered.some(r => r.Ticker === {last['Ticker']!r})".replace("'", '"'), timeout=8000)
        page.wait_for_function(f"!!document.querySelector('#universe-tbody tr[data-t=\"{last['Ticker']}\"]')", timeout=8000)
        page.fill("#filter-search", "")
        page.wait_for_function("Number(document.getElementById('universe-table').getAttribute('aria-rowcount')) === 51", timeout=8000)
        assert errors == []
    finally:
        ctx.close()


def test_phone_deep_link_opens_a_stock_beyond_the_first_fifty(browser):
    ctx0, page0, _ = _open(browser, 375, 812)
    t = page0.evaluate("D.table_data.find(r => r.Rank === 300 || r.Rank > 250).Ticker")
    ctx0.close()
    ctx, page, errors = _open(browser, 375, 812, touch=True, hash_=f"#stock={t}")
    try:
        page.wait_for_selector("#stock-modal .modal-body", state="visible", timeout=8000)
        assert t in page.inner_text("#modal-ticker")
        assert errors == []
    finally:
        ctx.close()


def test_desktop_rankings_are_not_capped(browser):
    ctx, page, errors = _open(browser)
    try:
        n = page.evaluate("D.table_data.length")
        assert page.evaluate("Number(document.getElementById('universe-table').getAttribute('aria-rowcount'))") == n + 1
        assert page.evaluate("document.getElementById('table-more').hidden")
        assert errors == []
    finally:
        ctx.close()


# --- 6. One tab stop for the table ------------------------------------------------------------------

def test_the_rankings_table_is_one_tab_stop_with_arrow_keys_inside(browser):
    ctx, page, errors = _open(browser)
    try:
        zero = "() => [...document.querySelectorAll('#universe-tbody tr[data-t]')].filter(r => r.tabIndex === 0).map(r => r.dataset.t)"
        assert page.evaluate(zero) == [page.evaluate("tableState.filtered[0].Ticker")]
        page.focus("#universe-tbody tr[data-t]")
        for _ in range(3):
            page.keyboard.press("ArrowDown")
        want = page.evaluate("tableState.filtered[3].Ticker")
        assert page.evaluate("document.activeElement.dataset.t") == want
        assert page.evaluate(zero) == [want]                               # the stop follows the reader
        page.keyboard.press("Tab")
        assert not page.evaluate("document.getElementById('universe-tbody').contains(document.activeElement)")
        page.keyboard.press("Shift+Tab")
        # Shift+Tab comes back into the list: to the remembered row when it is still rendered, otherwise to
        # the first row of the window on screen (the page scrolled to the footer, so the window moved)
        back = page.evaluate("document.activeElement.dataset ? document.activeElement.dataset.t : null")
        assert back is not None and page.evaluate("document.getElementById('universe-tbody').contains(document.activeElement)"), back
        assert page.get_attribute("#universe-table", "aria-describedby") == "universe-keys"
        assert errors == []
    finally:
        ctx.close()


# --- 7. Tablet and laptop widths -----------------------------------------------------------------------

@pytest.mark.parametrize("width", [768, 820, 1024, 1180, 1280])
def test_every_column_fits_and_the_spacers_span_only_shown_columns(browser, width):
    ctx, page, errors = _open(browser, width, 900)
    try:
        page.evaluate("document.fonts ? document.fonts.ready : null")
        m = page.evaluate("""() => { const t = document.getElementById('universe-table'), box = document.querySelector('.table-section').getBoundingClientRect();
            const th = [...t.tHead.rows[0].cells].filter(c => getComputedStyle(c).display !== 'none');
            const cut = th.filter(c => c.getBoundingClientRect().right > box.right + 1).map(c => c.textContent.trim());
            const sp = document.querySelector('#universe-tbody tr.vt-spacer td');
            const lastRow = document.querySelector('#universe-tbody tr[data-t]').getBoundingClientRect();
            const co = t.tHead.rows[0].cells[3];
            return { cut: cut, shown: th.length, span: sp ? sp.colSpan : th.length, rowRight: lastRow.right, boxRight: box.right,
                     company: getComputedStyle(co).display === 'none' ? null : co.offsetWidth }; }""")
        assert m["cut"] == [], m
        assert m["span"] == m["shown"], m
        assert m["rowRight"] >= m["boxRight"] - 3, m                       # the rows fill the box: no phantom column
        if m["company"] is not None:
            assert m["company"] >= 140, m                                  # a name, not "Valero Energy Corpo..."
        assert errors == []
    finally:
        ctx.close()


def test_the_filters_do_not_stack_in_a_column_at_768(browser):
    ctx, page, errors = _open(browser, 768, 1024)
    try:
        h = page.evaluate("document.querySelector('.filters-bar').getBoundingClientRect().height")
        assert h < 260, h                                                   # was ~600px, one control per row
        assert errors == []
    finally:
        ctx.close()


# --- 8. A trap flag on a phone card ---------------------------------------------------------------------

def test_phone_flag_sits_under_the_name_and_both_flags_fit(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        page.select_option("#filter-vt", "any")
        page.wait_for_function("document.querySelector('#universe-tbody .vt-cell .flag') !== null", timeout=8000)
        page.evaluate("""() => { tableState.filtered.sort((a, b) => ((b.Value_Trap_Flag && b.Growth_Trap_Flag) ? 1 : 0) - ((a.Value_Trap_Flag && a.Growth_Trap_Flag) ? 1 : 0));
                                 renderWindow(true); }""")
        bad = page.evaluate("""() => [...document.querySelectorAll('#universe-tbody tr[data-t]')].slice(0, 12).map(tr => {
            const co = tr.querySelector('td.company').getBoundingClientRect(), vt = tr.querySelector('td.vt-cell');
            const flags = [...vt.querySelectorAll('.flag')].map(f => f.getBoundingClientRect());
            const vb = vt.getBoundingClientRect();
            const clipped = flags.some(f => f.right > vb.right + 0.5);
            const under = flags.every(f => f.top >= co.top + 14);
            return { t: tr.dataset.t, w: Math.round(co.width), clipped: clipped, under: under, n: flags.length }; })
            .filter(x => x.n && (x.w < 120 || x.clipped || !x.under))""")
        assert bad == [], bad
        assert errors == []
    finally:
        ctx.close()


# --- 9. The phone Menu pill ---------------------------------------------------------------------------

def test_phone_menu_steps_aside_while_scrolling_down_and_returns_on_the_way_up(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        page.evaluate("window.scrollTo(0, 3000)")
        page.wait_for_function("!document.getElementById('m-nav-btn').hidden", timeout=8000)
        page.evaluate("window.scrollTo(0, 4200)")
        page.wait_for_function("getComputedStyle(document.getElementById('m-nav-btn')).visibility === 'hidden'", timeout=8000)
        page.evaluate("window.scrollTo(0, 3600)")
        page.wait_for_function("getComputedStyle(document.getElementById('m-nav-btn')).visibility === 'visible'", timeout=8000)
        # at the end of the page it is there, and it covers no footer link
        page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
        page.wait_for_function("getComputedStyle(document.getElementById('m-nav-btn')).visibility === 'visible' && !document.getElementById('m-nav-btn').classList.contains('tuck')", timeout=8000)
        over = page.evaluate("""() => { const b = document.getElementById('m-nav-btn').getBoundingClientRect();
            return [...document.querySelectorAll('.dashboard-footer a, .dashboard-footer button')].filter(a => {
                const r = a.getBoundingClientRect(); return r.width && r.right > b.left && r.left < b.right && r.bottom > b.top && r.top < b.bottom; })
                .map(a => a.textContent.trim()); }""")
        assert over == [], over
        assert errors == []
    finally:
        ctx.close()


# --- 10. The Track record tile ------------------------------------------------------------------------

def test_the_track_record_tile_states_the_record_is_short_in_the_panels_figure(browser):
    ctx, page, errors = _open(browser)
    try:
        _ctx_loaded(page)
        x = page.evaluate("(D.track && D.track.excess) ? D.track.excess : null")
        if not x or x.get("t") is None:
            pytest.skip("the published track record carries no excess t")
        page.wait_for_function("document.getElementById('kpi-track-v').textContent.trim() !== '…'", timeout=8000)
        cav = page.evaluate("(document.querySelector('#kpi-track .kpi-caveat') || {}).textContent || ''")
        if abs(x["t"]) < 2:
            assert "skill from luck" in cav and f"{abs(x['t']):.1f} standard errors" in cav, cav
            # the panel prints the same figure
            page.evaluate("goToSection('sec-track')")
            page.wait_for_function("document.getElementById('track-body') && document.getElementById('track-body').textContent.includes('standard errors')", timeout=8000)
            panel = page.inner_text("#track-body")
            assert f"{x['t']:.1f} standard errors" in panel
        else:
            assert cav == ""
        assert errors == []
    finally:
        ctx.close()


# --- 11. My holdings' remove button -------------------------------------------------------------------

def test_holdings_remove_buttons_are_named_and_thumb_sized(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True, holdings=["AAPL", "JPM"])
    try:
        page.evaluate("goToSection('sec-holdings')")
        page.wait_for_function("document.querySelectorAll('#holdings-body .holding-remove').length === 2", timeout=8000)
        b = page.evaluate("""() => [...document.querySelectorAll('#holdings-body .holding-remove')].map(x => {
            const r = x.getBoundingClientRect(); return { l: x.getAttribute('aria-label'), w: r.width, h: r.height }; })""")
        assert [x["l"] for x in b] == ["Remove AAPL from your list", "Remove JPM from your list"] or \
               sorted(x["l"] for x in b) == ["Remove AAPL from your list", "Remove JPM from your list"], b
        assert all(x["w"] >= 44 and x["h"] >= 44 for x in b), b
        assert errors == []
    finally:
        ctx.close()
