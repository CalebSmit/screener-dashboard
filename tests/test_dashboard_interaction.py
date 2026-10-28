"""Ten functional defects in the public dashboard, fixed 2026-10-16, and the checks that keep them fixed.

None of these changes a number. Each was a way the page failed a reader who used the keyboard, a
saved list, a link, the Back button or a slow network:

1. The nine section headers were ``div``s with an ``onclick``: no tab stop, no Enter, no
   ``aria-expanded``. Collapsed bodies hid with ``max-height:0; opacity:0`` and stayed in the tab
   order - 33 invisible tab stops at load. The mover rows and the holdings search had no keyboard
   path either. (The holdings half is in ``tests/test_holdings_panel.py``.)
2. My Holdings erased saved names this run did not score - see ``tests/test_holdings_panel.py``.
3. The Google Fonts stylesheet was render-blocking: with that request stalled there was no first
   paint and the data script waited behind it.
4. The table's change column read the one-day change and coloured 466 of 501 rows, against a
   measured noise floor the rest of the page uses (one-month window, 37-rank threshold).
5. The Track record tile named only the benchmark the top 25 beat.
6. Each open-and-close of a stock left a dead Back press behind.
7. "/" focused the search box behind an open dialog, and Esc closed two dialogs at once.
8. With the context file blocked, the Context filter gave "0 of 501" with no reason, the
   earnings filter (which needs only the main payload) was dead, and Market and Track said the
   data was "not available for this run", which was false.
9. ``#stock=BRK.B`` and ``#stock=ZZZZ`` did nothing at all.
10. The footer disclaimer sat at 4.46:1 contrast.

The browser tests serve the published files over a local HTTP server (read-only), because request
interception does not apply to ``file://`` pages, and skip where Playwright or Chromium is absent.
"""

from __future__ import annotations

import functools
import http.server
import re
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402

PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"
SECTIONS = ["sec-top5", "sec-holdings", "sec-changed", "sec-reporting", "sec-market", "sec-track",
            "sec-analytics", "sec-defensibility", "sec-universe"]


@pytest.fixture(scope="module")
def html() -> str:
    return gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")


# ---------------------------------------------------------------------------
# markup the generator emits
# ---------------------------------------------------------------------------

def test_every_section_header_is_a_keyboard_button_that_names_its_body(html):
    for sec in SECTIONS:
        m = re.search(r'<div class="section-header"([^>]*)onclick="toggleSection\(\'%s\'\)">' % sec, html)
        assert m, sec
        attrs = m.group(1)
        assert 'role="button"' in attrs and 'tabindex="0"' in attrs, sec
        assert 'aria-controls="%s-body"' % sec in attrs, sec
        section_tag = html[html.rindex("<section", 0, m.start()):m.start()]
        collapsed = "collapsed" in section_tag.split(">")[0]
        assert 'aria-expanded="%s"' % ("false" if collapsed else "true") in attrs, sec
        assert html.count('id="%s-body"' % sec) == 1, sec
    # no section header is left without the button role
    assert html.count('<div class="section-header" onclick=') == 0


def test_a_closed_body_leaves_the_tab_order_but_still_prints():
    css = gd._css()
    closed = css[css.index(".collapsible-section.collapsed .section-body {"):]
    closed = closed[:closed.index("}")]
    assert "visibility: hidden" in closed
    # the visibility flip waits for the fade, so closing still animates
    assert "visibility 0s linear var(--t-base)" in closed
    printed = css[css.index("@media print"):]
    assert re.search(r"\.collapsible-section\.collapsed \.section-body \{[^}]*visibility: visible", printed)


def test_web_fonts_do_not_block_first_paint(html):
    head = html[:html.index("</head>")]
    outside_noscript = re.sub(r"<noscript>.*?</noscript>", "", head, flags=re.S)
    for tag in re.findall(r"<link[^>]+fonts\.googleapis\.com/css2[^>]*>", outside_noscript):
        assert 'rel="stylesheet"' not in tag, tag
    assert 'rel="preload"' in outside_noscript and "this.rel='stylesheet'" in outside_noscript
    assert re.search(r"<noscript><link[^>]+fonts\.googleapis\.com/css2[^>]+rel=\"stylesheet\"", head)
    # the fallback stack is still there for when the fonts never arrive: since round three
    # (2026-10-27) the metric-matched 'Inter Fallback' (local Arial cut to Inter's measure) first,
    # then the system stack for where Arial is not installed
    assert "--font-body: 'Inter', 'Inter Fallback', system-ui" in gd._css()


def test_new_copy_carries_no_advice_language(html):
    import stock_summary
    for phrase in ("is not in this run&rsquo;s ranking: it has left the S&amp;P 500 or its data could not be fetched.",
                   "the context data could not be loaded", "No stock "):
        assert phrase in html, phrase
        assert stock_summary.advice_terms_in(phrase) == []


# ---------------------------------------------------------------------------
# in a browser
# ---------------------------------------------------------------------------

sync_api = None
try:  # pragma: no cover - environment dependent
    from playwright import sync_api  # type: ignore
except Exception:  # noqa: BLE001
    pass

needs_browser = pytest.mark.skipif(sync_api is None or not PAGE.exists() or not PAYLOAD.exists(),
                                   reason="playwright or the generated dashboard is not present")


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # noqa: D401 - keep the test output clean
        pass


@pytest.fixture(scope="module")
def server():
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Quiet, directory=str(ROOT)))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield "http://127.0.0.1:%d/index.html" % httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _open(browser, url, width=1440, height=900, hash_="", routes=(), wait_until="load"):
    ctx = browser.new_context(viewport={"width": width, "height": height})
    for pattern, handler in routes:
        ctx.route(pattern, handler)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url + hash_, wait_until=wait_until, timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
    return ctx, page, errors


@needs_browser
def test_nothing_inside_a_closed_section_can_take_focus(browser, server):
    ctx, page, errors = _open(browser, server)
    try:
        page.wait_for_timeout(400)
        out = page.evaluate("""(() => {
            const sel = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea, [tabindex]:not([tabindex="-1"])';
            const inside = [...document.querySelectorAll('.collapsible-section.collapsed .section-body')]
                .flatMap(b => [...b.querySelectorAll(sel)]);
            const took = inside.filter(el => { el.focus(); return document.activeElement === el; });
            return { inside: inside.length, took: took.map(el => el.id || el.className || el.tagName) };
        })()""")
        assert out["inside"] > 0, "nothing to check: no focusable element inside a closed section"
        assert out["took"] == []
        # and every header is itself a tab stop
        assert page.evaluate("[...document.querySelectorAll('.collapsible-section > .section-header')].every(h => h.tabIndex === 0)")
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_enter_and_space_open_and_close_a_section_and_aria_follows(browser, server):
    ctx, page, errors = _open(browser, server)
    try:
        hdr = "#sec-analytics > .section-header"
        assert page.get_attribute(hdr, "aria-expanded") == "false"
        page.focus(hdr)
        page.keyboard.press("Enter")
        assert not page.evaluate("document.getElementById('sec-analytics').classList.contains('collapsed')")
        assert page.get_attribute(hdr, "aria-expanded") == "true"
        # the button's own transition starts from the hidden value, so it is focusable a frame later
        page.wait_for_function("document.getElementById('btn-mean').checkVisibility({ visibilityProperty: true })")
        page.focus("#btn-mean")
        assert page.evaluate("document.activeElement.id") == "btn-mean"   # the open body is reachable
        page.focus(hdr)
        page.keyboard.press(" ")
        assert page.evaluate("document.getElementById('sec-analytics').classList.contains('collapsed')")
        assert page.get_attribute(hdr, "aria-expanded") == "false"
        # a section opened by another path (the header nav) reports itself open too
        page.evaluate("goToSection('sec-defensibility')")
        page.wait_for_timeout(50)
        assert page.get_attribute("#sec-defensibility > .section-header", "aria-expanded") == "true"
        # printing shows a closed section's body
        page.emulate_media(media="print")
        assert page.evaluate("getComputedStyle(document.getElementById('sec-analytics-body')).visibility") == "visible"
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_a_mover_row_opens_its_stock_from_the_keyboard(browser, server):
    ctx, page, errors = _open(browser, server)
    try:
        if not page.evaluate("!!(H.available && H.movers)"):
            pytest.skip("payload has no comparable history")
        page.evaluate("goToSection('sec-changed')")
        page.wait_for_timeout(300)
        row = page.query_selector("#movers-up .mover-row, #movers-down .mover-row")
        assert row is not None and row.get_attribute("tabindex") == "0"
        t = row.query_selector(".mover-ticker").inner_text()
        row.focus()
        page.keyboard.press("Enter")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        assert page.inner_text("#modal-ticker") == t
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_the_holdings_box_adds_from_the_keyboard(browser, server):
    ctx, page, errors = _open(browser, server)
    try:
        page.evaluate("goToSection('sec-holdings')")
        page.wait_for_timeout(300)
        page.fill("#holdings-search-input", "jpm")
        page.keyboard.press("Enter")                        # the exact ticker typed
        assert page.evaluate("holdings.slice()") == ["JPM"]
        page.click("#holdings-search-input")
        page.keyboard.type("bank")
        page.wait_for_selector("#holdings-search-results .peer-search-item")
        names = page.evaluate("[...document.querySelectorAll('#holdings-search-results .peer-search-item .peer-search-ticker')].map(e => e.textContent)")
        assert len(names) >= 2
        page.keyboard.press("ArrowDown")
        page.keyboard.press("ArrowDown")
        assert page.evaluate("document.querySelector('#holdings-search-results .peer-search-item.sel .peer-search-ticker').textContent") == names[1]
        assert page.get_attribute("#holdings-search-input", "aria-activedescendant") == "hs-opt-1"
        page.keyboard.press("Enter")                        # the one picked with the arrows
        assert page.evaluate("holdings.slice()") == ["JPM", names[1]]
        page.click("#holdings-search-input")
        page.keyboard.type("apple")
        page.wait_for_selector("#holdings-search-results .peer-search-item")
        first = page.evaluate("document.querySelector('#holdings-search-results .peer-search-ticker').textContent")
        page.keyboard.press("Enter")                        # no exact ticker: the first result
        assert page.evaluate("holdings.slice()")[-1] == first
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_a_saved_name_outside_this_run_renders_and_is_kept(browser, server):
    """Row 5's constraint on the live page: a name that left the index stays on the list."""
    ctx, page, errors = _open(browser, server)
    try:
        page.evaluate("localStorage.setItem('screener_holdings_v1', JSON.stringify(['WBD', 'AAPL'])); location.reload()")
        page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
        if page.evaluate("!!D.stock_detail.WBD"):
            pytest.skip("WBD is in this run's universe")
        body = page.inner_text("#holdings-body")
        assert "WBD is not in this run’s ranking: it has left the S&P 500 or its data could not be fetched." in body
        page.evaluate("removeHolding('AAPL')")
        assert page.evaluate("JSON.parse(localStorage.getItem('screener_holdings_v1'))") == ["WBD"]
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_the_page_renders_while_the_font_request_hangs(browser, server):
    """A stalled Google Fonts request used to hold first paint and the data script with it."""
    stalled = []

    def hang(route):  # never answered: the request stays pending for the life of the context
        stalled.append(route.request.url)

    ctx, page, errors = _open(browser, server, routes=[("**/fonts.googleapis.com/**", hang)],
                              wait_until="domcontentloaded")
    try:
        assert stalled, "the font request was never made, so nothing was tested"
        assert page.evaluate("document.querySelectorAll('#universe-tbody tr[data-t]').length") > 10
        assert "Inter" in page.evaluate("getComputedStyle(document.body).fontFamily")  # the stack is unchanged
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_the_change_column_reads_the_one_month_window_and_colours_only_material_moves(browser, server):
    ctx, page, errors = _open(browser, server)
    try:
        if not page.evaluate("!!(H.available && H.compare && H.compare.m1)"):
            pytest.skip("payload has no one-month comparison")
        bad = page.evaluate("""(() => {
            const out = [], floor = H.noise.material_threshold;
            document.querySelectorAll('#universe-tbody tr[data-t]').forEach(tr => {
                const t = tr.dataset.t, cell = tr.querySelector('.delta-cell');
                const d = (H.delta[t] || {}).m1;
                const want = (!d || d.new || d.dr == null) ? null : d.dr;
                const txt = cell.textContent.replace(/[^0-9]/g, '');
                if (want === null) { if (!cell.classList.contains('muted')) out.push(t + ': expected a dash'); return; }
                if (want === 0) { if (txt !== '') out.push(t + ': expected blank'); return; }
                if (Number(txt) !== Math.abs(want)) out.push(t + ': ' + txt + ' vs m1 ' + want);
                const coloured = cell.classList.contains('pos') || cell.classList.contains('neg');
                if (Math.abs(want) < floor && coloured) out.push(t + ': coloured below the floor');
                if (Math.abs(want) >= floor && !cell.classList.contains(want > 0 ? 'pos' : 'neg')) out.push(t + ': material move not coloured');
            });
            return out;
        })()""")
        assert bad == []
        assert page.evaluate("document.querySelectorAll('#universe-tbody .delta-cell.sub').length") > 0
        title = page.get_attribute("#th-rank-delta", "title")
        assert page.evaluate("prettyDate(H.compare.m1.date)") in title
        assert str(page.evaluate("H.noise.material_threshold")) + " ranks" in title
        # sorting by the column uses the same figure
        assert page.evaluate("rankDelta(D.table_data[0].Ticker)") == page.evaluate(
            "((H.delta[D.table_data[0].Ticker] || {}).m1 || {}).dr ?? null")
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_the_track_record_tile_names_both_benchmarks(browser, server):
    ctx, page, errors = _open(browser, server)
    try:
        page.wait_for_function("window.SCREENER_CONTEXT !== undefined", timeout=30000)
        page.wait_for_timeout(200)
        if not page.evaluate("!!(D.track && D.track.available && D.track.total && D.track.total.SPY != null)"):
            pytest.skip("payload has no track record against SPY")
        f = page.evaluate("""(() => { const f = v => (v >= 0 ? '+' : '−') + Math.abs(v * 100).toFixed(1) + '%';
            return [f(D.track.total.RSP), f(D.track.total.SPY)]; })()""")
        sub = page.inner_text("#kpi-track-s")
        assert f[0] + " equal‑weight" in sub and f[1] + " S&P 500" in sub
        assert "-month record" in sub
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_open_and_close_leave_no_dead_back_presses(browser, server):
    ctx, page, errors = _open(browser, server)
    try:
        for t in ("JPM", "MSFT", "EXPE"):
            page.evaluate(f"openStockDetail('{t}')")
            page.wait_for_selector("#stock-modal .modal-body", state="visible")
            assert page.evaluate("location.hash") == "#stock=" + t
            page.keyboard.press("Escape")
            page.wait_for_function("location.hash === '' && !sheetOpen()")
        page.go_back()
        page.wait_for_timeout(300)
        assert not page.url.startswith(server)            # one Back leaves the page
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_a_direct_link_closes_without_leaving_the_page(browser, server):
    ctx, page, errors = _open(browser, server, hash_="#stock=MSFT")
    try:
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        page.keyboard.press("Escape")
        page.wait_for_timeout(200)
        assert page.url.startswith(server) and page.evaluate("location.hash") == ""
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_shortcuts_act_on_the_topmost_dialog_only(browser, server):
    ctx, page, errors = _open(browser, server)
    try:
        page.evaluate("openStockDetail('JPM')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        page.keyboard.press("/")
        assert page.evaluate("document.activeElement.id") != "filter-search"
        assert page.evaluate("sheetOpen()")
        page.keyboard.press("Escape")
        page.wait_for_function("!sheetOpen()")
        # Methodology with the sheet over it: Esc closes the sheet and leaves Methodology open
        page.evaluate("openMethodology()")
        page.evaluate("openStockDetail('MSFT')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        page.keyboard.press("Escape")
        page.wait_for_function("!sheetOpen()")
        assert page.evaluate("document.getElementById('methodology-modal').style.display") not in ("none", "")
        page.keyboard.press("Escape")
        assert page.evaluate("document.getElementById('methodology-modal').style.display") == "none"
        # with nothing open, "/" still goes to the search box
        page.keyboard.press("/")
        assert page.evaluate("document.activeElement.id") == "filter-search"
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_a_blocked_context_file_is_explained_and_the_earnings_filter_still_works(browser, server):
    ctx, page, errors = _open(browser, server, routes=[("**/dashboard_context.js*", lambda r: r.abort())])
    try:
        page.wait_for_function("CTX_FAILED === true", timeout=30000)
        opts = page.evaluate("[...document.getElementById('filter-ctx').options].map(o => [o.value, o.disabled, o.textContent])")
        by = {v: (d, t) for v, d, t in opts}
        for v in ("up", "down", "insider"):
            assert by[v][0] is True and "not loaded" in by[v][1], (v, by[v])
        assert by["earn14"][0] is False and by["all"][0] is False
        page.select_option("#filter-ctx", "earn14")
        page.wait_for_timeout(250)
        got = page.evaluate("tableState.filtered.map(r => r.Ticker).sort()")
        want = page.evaluate("""D.table_data.filter(r => { const d = daysUntil((D.stock_detail[r.Ticker].earn || {}).d);
                                 return d !== null && d >= 0 && d <= 14; }).map(r => r.Ticker).sort()""")
        assert got == want
        page.evaluate("goToSection('sec-market'); goToSection('sec-track')")
        for host in ("#market-body", "#track-body"):
            text = page.inner_text(host)
            assert "the context data could not be loaded" in text, (host, text)
            assert "not available for this run" not in text
        assert "not loaded" in page.inner_text("#kpi-track")
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_the_earnings_filter_works_before_the_context_file_lands(browser, server):
    hold = []
    # the held request keeps the load event from firing, so wait for the DOM and the table instead
    ctx, page, errors = _open(browser, server, routes=[("**/dashboard_context.js*", lambda r: hold.append(r))],
                              wait_until="domcontentloaded")
    try:
        assert page.evaluate("CTX_LOADED") is False
        assert page.evaluate("document.querySelector('#filter-ctx option[value=up]').disabled") is True
        n = page.evaluate("tableState.filtered.length")
        page.select_option("#filter-ctx", "earn14")
        page.wait_for_timeout(250)
        assert 0 < page.evaluate("tableState.filtered.length") < n
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_deep_links_accept_a_dot_and_name_a_miss(browser, server):
    ctx, page, errors = _open(browser, server, hash_="#stock=BRK.B")
    try:
        if not page.evaluate("!!D.stock_detail['BRK-B']"):
            pytest.skip("BRK-B is not in this run")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        assert page.inner_text("#modal-ticker") == "BRK-B"
        assert errors == []
    finally:
        ctx.close()
    ctx, page, errors = _open(browser, server, hash_="#stock=ZZZZ")
    try:
        page.wait_for_timeout(200)
        assert not page.evaluate("sheetOpen()")
        assert page.inner_text("#toast") == "No stock ZZZZ in this run"
        assert page.evaluate("location.hash") == ""
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_the_footer_disclaimer_meets_the_contrast_floor(browser, server):
    ctx, page, errors = _open(browser, server)
    try:
        ratio = page.evaluate("""(() => {
            const el = document.querySelector('.dashboard-footer .footer-disclaimer');
            const parse = c => c.match(/[\\d.]+/g).map(Number);
            let op = 1, n = el;
            while (n && n.nodeType === 1) { op *= parseFloat(getComputedStyle(n).opacity); n = n.parentElement; }
            let bg = [0, 0, 0]; n = el;
            while (n && n.nodeType === 1) { const c = parse(getComputedStyle(n).backgroundColor); if (c.length < 4 || c[3] > 0) { bg = c; break; } n = n.parentElement; }
            const fg = parse(getComputedStyle(el).color);
            const a = (fg[3] === undefined ? 1 : fg[3]) * op;
            const mix = [0, 1, 2].map(i => fg[i] * a + bg[i] * (1 - a));
            const L = c => { const s = c.map(v => { v /= 255; return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4); });
                             return 0.2126 * s[0] + 0.7152 * s[1] + 0.0722 * s[2]; };
            const l1 = L(mix), l2 = L(bg.slice(0, 3));
            return (Math.max(l1, l2) + 0.05) / (Math.min(l1, l2) + 0.05);
        })()""")
        assert ratio >= 4.5, ratio
        assert errors == []
    finally:
        ctx.close()
