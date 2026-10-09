"""The navigation layer: search palette, stock links, stepping, compare.

WHY THIS EXISTS - 2026-10-07 (owner-run UI pass 2, ``plan/dashboard-redesign-master.md``).

The owner asked for the page to be better to move around in. A 502-stock tool had one way to
reach a stock - scroll to the table, find it, click - and no way to point anyone else at one.
This pass added:

* a search palette (Ctrl/Cmd+K) that reaches any stock or section from anywhere;
* ``#stock=TICKER`` links that open a stock, with the browser's back closing the sheet;
* J / K to step through the table's current order without closing the drilldown;
* a side-by-side comparison that takes the composite gap between two stocks apart into the
  category points that make it.

The last one says how a number is computed, so it is held to the same rule as every other such
sentence on the page (``claims.py``, ``compare.composite_gap``): the payload must make it true.
Those checks live in ``test_calculation_reproducibility.py``, which the data loop's publish gate
runs; this module holds the markup and browser checks, which it does not.

The browser tests skip where Playwright or Chromium is absent, like ``test_dashboard_browser``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
import stock_summary  # noqa: E402

PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"
CATS = ["valuation", "quality", "growth", "momentum", "risk", "revisions", "size", "investment"]


@pytest.fixture(scope="module")
def html() -> str:
    return gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")


# ---------------------------------------------------------------------------
# markup the generator emits
# ---------------------------------------------------------------------------

def test_the_drilldown_reads_in_the_order_its_jump_links_promise(html):
    """The first version's jump links said Why / Scores / How it adds up / The workings / Price
    targets, and the body ran Why / Scores / About / History / Price targets / ... / How it adds
    up / The workings - so 'How it adds up' jumped past four sections. Reading order is the nav."""
    nav = html[html.index('class="modal-nav"'):]
    nav = nav[:nav.index("</nav>")]
    targets = re.findall(r'goToModal\(\'([a-z-]+)\'\)', nav)
    assert targets[:4] == ["modal-summary", "modal-score-row", "section-contribution", "section-categories"]
    body = html[html.index('<div class="modal-body">'):html.index("<!-- Methodology Modal -->")]
    positions = [body.index('id="%s"' % t) for t in targets]
    assert positions == sorted(positions), list(zip(targets, positions))


def test_section_titles_match_the_jump_links(html):
    assert "<span>How it adds up</span>" in html
    assert "<span>The workings</span>" in html


def test_the_navigation_layer_wraps_rather_than_forks_the_drilldown(html):
    """Everything that opens a stock (rows, Top 5, movers, holdings, peers, the palette) must get
    links and stepping, which only holds if the one opener is wrapped, not copied."""
    js = gd._js_ux()
    assert "const _baseOpenStockDetail = openStockDetail;" in js
    assert "openStockDetail = function(" in js
    assert "const _baseCloseModal = closeModal;" in js
    assert html.count("function openStockDetail(") == 1


def test_overlays_and_controls_are_present(html):
    for marker in ('id="palette"', 'id="pal-input"', 'id="compare-modal"', 'id="cmp-tray"',
                   'id="shortcuts"', 'id="mt-prev"', 'id="mt-next"', 'id="mt-compare"',
                   'id="mt-hold"', 'id="mt-link"', 'class="cmdk-btn"', 'id="guide"'):
        assert marker in html, marker
    assert "initUX();" in html


def test_the_trap_chart_is_html_not_a_clipping_canvas(html):
    assert 'id="vt-chart"' not in html
    assert 'id="trap-bars"' in html


def test_new_copy_carries_no_advice_language(html):
    """The guide, compare view, toolbar and shortcuts are prose a student reads first."""
    def text_of(fragment: str) -> str:
        return re.sub(r"<[^>]+>", " ", fragment)

    guide = html[html.index('<section class="guide"'):html.index("<!-- Top 5 Stocks -->")]
    overlays = html[html.index("<!-- Search palette"):html.index('<footer class="dashboard-footer">')]
    js_strings = " ".join(re.findall(r"'([^'\n]{12,})'", gd._js_ux()))
    for name, chunk in (("guide", text_of(guide)), ("overlays", text_of(overlays)), ("js", js_strings)):
        assert stock_summary.advice_terms_in(chunk) == [], (name, stock_summary.advice_terms_in(chunk))


def test_storage_is_guarded_everywhere_it_is_touched():
    """Per-viewer conveniences only, and the page must work with storage blocked."""
    js = gd._js_ux()
    for m in re.finditer(r"localStorage\.(getItem|setItem)", js):
        before = js[max(0, m.start() - 60):m.start()]
        assert "try {" in before, js[m.start() - 80:m.end() + 20]


def test_the_claim_is_registered():
    import claims
    ids = {c.id for c in claims.CLAIMS}
    assert "compare.composite_gap" in ids


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


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _open(browser, width=1440, height=900, hash_=""):
    ctx = browser.new_context(viewport={"width": width, "height": height})
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri() + hash_, wait_until="load", timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
    return ctx, page, errors


@needs_browser
def test_ctrl_k_finds_a_stock_and_opens_it_with_a_link(browser):
    ctx, page, errors = _open(browser)
    try:
        page.keyboard.press("Control+k")
        assert page.is_visible("#pal-input")
        page.keyboard.type("jpm")
        first = page.inner_text("#pal-list .pal-item.sel .pal-t")
        assert first == "JPM"
        page.keyboard.press("Enter")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        assert page.inner_text("#modal-ticker") == "JPM"
        assert page.evaluate("location.hash") == "#stock=JPM"
        assert not page.is_visible("#pal-input")
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_a_company_name_finds_its_ticker(browser):
    ctx, page, _ = _open(browser)
    try:
        page.evaluate("openPalette()")
        page.keyboard.type("expedia")
        assert page.inner_text("#pal-list .pal-item.sel .pal-t") == "EXPE"
        page.keyboard.press("Escape")
        assert not page.is_visible("#pal-input")
    finally:
        ctx.close()


@needs_browser
def test_a_stock_link_opens_that_stock_on_load(browser):
    ctx, page, errors = _open(browser, hash_="#stock=MSFT")
    try:
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        assert page.inner_text("#modal-ticker") == "MSFT"
        page.keyboard.press("Escape")
        assert page.evaluate("location.hash") == ""
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_back_closes_the_sheet_instead_of_leaving(browser):
    ctx, page, _ = _open(browser)
    try:
        page.evaluate("openStockDetail('EXPE')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        page.go_back()
        page.wait_for_timeout(200)
        assert not page.evaluate("sheetOpen()")
        assert page.url.startswith(PAGE.as_uri())  # still on the page
    finally:
        ctx.close()


@needs_browser
def test_j_and_k_step_through_the_tables_current_order(browser):
    ctx, page, _ = _open(browser)
    try:
        page.select_option("#filter-sector", "Health Care")
        page.wait_for_timeout(250)
        order = page.evaluate("tableState.filtered.map(r => r.Ticker)")
        page.evaluate(f"openFromRow('{order[0]}')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        page.keyboard.press("j")
        page.keyboard.press("j")
        assert page.evaluate("UX.current") == order[2]
        assert page.inner_text("#mt-pos").startswith("3 of %d" % len(order))
        page.keyboard.press("k")
        assert page.evaluate("UX.current") == order[1]
        assert page.evaluate("location.hash") == "#stock=" + order[1]
        page.go_back()  # stepping replaced, did not push: one back closes the sheet
        page.wait_for_timeout(200)
        assert not page.evaluate("sheetOpen()")
    finally:
        ctx.close()


@needs_browser
def test_compare_lines_up_scores_and_reconciles_the_gap(browser):
    ctx, page, errors = _open(browser)
    try:
        page.evaluate("openStockDetail('EXPE')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        page.keyboard.press("c")
        page.keyboard.press("j")
        page.keyboard.press("c")
        page.keyboard.press("Escape")
        assert page.is_visible("#cmp-tray")
        page.click("#cmp-open")
        page.wait_for_selector("#cmp-body .cmp-table", state="visible")
        assert page.evaluate("document.querySelectorAll('#cmp-body .cmp-table thead th').length") == 3
        lines = page.evaluate("""() => [...document.querySelectorAll('#cmp-body .gap-block')[0].querySelectorAll('.gap-row:not(.gap-total) .num')]
            .map(e => parseFloat(e.textContent.replace('\\u2212', '-')))""")
        total = page.evaluate("""() => parseFloat(document.querySelector('#cmp-body .gap-total .num').textContent.replace('\\u2212', '-'))""")
        assert len(lines) >= 8
        assert abs(sum(lines) - total) < 0.1, (lines, total)
        assert not page.is_visible("#cmp-tray")  # the tray never covers the comparison
        page.keyboard.press("Escape")
        assert not page.is_visible("#compare-modal .cmp-content")
        assert errors == []
    finally:
        page.evaluate("clearCompare()")
        ctx.close()


@needs_browser
def test_phone_compare_and_sheet_do_not_scroll_the_page_sideways(browser):
    ctx, page, _ = _open(browser, 375, 812)
    try:
        page.evaluate("UX.compare = ['EXPE', 'HST', 'BBY']; saveCompare(); openStockDetail('EXPE')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        page.wait_for_timeout(350)
        assert page.evaluate("document.documentElement.scrollWidth") <= 376
        tools = page.evaluate("""() => { const r = document.querySelector('#stock-modal .modal-tools').getBoundingClientRect(); return [r.left, r.right]; }""")
        assert tools[0] >= 0 and tools[1] <= 376, tools
        page.evaluate("closeModal(); openCompare()")
        page.wait_for_timeout(300)
        box = page.evaluate("""() => { const r = document.querySelector('#compare-modal .cmp-content').getBoundingClientRect(); return [r.left, r.right]; }""")
        assert box[0] >= 0 and box[1] <= 376, box
    finally:
        page.evaluate("clearCompare()")
        ctx.close()


@needs_browser
def test_the_guide_dismisses_and_stays_dismissed(browser):
    ctx, page, _ = _open(browser)
    try:
        assert page.is_visible("#guide")
        page.click("#guide .guide-close")
        assert not page.is_visible("#guide")
        page.reload(wait_until="load")
        page.wait_for_timeout(500)
        assert not page.is_visible("#guide")
    finally:
        ctx.close()


@needs_browser
def test_the_workings_csv_redoes_the_arithmetic_on_its_own(browser, tmp_path):
    """The download is only worth having if a student can rebuild the score from the file alone:
    category points add to the composite, and each category's metric points add to its score."""
    import csv
    ctx, page, errors = _open(browser)
    try:
        page.evaluate("openStockDetail('JPM')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        with page.expect_download() as dl:
            page.click("#modal-categories .wk-download")
        path = tmp_path / "w.csv"
        dl.value.save_as(path)
        rows = list(csv.reader(path.read_text(encoding="utf-8-sig").splitlines()))
        composite = float(next(r for r in rows if r[:1] == ["Composite"] and len(r) == 2)[1])
        start = next(i for i, r in enumerate(rows) if r[:1] == ["Category"] and "Points" in r) + 1
        cat_points = []
        for r in rows[start:start + 8]:
            cat_points.append(float(r[3]))
        assert abs(sum(cat_points) - composite) < 0.06, (cat_points, composite)
        metric_rows = [r for r in rows if len(r) == 7 and r[0] == "Valuation" and r[1] != "Metric"]
        score_row = next(r for r in metric_rows if r[1] == "Category score")
        pts = [float(r[5]) for r in metric_rows if r[1] != "Category score" and r[5]]
        assert abs(sum(pts) - float(score_row[5])) < 1e-3
        jpm = page.evaluate("D.stock_detail.JPM.cat_scores.valuation")
        assert abs(float(score_row[5]) - jpm) < 0.06
        assert any(r[:1] == ["Field"] for r in rows), "the reported inputs section is missing"
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_a_trap_bar_lists_exactly_the_flagged_stocks_in_that_sector(browser):
    ctx, page, errors = _open(browser)
    try:
        page.evaluate("toggleSection('sec-analytics')")
        page.wait_for_timeout(200)
        row = page.query_selector("#trap-bars .trap-row[data-sector]")
        sector = row.get_attribute("data-sector")
        flagged = int(row.inner_text().split(" of ")[0].split()[-1])
        row.click()
        page.wait_for_timeout(300)
        assert page.input_value("#filter-sector") == sector
        assert page.input_value("#filter-vt") == "vt"
        assert page.evaluate("tableState.filtered.length") == flagged
        assert page.evaluate("tableState.filtered.every(r => r.Value_Trap_Flag && r.Sector === %s)" % json.dumps(sector))
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_every_metric_row_shows_its_own_numbers(browser):
    """The owner's two asks, 2026-10-07. First: the workings showed a percentile and a direction
    but 'no numbers anywhere ... going into the scoring'. Then, seeing them on every row: 'a
    little cluttery ... maybe if they click on each one, they can see the calculation. But it
    should be easy to recognize ... that it is an option'. So: rows are quiet, every scored row
    carries a labelled Calculation control, the whole row opens it, and the card it opens leads
    with the stock's figures through the formula, its rank, and percentile x weight = points."""
    ctx, page, errors = _open(browser)
    try:
        page.evaluate("openStockDetail('EXPE')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        page.evaluate("openWorkings('valuation')")
        page.wait_for_timeout(300)
        assert "Select any metric to see the numbers behind it" in page.inner_text("#modal-categories")
        assert page.evaluate("document.querySelectorAll('#modal-categories .wk-card').length") == 0, "calculations must start closed"
        rows = page.evaluate("""() => [...document.querySelectorAll('#modal-categories tr.metric-row')]
            .filter(r => !r.classList.contains('wk-nodata'))
            .map(r => [r.dataset.metric, !!r.querySelector('.wk-more'), r.classList.contains('wk-click')])""")
        assert len(rows) > 20 and all(r[1] and r[2] for r in rows), [r for r in rows if not (r[1] and r[2])]
        assert page.inner_text("#cat-detail-valuation tr[data-metric='fcf_yield'] .wk-more").strip() == "Calculation"
        # the whole row opens it, not only the control
        page.click("#cat-detail-valuation tr[data-metric='fcf_yield'] .metric-raw")
        card = page.inner_text("#cat-detail-valuation tr.wk-detail .wk-card")
        raw = page.inner_text("#cat-detail-valuation tr[data-metric='fcf_yield'] .metric-raw").strip()
        assert "free cash flow" in card and "enterprise value" in card and "=" in card, card
        assert raw in card, (raw, card)
        assert "Ranked" in card and "median" in card and "×" in card and "points" in card, card
        assert page.get_attribute("#cat-detail-valuation tr[data-metric='fcf_yield'] .wk-more", "aria-expanded") == "true"
        # every scored metric on the page has a calculation line in its card
        missing = page.evaluate("""() => {
            const s = D.stock_detail.EXPE, out = [];
            Object.keys(D.lineage).forEach(m => {
                if (s.raw[m] === null || s.raw[m] === undefined) return;
                const h = metricDetailHtml(m, s, null);
                if (h.indexOf('>Calculation<') < 0) out.push(m);
            });
            return out; }""")
        assert missing == [], missing
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_unused_metrics_say_why(browser):
    """Owner: 'it should say a little bit about why something was not used in the score'."""
    ctx, page, errors = _open(browser)
    try:
        page.evaluate("openStockDetail('EXPE')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        page.evaluate("openWorkings('valuation')")
        off = page.inner_text("#cat-detail-valuation .wk-off")
        assert "P/B Ratio" in off and "banks and insurers" in off, off
        assert "Dividend Yield" in off and "no weight" in off, off
        page.evaluate("openStockDetail('JPM')")
        page.wait_for_timeout(300)
        page.evaluate("openWorkings('valuation')")
        off = page.inner_text("#cat-detail-valuation .wk-off")
        assert "FCF Yield" in off and "Not used for banks" in off, off
        rows = page.evaluate("[...document.querySelectorAll('#modal-categories .wk-off-row')].map(r => r.querySelector('.wk-off-why').textContent.trim())")
        assert rows and all(len(r) > 20 for r in rows), rows
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_jumping_inside_the_sheet_never_lifts_its_header(browser):
    """scrollIntoView() inside the sheet also scrolled the fixed overlay holding it, lifting the
    header 164px off the screen (found by screenshot, 2026-10-07). The overlay must not scroll."""
    ctx, page, _ = _open(browser, 1440, 1000)
    try:
        page.evaluate("openStockDetail('EXPE')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        for js in ("openWorkings('valuation')", "goToModal('section-peers')", "openWorkings('risk')"):
            page.evaluate(js)
            page.wait_for_timeout(500)
            top = page.evaluate("document.querySelector('#stock-modal .modal-content').getBoundingClientRect().top")
            assert abs(top) < 1, (js, top)
        page.click("#cat-detail-risk tr.metric-row.wk-click .wk-more")
        page.wait_for_timeout(200)
        assert abs(page.evaluate("document.querySelector('#stock-modal .modal-content').getBoundingClientRect().top")) < 1
    finally:
        ctx.close()




@needs_browser
def test_arrow_keys_walk_the_rankings_past_the_rendered_window(browser):
    ctx, page, _ = _open(browser)
    try:
        page.focus("#universe-tbody tr[data-t]")
        first = page.evaluate("document.activeElement.dataset.t")
        assert first == page.evaluate("tableState.filtered[0].Ticker")
        for _ in range(60):  # well past the rows rendered at load
            page.keyboard.press("ArrowDown")
        assert page.evaluate("document.activeElement.dataset.t") == page.evaluate("tableState.filtered[60].Ticker")
        page.keyboard.press("End")
        assert page.evaluate("document.activeElement.dataset.t") == page.evaluate("tableState.filtered[tableState.filtered.length - 1].Ticker")
        page.keyboard.press("Enter")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
    finally:
        ctx.close()


@needs_browser
def test_tab_stays_inside_an_open_drilldown(browser):
    ctx, page, _ = _open(browser)
    try:
        page.evaluate("openStockDetail('EXPE')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        for _ in range(80):
            page.keyboard.press("Tab")
            assert page.evaluate("document.querySelector('#stock-modal .modal-content').contains(document.activeElement)")
    finally:
        ctx.close()


@needs_browser
def test_context_panels_render_without_errors(browser):
    """Before you decide, Market Backdrop and Track Record render from the published payload,
    and the context filter narrows the table without touching the ranking."""
    ctx, page, errors = _open(browser)
    try:
        page.wait_for_function("typeof CTX_LOADED !== 'undefined' && CTX_LOADED", timeout=30000)
        has_ctx = page.evaluate("Object.values(D.stock_detail).some(s => s.ctx)")
        if not has_ctx:
            pytest.skip("payload predates the context layer")
        t = page.evaluate("Object.keys(D.stock_detail).find(k => (D.stock_detail[k].ctx || {}).s200)")
        page.evaluate(f"openStockDetail('{t}')")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        body = page.inner_text("#modal-context")
        assert "Context, not part of the score" in body and "200-day" in body
        assert page.evaluate("!!document.querySelector('#modal-context .pc-svg')")
        page.keyboard.press("Escape")
        if page.evaluate("!!(D.market && D.market.series)"):
            page.evaluate("goToSection('sec-market')")
            assert "FRED" in page.inner_text("#market-body")
        if page.evaluate("!!(D.track && D.track.available)"):
            page.evaluate("goToSection('sec-track')")
            tb = page.inner_text("#track-body")
            assert "not a portfolio to follow" in tb and "Read with care" in tb
        before = page.evaluate("tableState.filtered.map(r => r.Rank)")
        page.select_option("#filter-ctx", "up")
        page.wait_for_timeout(200)
        after = page.evaluate("tableState.filtered.map(r => r.Rank)")
        assert 0 < len(after) < len(before)
        assert after == sorted(after)                      # still in rank order; ranks unchanged
        assert page.evaluate("tableState.filtered.every(r => trendOf((D.stock_detail[r.Ticker] || {}).ctx) === 'up')")
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_reporting_soon_is_a_calendar_not_a_leaderboard(browser):
    """Reporting Soon lists every scored stock reporting within the window of the run date,
    ordered by date then rank - never by the size of the expected move - and opens the sheet."""
    ctx, page, errors = _open(browser)
    try:
        page.wait_for_function("typeof CTX_LOADED !== 'undefined' && CTX_LOADED", timeout=30000)
        page.evaluate("goToSection('sec-reporting')")
        page.wait_for_selector("#reporting-body .rp-controls")
        for days in (7, 14):
            page.evaluate(f"setReporting('days', {days})")
            want = page.evaluate(f"""(() => {{
                const run = String(D.kpis.run_timestamp).slice(0, 10);
                return D.table_data.filter(r => {{
                    const e = (D.stock_detail[r.Ticker] || {{}}).earn;
                    if (!e || !e.d) return false;
                    const d = Math.round((new Date(e.d + 'T00:00:00') - new Date(run + 'T00:00:00')) / 86400000);
                    return d >= 0 && d <= {days};
                }}).map(r => r.Ticker).sort();
            }})()""")
            got = page.evaluate("[...document.querySelectorAll('#reporting-body .rp-row .rp-tk')].map(e => e.textContent)")
            assert sorted(got) == want
            keys = page.evaluate("""[...document.querySelectorAll('#reporting-body .rp-row .rp-tk')].map(e => {
                const t = e.textContent; return [D.stock_detail[t].earn.d, D.table_data.find(r => r.Ticker === t).Rank]; })""")
            assert keys == sorted(keys)                        # date, then rank
        page.evaluate("localStorage.removeItem('screener_holdings_v1'); setReporting('scope', 'held')")
        assert "None of your saved holdings" in page.inner_text("#reporting-body")
        page.evaluate("setReporting('scope', 'all'); setReporting('days', 14)")
        first = page.evaluate("document.querySelector('#reporting-body .rp-row .rp-tk').textContent")
        page.click("#reporting-body .rp-row")
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        assert first in page.inner_text("#stock-modal .modal-header, #stock-modal")
        from stock_summary import advice_terms_in
        assert advice_terms_in(page.inner_text("#reporting-body")) == []
        assert errors == []
    finally:
        ctx.close()


@needs_browser
def test_column_tooltips_state_the_published_weights(browser):
    """Each category header lists exactly the metrics with weight in the run's own table."""
    ctx, page, errors = _open(browser)
    try:
        bad = page.evaluate("""(() => {
            const out = [];
            document.querySelectorAll('#universe-table th[data-wcat]').forEach(th => {
                const t = th.title;
                if (t.indexOf('@') >= 0) out.push(th.dataset.wcat + ': placeholder left');
                const g = D.weights.profiles[th.dataset.wcat].generic;
                Object.entries(g).forEach(([m, w]) => {
                    const lab = (D.metric_meta[m] || {}).label || m;
                    const listed = t.indexOf(lab + ' (') >= 0;
                    if (w > 0 && !listed) out.push(th.dataset.wcat + ': missing ' + lab);
                });
            });
            return out;
        })()""")
        assert bad == []
        assert errors == []
    finally:
        ctx.close()
