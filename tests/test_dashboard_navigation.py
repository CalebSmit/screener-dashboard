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

    guide = html[html.index('<section class="guide"'):html.index("<!-- KPI Row -->")]
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
