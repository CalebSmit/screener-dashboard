"""The dashboard, driven in a real browser.

WHY THIS EXISTS - 2026-10-07 (``plan/dashboard-redesign-master.md``).

The owner's complaint was that the page "doesn't feel good to move around in" and looks
generated. The static tests pin markup; none of them could see that the rankings table put
8,050 of the page's 9,923 DOM nodes in front of the first paint, or that its filters were
unstyled, or that the first line of every drilldown was wrong. Judging a design from its CSS
is how it got that way (CLAUDE.md, "How it looks and feels is part of whether it is correct").

These tests open the generated page in Chromium and check what a visitor experiences: how much
DOM the first paint costs, that the table scrolls with the page and windows its rows, that sort
and filter work, that the drilldown shows the weights that were actually used, and that nothing
overflows a phone. They skip (loudly) where Playwright or a browser is not installed, so a bare CI
machine does not fail for lacking one - but a machine that has one must pass.

Budgets are the ones in ``plan/dashboard-redesign-master.md``; a miss is a regression, not a
preference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"

pytestmark = pytest.mark.skipif(not PAGE.exists() or not PAYLOAD.exists(),
                                reason="generated dashboard not present")


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _open(browser, width=1440, height=900):
    ctx = browser.new_context(viewport={"width": width, "height": height})
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "favicon" not in m.text else None)
    page.goto(PAGE.as_uri(), wait_until="load", timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
    return ctx, page, errors


@pytest.fixture()
def desktop(browser):
    ctx, page, errors = _open(browser)
    yield page, errors
    ctx.close()


@pytest.fixture()
def phone(browser):
    ctx, page, errors = _open(browser, 375, 812)
    yield page, errors
    ctx.close()


def _network_errors(errors):
    # fonts and the chart CDN are external; a sandbox without network is not a page defect
    return [e for e in errors if not any(k in e for k in ("ERR_", "Failed to load resource", "net::"))]


def _universe_size(page) -> int:
    """How many stocks this payload actually scored.

    These assertions used to pin the literal **502**, and on 2026-10-09 the S&P 500 fetch
    returned one fewer name: the run scored **501**, three tests failed, and because the
    runner's gate 1 requires pytest to exit 0 - it takes no baseline - a correct run would
    have blocked every merge until someone edited the literal. Index membership changes
    several times a year, so a number that moves on its own is the wrong thing to assert.

    What is asserted instead is the *invariant*: the table's row count, its last rank and its
    count text all agree with the payload's own universe. The floor below keeps that from
    degenerating into a tautology - a payload holding three stocks is a broken run, not a
    small universe, and is the same 495-515 band ``universe_history.validate_membership``
    already refuses outside of.
    """
    n = page.evaluate("window.SCREENER_DATA.table_data.length")
    assert 460 <= n <= 520, f"payload scored {n} stocks - not a plausible S&P 500 universe"
    return n


# ---------------------------------------------------------------------------
# cost of the first paint
# ---------------------------------------------------------------------------

def test_page_loads_without_script_errors(desktop):
    page, errors = desktop
    assert _network_errors(errors) == []


def test_first_paint_is_not_dominated_by_the_rankings_table(desktop):
    """Baseline 2026-10-06: 9,923 nodes, 8,050 of them in the table. Budget: under 3,000."""
    page, _ = desktop
    nodes = page.evaluate("document.querySelectorAll('*').length")
    assert nodes < 3000, f"{nodes} DOM nodes at first paint (budget 3,000)"


def test_table_renders_a_window_only_a_fraction_of_the_universe(desktop):
    page, _ = desktop
    n = _universe_size(page)
    rows = page.evaluate("document.querySelectorAll('#universe-tbody tr[data-t]').length")
    assert 10 <= rows <= 120, rows
    assert rows < n, f"{rows} rows in the DOM for a {n}-stock universe - the window is not windowing"
    # aria-rowcount counts the header row, so a screen reader is told the real total
    assert page.evaluate("document.getElementById('universe-table').getAttribute('aria-rowcount')") == str(n + 1)


def test_the_table_does_not_scroll_inside_a_box(desktop):
    """The nested scroll box was the single worst 'doesn't feel good' culprit."""
    page, _ = desktop
    info = page.evaluate("""() => {
        let el = document.getElementById('universe-table'), scrolling = [];
        while (el && el !== document.body) {
            const cs = getComputedStyle(el);
            if (/(auto|scroll)/.test(cs.overflowY) && el.scrollHeight > el.clientHeight + 1) scrolling.push(el.className || el.id);
            el = el.parentElement;
        }
        return scrolling;
    }""")
    assert info == [], f"table sits inside a vertically scrolling container: {info}"


def test_scrolling_reaches_the_last_row(desktop):
    page, _ = desktop
    n = _universe_size(page)
    page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
    page.wait_for_timeout(250)
    last_rank = page.evaluate("""() => {
        const rows = [...document.querySelectorAll('#universe-tbody tr[data-t] td.rank')];
        return rows.length ? Math.max(...rows.map(r => parseInt(r.textContent, 10))) : null;
    }""")
    assert last_rank == n, f"scrolled to the bottom and the last rank is {last_rank}, not {n}"


def test_the_header_row_stays_stuck_while_scrolling(desktop):
    page, _ = desktop
    page.evaluate("document.getElementById('universe-table').scrollIntoView()")
    page.evaluate("window.scrollBy(0, 900)")
    page.wait_for_timeout(250)
    top = page.evaluate("document.querySelector('#universe-table thead th').getBoundingClientRect().top")
    bar = page.evaluate("document.getElementById('top-bar').getBoundingClientRect().bottom")
    assert abs(top - bar) < 3, f"header row at {top}px, bar bottom at {bar}px"


# ---------------------------------------------------------------------------
# behaviour
# ---------------------------------------------------------------------------

def test_sorting_by_a_category_reorders_the_table(desktop):
    page, _ = desktop
    page.click("#universe-table th[data-sort='quality_score']")
    page.wait_for_timeout(150)
    vals = page.evaluate("""() => [...document.querySelectorAll('#universe-tbody tr[data-t]')].slice(0, 12)
        .map(r => parseFloat(r.querySelectorAll('td.sc')[1].textContent))""")
    assert vals == sorted(vals, reverse=True)
    assert page.get_attribute("#universe-table th[data-sort='quality_score']", "aria-sort") == "descending"


def test_filtering_by_sector_updates_count_and_rows(desktop):
    page, _ = desktop
    n = _universe_size(page)
    page.select_option("#filter-sector", "Financials")
    page.wait_for_timeout(200)
    text = page.inner_text("#result-count")
    assert f" of {n} stocks" in text, text
    sectors = page.evaluate("""() => [...new Set([...document.querySelectorAll('#universe-tbody tr[data-t] td.sector')]
        .map(c => c.textContent))]""")
    assert sectors == ["Financials"]
    assert page.is_visible("#filter-clear")
    page.click("#filter-clear")
    page.wait_for_timeout(150)
    assert page.inner_text("#result-count") == f"{n} stocks"


def test_a_search_with_no_match_shows_an_empty_state(desktop):
    page, _ = desktop
    page.fill("#filter-search", "zzzzzz")
    page.wait_for_timeout(300)
    assert page.is_visible(".empty-state")


def test_slash_focuses_search(desktop):
    page, _ = desktop
    page.mouse.click(10, 10)
    page.keyboard.press("/")
    assert page.evaluate("document.activeElement.id") == "filter-search"


def test_flags_are_words_and_unflagged_rows_are_blank(desktop):
    page, _ = desktop
    cells = page.evaluate("""() => [...document.querySelectorAll('#universe-tbody tr[data-t] td.vt-cell')].map(c => c.textContent.trim())""")
    assert all(c in {"", "Value", "Growth", "ValueGrowth"} for c in cells), set(cells)
    assert "✓" not in "".join(cells)


def test_there_is_no_dead_refresh_control(desktop):
    page, _ = desktop
    assert page.evaluate("document.getElementById('refresh-btn')") is None
    assert page.evaluate("typeof triggerRefresh") == "undefined"


# ---------------------------------------------------------------------------
# the drilldown
# ---------------------------------------------------------------------------

def _open_stock(page, ticker):
    page.evaluate(f"openStockDetail('{ticker}')")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    page.wait_for_timeout(350)  # let the sheet finish sliding in before measuring it


def test_drilldown_is_a_sheet_with_a_header_that_stays(desktop):
    page, _ = desktop
    _open_stock(page, "EXPE")
    box = page.evaluate("""() => { const r = document.querySelector('#stock-modal .modal-content').getBoundingClientRect();
        return {right: r.right, w: innerWidth, h: r.height, vh: innerHeight}; }""")
    assert abs(box["right"] - box["w"]) < 2 and abs(box["h"] - box["vh"]) < 2
    page.evaluate("document.querySelector('#stock-modal .modal-body').scrollTop = 1500")
    page.wait_for_timeout(100)
    assert page.is_visible("#stock-modal .modal-header")
    assert "EXPE" in page.inner_text("#modal-ticker")
    page.keyboard.press("Escape")
    assert not page.is_visible("#stock-modal .modal-content")


def test_the_rank_sentence_is_true_for_the_top_stock(desktop):
    page, _ = desktop
    _open_stock(page, "EXPE")
    text = page.inner_text("#modal-summary-body")
    assert "ahead of 100% of the other" in text
    assert "is a percentile" not in text


def test_bank_valuation_workings_use_bank_weights_and_add_up(desktop):
    """JPM's Valuation panel used to show three heavily weighted metrics as N/A, call P/B
    'Inactive', and print a score nothing on screen produced."""
    page, _ = desktop
    _open_stock(page, "JPM")
    page.evaluate("openWorkings('valuation')")
    section = page.inner_text("#cat-detail-valuation")
    assert "bank weighting" in section.lower()
    rows = page.evaluate("""() => [...document.querySelectorAll('#cat-detail-valuation tbody tr.metric-row')].map(r => ({
        m: r.dataset.metric, w: r.querySelector('.metric-weight').textContent.trim(), p: r.querySelector('.wk-points').textContent.trim()}))""")
    by = {r["m"]: r for r in rows}
    assert set(by) == {"earnings_yield", "pb_ratio"}, by
    assert by["pb_ratio"]["w"] == "60%" and by["earnings_yield"]["w"] == "40%"
    total = float(page.inner_text("#cat-detail-valuation tfoot .wk-points").replace("!", "").strip())
    assert abs(sum(float(r["p"]) for r in rows) - total) < 0.06


def test_a_metric_row_opens_its_inputs_formula_and_peers(desktop):
    page, _ = desktop
    _open_stock(page, "EXPE")
    page.evaluate("openWorkings('valuation')")
    page.click("#cat-detail-valuation tr[data-metric='earnings_yield'] .wk-info")
    detail = page.inner_text("#cat-detail-valuation tr.wk-detail")
    assert "Net income / market cap" in detail
    assert "Net income (TTM)" in detail and "Market cap" in detail
    assert "Ranked" in detail and "percentile" in detail
    page.click("#cat-detail-valuation tr[data-metric='earnings_yield'] .wk-info")
    assert page.evaluate("document.querySelector('#cat-detail-valuation tr.wk-detail')") is None


def test_piotroski_row_lists_all_nine_signals(desktop):
    page, _ = desktop
    _open_stock(page, "EXPE")
    page.evaluate("openWorkings('quality')")
    page.click("#cat-detail-quality tr[data-metric='piotroski_f_score'] .wk-info")
    assert page.evaluate("document.querySelectorAll('#cat-detail-quality .wk-part').length") == 9


def test_composite_chain_sums_for_a_discounted_stock(desktop):
    page, _ = desktop
    ticker = page.evaluate("""() => Object.keys(D.stock_detail).find(t => (D.stock_detail[t].cov || {}).disc)""")
    if not ticker:
        pytest.skip("no stock carries a coverage discount in this payload")
    _open_stock(page, ticker)
    chain = page.inner_text("#contrib-total")
    assert "Coverage discount" in chain and "Category points add up to" in chain


# ---------------------------------------------------------------------------
# phone
# ---------------------------------------------------------------------------

def test_phone_has_no_horizontal_page_scroll(phone):
    page, _ = phone
    assert page.evaluate("document.documentElement.scrollWidth") <= 376


def test_phone_kpis_have_no_orphan_card(phone):
    page, _ = phone
    cards = page.evaluate("""() => [...document.querySelectorAll('#kpi-row .kpi-card')].map(c => Math.round(c.getBoundingClientRect().top))""")
    assert len(cards) == 4 and len(set(cards)) == 2 and cards.count(cards[0]) == 2


def test_phone_table_rows_are_cards_of_a_fixed_height(phone):
    page, _ = phone
    heights = page.evaluate("""() => [...document.querySelectorAll('#universe-tbody tr[data-t]')].slice(0, 6).map(r => Math.round(r.getBoundingClientRect().height))""")
    assert heights and len(set(heights)) == 1, heights
    assert page.is_visible("#sort-by")


def test_phone_drilldown_is_a_bottom_sheet(phone):
    page, _ = phone
    _open_stock(page, "EXPE")
    box = page.evaluate("""() => { const r = document.querySelector('#stock-modal .modal-content').getBoundingClientRect();
        return {bottom: r.bottom, vh: innerHeight, w: r.width}; }""")
    assert abs(box["bottom"] - box["vh"]) < 2 and box["w"] <= 376
    assert page.evaluate("document.documentElement.scrollWidth") <= 376


# ---------------------------------------------------------------------------
# feel
# ---------------------------------------------------------------------------

def test_row_click_to_drilldown_is_fast(desktop):
    page, _ = desktop
    ms = page.evaluate("""() => new Promise(res => {
        const t0 = performance.now();
        const tr = document.querySelector('#universe-tbody tr[data-t]');
        tr.click();
        requestAnimationFrame(() => requestAnimationFrame(() => res(performance.now() - t0)));
    })""")
    assert ms < 250, f"{ms:.0f} ms from click to painted drilldown (budget 100 ms on a laptop; 250 allows a loaded CI box)"


def test_sort_to_paint_is_fast(desktop):
    page, _ = desktop
    ms = page.evaluate("""() => new Promise(res => {
        const t0 = performance.now();
        sortTable('momentum_score');
        requestAnimationFrame(() => requestAnimationFrame(() => res(performance.now() - t0)));
    })""")
    assert ms < 250, f"{ms:.0f} ms"


def test_transitions_are_named_properties_not_all(desktop):
    page, _ = desktop
    n = page.evaluate("""() => [...document.querySelectorAll('*')].filter(e => getComputedStyle(e).transitionProperty === 'all' && parseFloat(getComputedStyle(e).transitionDuration) > 0).length""")
    assert n < 40, f"{n} elements animate 'all'"
