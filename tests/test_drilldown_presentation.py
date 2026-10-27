"""The stock sheet's presentation layer (UI elevation, 2026-10-27, ``plan/ui-elevation.md``).

The drilldown gained a masthead (the company in the display serif, rank and composite first, a
strip of every published composite with this one marked), trap-flag chips, figures set in ink
inside the "Why it ranks here" sentences, a stacked bar of category points under How it adds up,
a header that folds once the reader scrolls, and keyboard access to the clickable cards.

All of it is display. These tests pin the two promises that make it safe to ship:

- **it never changes a word or a number** - the sentences are ``stock_summary.py``'s, character for
  character, and the strip and the stacked bar draw the payload's own composites and points;
- **it works at both widths and from the keyboard** - nothing overflows 375px, touch targets are at
  least 40px on a touch screen, and Enter works every card that a click works.

They skip (loudly) where Playwright or a browser is not installed, like ``test_dashboard_browser``.
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


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _open(browser, width=1440, height=900, touch=False):
    ctx = browser.new_context(viewport={"width": width, "height": height}, has_touch=touch, is_mobile=touch)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri(), wait_until="load", timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
    return ctx, page, errors


def _sheet(page, ticker):
    page.evaluate(f"openStockDetail({ticker!r})")
    page.wait_for_selector("#stock-modal .modal-body", state="visible")
    page.wait_for_timeout(300)


def test_figure_emphasis_never_changes_a_sentence(browser):
    """Every stock's rendered sentences are exactly its published ``summary`` texts."""
    ctx, page, errors = _open(browser)
    try:
        bad = page.evaluate("""() => {
            const out = [];
            Object.keys(D.stock_detail).forEach(t => {
                const s = D.stock_detail[t];
                renderSummary(s);
                const got = [...document.querySelectorAll('#modal-summary-body .summary-fact')].map(p => p.textContent).sort();
                const want = (s.summary || []).map(f => String(f.t || '')).sort();
                if (JSON.stringify(got) !== JSON.stringify(want)) out.push(t);
            });
            return out;
        }""")
        assert bad == [], f"{len(bad)} stocks render a sentence that is not the published one: {bad[:5]}"
        # and the emphasis did happen: a figure with a decimal point is wrapped
        page.evaluate("renderSummary(D.stock_detail.EXPE)")
        assert page.locator("#modal-summary-body .summary-lead .fig").count() >= 2
        assert not errors, errors
    finally:
        ctx.close()


def test_figure_emphasis_leaves_entities_and_plain_counts_alone(browser):
    ctx, page, _ = _open(browser)
    try:
        out = page.evaluate("""() => [
            figEmph(escapeHtml("It's 12.5% of $3.2B, 1st of 498 in 2026")),
            figEmph(escapeHtml("A <b> & 39 'x' #39"))
        ]""")
        assert out[0] == ("It&#39;s <span class=\"fig\">12.5%</span> of <span class=\"fig\">$3.2</span>B, "
                          "<span class=\"fig\">1st</span> of 498 in 2026")
        assert out[1] == "A &lt;b&gt; &amp; 39 &#39;x&#39; #39"
    finally:
        ctx.close()


def test_masthead_marks_the_published_composite_among_all_of_them(browser):
    ctx, page, _ = _open(browser)
    try:
        _sheet(page, "EXPE")
        got = page.evaluate("""() => {
            const s = D.stock_detail.EXPE, svg = document.querySelector('#modal-headline .cd-svg');
            return { label: svg.getAttribute('aria-label'), on: svg.querySelectorAll('rect.cd-on').length,
                     comp: fmt(s.composite, 'score'), med: fmt(D.kpis.median_composite, 'score'),
                     n: Object.keys(D.stock_detail).filter(t => typeof D.stock_detail[t].composite === 'number').length,
                     bins: compDist().bins.reduce((a, b) => a + b, 0),
                     head: document.getElementById('modal-headline').textContent };
        }""")
        assert got["on"] == 1
        assert got["bins"] == got["n"]
        assert f"Composite {got['comp']} among the {got['n']} published composites" in got["label"]
        assert f"the median is {got['med']}" in got["label"]
        # rank, then composite, then price: the screener's outputs before the market's
        head = got["head"]
        assert head.index("Rank") < head.index("Composite") < head.index("Last price")
    finally:
        ctx.close()


def test_trap_chips_follow_the_payload(browser):
    ctx, page, _ = _open(browser)
    try:
        for t in page.evaluate("""() => {
            const all = Object.keys(D.stock_detail);
            const both = all.find(t => D.stock_detail[t].vt && D.stock_detail[t].gt);
            const none = all.find(t => !D.stock_detail[t].vt && !D.stock_detail[t].gt);
            const one = all.find(t => !D.stock_detail[t].vt !== !D.stock_detail[t].gt);
            return [both, none, one].filter(Boolean);
        }"""):
            _sheet(page, t)
            want = page.evaluate(f"(s => (s.vt ? ['Value trap'] : []).concat(s.gt ? ['Growth trap'] : []))(D.stock_detail[{t!r}])")
            got = page.evaluate("[...document.querySelectorAll('#modal-chips .flag')].map(e => e.textContent)")
            assert got == want, (t, got, want)
    finally:
        ctx.close()


def test_the_stacked_total_draws_each_categorys_published_points(browser):
    ctx, page, _ = _open(browser)
    try:
        _sheet(page, "JPM")
        got = page.evaluate("""() => {
            const s = D.stock_detail.JPM;
            const segs = [...document.querySelectorAll('#contrib-total .ct-seg')].map(e => parseFloat(e.style.width));
            const want = Object.keys(s.contrib).map(c => s.contrib[c]).filter(v => v > 0);
            return { segs: segs, want: want };
        }""")
        assert len(got["segs"]) == len(got["want"])
        assert abs(sum(got["segs"]) - sum(got["want"])) < 0.05
    finally:
        ctx.close()


def test_header_folds_when_reading_and_opens_at_the_top(browser):
    ctx, page, _ = _open(browser)
    try:
        _sheet(page, "EXPE")
        assert not page.evaluate("document.querySelector('#stock-modal .modal-content').classList.contains('is-compact')")
        page.evaluate("document.querySelector('#stock-modal .modal-body').scrollTop = 1500")
        page.wait_for_timeout(150)
        assert page.evaluate("document.querySelector('#stock-modal .modal-content').classList.contains('is-compact')")
        assert page.is_visible("#mt-next") and page.is_visible("#modal-ticker")
        page.evaluate("document.querySelector('#stock-modal .modal-body').scrollTop = 0")
        page.wait_for_timeout(150)
        assert not page.evaluate("document.querySelector('#stock-modal .modal-content').classList.contains('is-compact')")
    finally:
        ctx.close()


def test_enter_works_the_cards_a_click_works(browser):
    ctx, page, _ = _open(browser)
    try:
        _sheet(page, "EXPE")
        page.focus("#modal-score-row .modal-score-card[role=button]")
        page.keyboard.press("Enter")
        page.wait_for_timeout(200)
        assert not page.evaluate("document.getElementById('cat-detail-valuation').classList.contains('collapsed')")
        page.evaluate("document.getElementById('cat-detail-quality').classList.add('collapsed')")
        page.focus("#cat-detail-quality .cat-detail-header")
        page.keyboard.press(" ")
        assert not page.evaluate("document.getElementById('cat-detail-quality').classList.contains('collapsed')")
    finally:
        ctx.close()


def test_phone_sheet_closes_on_a_long_drag_down_and_springs_back_on_a_short_one(browser):
    ctx, page, _ = _open(browser, 375, 812, touch=True)
    drag = """dy => {
        const head = document.querySelector('#stock-modal .modal-company');
        const r = head.getBoundingClientRect(), x = r.left + 10, y = r.top + 5;
        const t = (yy) => new Touch({ identifier: 1, target: head, clientX: x, clientY: yy });
        head.dispatchEvent(new TouchEvent('touchstart', { touches: [t(y)], bubbles: true }));
        head.dispatchEvent(new TouchEvent('touchmove', { touches: [t(y + dy)], bubbles: true }));
        const mid = document.querySelector('#stock-modal .modal-content').style.transform;
        head.dispatchEvent(new TouchEvent('touchend', { touches: [], bubbles: true }));
        return mid;
    }"""
    try:
        _sheet(page, "EXPE")
        assert page.evaluate(drag, 40) == "translateY(40px)"
        page.wait_for_timeout(250)
        assert page.is_visible("#stock-modal .modal-content")
        assert page.evaluate("document.querySelector('#stock-modal .modal-content').style.transform") == ""
        page.evaluate(drag, 140)
        page.wait_for_timeout(100)
        assert not page.is_visible("#stock-modal .modal-content")
    finally:
        ctx.close()


@pytest.mark.parametrize("ticker", ["EXPE", "JPM"])
def test_phone_sheet_fits_and_its_controls_are_thumb_sized(browser, ticker):
    ctx, page, _ = _open(browser, 375, 812, touch=True)
    try:
        long_name = page.evaluate("Object.keys(D.stock_detail).sort((a, b) => (D.stock_detail[b].company || '').length - (D.stock_detail[a].company || '').length)[0]")
        flagged = page.evaluate("Object.keys(D.stock_detail).find(t => D.stock_detail[t].vt && D.stock_detail[t].gt) || 'EXPE'")
        for t in (ticker, long_name, flagged):
            _sheet(page, t)
            over = page.evaluate("""() => [...document.querySelectorAll('#stock-modal .modal-header *')]
                .filter(e => e.offsetParent !== null).map(e => e.getBoundingClientRect())
                .filter(r => r.width && (r.left < -0.5 || r.right > innerWidth + 0.5)).length""")
            assert over == 0, t
            assert page.evaluate("document.documentElement.scrollWidth") <= 376
            small = page.evaluate("""() => [...document.querySelectorAll('#stock-modal .mt-btn, #stock-modal .modal-close, #stock-modal .modal-nav a')]
                .filter(e => e.offsetParent !== null).map(e => [e.id || e.textContent.trim(), Math.round(e.getBoundingClientRect().height)])
                .filter(x => x[1] < 40)""")
            assert small == [], (t, small)
    finally:
        ctx.close()
