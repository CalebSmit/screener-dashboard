"""The stock sheet read by a student, a keyboard, a screen reader, a phone and a printer
(owner-run drilldown pass, 2026-10-27 night; ``plan/ui-elevation.md``).

What this pins:

- **plain words** - every metric the workings can show has a one-line "What it measures"
  (``generate_dashboard.METRIC_PLAIN``) that is definitional only: no digit, no computing verb, no
  advice term; and each category opens with the score key's own line (``SCORE_KEY``, claim
  ``dashboard.score_key``);
- **structure** - every section title in the sheet is a real heading, the dialog is named by the
  ticker *and* the company, and every collapsible header says whether it is open;
- **moving around** - a tab moves focus to the section it names and is marked ``aria-current``;
  a strip wider than the sheet says so; a step keeps focus on the control that stepped and tells
  a screen reader which stock is showing;
- **the research under a context reading** opens from a real button;
- **a phone** labels the workings' figures, whose column heads it hides;
- **paper** - printing with a sheet open prints the sheet, every section open, and puts it back.

Presentation only: nothing here reads a number the sheet does not already print. Browser tests skip
loudly where Playwright or Chromium is missing, like ``test_dashboard_browser``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import generate_dashboard as gd
from stock_summary import advice_terms_in

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"
WAIT = 8000

pytestmark = pytest.mark.skipif(not PAGE.exists() or not PAYLOAD.exists(),
                                reason="generated dashboard not present")


def _payload() -> dict:
    txt = PAYLOAD.read_text(encoding="utf-8")
    return json.loads(txt[txt.index("{"):txt.rindex("}") + 1])


# ---------------------------------------------------------------------------
# plain words (no browser)
# ---------------------------------------------------------------------------

def test_every_metric_the_workings_can_show_says_what_it_measures():
    D = _payload()
    shown = set(D["metric_meta"])
    for profs in D["weights"]["profiles"].values():
        for table in profs.values():
            shown |= {m for m, w in table.items() if w and w > 0}
    missing = sorted(m for m in shown if not gd.METRIC_PLAIN.get(m))
    assert missing == [], missing


COMPUTING = re.compile(r"\b(comput\w*|calculat\w*|divid\w*|multipl\w*|times|formula|equals?|sum of|minus|plus)\b", re.I)


@pytest.mark.parametrize("metric", sorted(gd.METRIC_PLAIN))
def test_a_plain_line_is_definitional_only(metric):
    """No figure, threshold or computing verb (the card's Calculation and Definition lines carry
    those, checked against every stock), and no advice."""
    text = gd.METRIC_PLAIN[metric]
    assert 40 <= len(text) <= 320, (metric, len(text))
    assert not re.search(r"\d", text), (metric, text)
    assert not COMPUTING.search(text), (metric, COMPUTING.search(text).group(0))
    assert advice_terms_in(text) == [], (metric, advice_terms_in(text))


def test_the_page_carries_the_words_it_shows():
    html = PAGE.read_text(encoding="utf-8")
    assert "const METRIC_PLAIN = " + json.dumps(gd.METRIC_PLAIN, ensure_ascii=False) + ";" in html
    means = {k: text for k, _s, _n, text in gd.SCORE_KEY}
    assert "const CAT_MEANS = " + json.dumps(means, ensure_ascii=False) + ";" in html


def test_every_sheet_section_title_is_a_heading_in_the_markup():
    html = PAGE.read_text(encoding="utf-8")
    sheet = html[html.index('id="stock-modal"'):html.index("<!-- Methodology Modal -->")]
    assert 'aria-labelledby="modal-ticker modal-company"' in sheet
    for title in ("Why it ranks here", "The case against", "About", "How it adds up", "The workings",
                  "Before you decide", "Rank history", "Analyst price targets", "Company snapshot",
                  "Sector peers", "Data provenance"):
        assert re.search(r"<h3[^>]*>\s*(?:<div[^>]*>\s*)?(?:<span>)?" + re.escape(title), sheet), title
    # no section header is a bare role="button" outside a heading any more
    assert sheet.count('class="collapsible-header"') == sheet.count('<h3 class="collapsible-h">') == 8


# ---------------------------------------------------------------------------
# in a browser
# ---------------------------------------------------------------------------

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _open(browser, width=1280, height=900, touch=False):
    ctx = browser.new_context(viewport={"width": width, "height": height}, has_touch=touch, is_mobile=touch)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri(), wait_until="load", timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
    return ctx, page, errors


def _sheet(page, ticker):
    page.evaluate(f"openStockDetail({ticker!r})")
    page.wait_for_function(f"UX.current === {ticker!r} && document.getElementById('modal-ticker').textContent === {ticker!r}"
                           " && getComputedStyle(document.getElementById('stock-modal')).display !== 'none'", timeout=WAIT)


def test_headings_name_and_open_state(browser):
    ctx, page, errors = _open(browser)
    try:
        _sheet(page, "JPM")
        got = page.evaluate("""() => {
            const m = document.querySelector('#stock-modal .modal-content');
            const name = m.getAttribute('aria-labelledby').split(' ').map(id => document.getElementById(id).textContent).join(' ');
            const h3 = [...document.querySelectorAll('#stock-modal h3')].filter(h => h.getClientRects().length).map(h => h.textContent.replace(/[\\u25bc\\u25be]/g, '').trim());
            const h4 = [...document.querySelectorAll('#modal-categories h4')].length;
            // a heading must not sit inside a role=button (it would be flattened)
            const flattened = [...document.querySelectorAll('#stock-modal [role=button] h1, #stock-modal [role=button] h2, #stock-modal [role=button] h3, #stock-modal [role=button] h4')].length;
            return { name, h3, h4, flattened };
        }""")
        assert got["name"] == "JPM JPMorgan Chase & Co.", got["name"]
        for t in ("Why it ranks here", "The case against", "How it adds up", "The workings", "Before you decide",
                  "Analyst price targets", "Company snapshot", "Sector peers", "Data provenance"):
            assert t in got["h3"], (t, got["h3"])
        assert got["h4"] == 8 and got["flattened"] == 0, got
        # the workings' category headers say whether they are open, whichever path opened them
        exp = "[...document.querySelectorAll('#modal-categories .cat-detail-header')].map(h => h.getAttribute('aria-expanded')).join(',')"
        assert page.evaluate(exp) == ",".join(["false"] * 8)
        page.evaluate("openWorkings('quality')")
        page.wait_for_function(f"{exp} === 'false,true,false,false,false,false,false,false'", timeout=WAIT)
        page.evaluate("toggleAllWorkings(true)")
        page.wait_for_function(f"{exp} === '{','.join(['true'] * 8)}'", timeout=WAIT)
        page.focus("#cat-detail-risk .cat-detail-header")
        page.keyboard.press("Enter")
        page.wait_for_function("document.querySelector('#cat-detail-risk .cat-detail-header').getAttribute('aria-expanded') === 'false'", timeout=WAIT)
        # the page-section headers inside headings still report their state
        page.evaluate("toggleSection('section-peers')")
        page.wait_for_function("document.querySelector('#section-peers .collapsible-header').getAttribute('aria-expanded') === 'true'", timeout=WAIT)
        assert errors == []
    finally:
        ctx.close()


def test_plain_words_in_the_workings(browser):
    ctx, page, errors = _open(browser)
    try:
        _sheet(page, "JPM")
        page.evaluate("openWorkings('valuation')")
        means = page.inner_text("#cat-detail-valuation .wk-cat-means")
        key = dict((k, t) for k, _s, _n, t in gd.SCORE_KEY)
        assert means == "A higher Valuation score means " + key["valuation"]
        page.click("#cat-detail-valuation tr[data-metric='pb_ratio'] .wk-more")
        page.wait_for_selector("#cat-detail-valuation tr.wk-detail .wk-plain", timeout=WAIT)
        plain = page.inner_text("#cat-detail-valuation tr.wk-detail .wk-plain")
        assert plain.startswith("What it measures") or plain.upper().startswith("WHAT IT MEASURES"), plain
        assert gd.METRIC_PLAIN["pb_ratio"] in plain
        # every weighted row of every category offers it, through the same card
        missing = page.evaluate("""() => { const s = D.stock_detail.JPM, out = [];
            document.querySelectorAll('#modal-categories tr.metric-row.wk-click').forEach(r => {
                if (metricDetailHtml(r.dataset.metric, s, null).indexOf('class="wk-plain"') < 0) out.push(r.dataset.metric); });
            return out; }""")
        assert missing == [], missing
        assert errors == []
    finally:
        ctx.close()


def test_a_tab_moves_focus_to_its_section_and_is_marked_current(browser):
    ctx, page, errors = _open(browser)
    try:
        _sheet(page, "EXPE")
        page.wait_for_function("document.querySelectorAll('#stock-modal .modal-nav a[aria-current=location]').length === 1", timeout=WAIT)
        page.focus("#stock-modal .modal-nav a[href='#section-peers']")
        page.keyboard.press("Enter")
        page.wait_for_function("document.activeElement === document.querySelector('#section-peers .collapsible-header')", timeout=WAIT)
        assert page.evaluate("document.getElementById('section-peers').classList.contains('collapsed')") is False
        page.focus("#stock-modal .modal-nav a[href='#modal-summary']")
        page.keyboard.press("Enter")
        page.wait_for_function("document.activeElement === document.getElementById('summary-title')", timeout=WAIT)
        # the lit tab carries aria-current and only one does
        page.wait_for_function("""() => { const a = [...document.querySelectorAll('#stock-modal .modal-nav a')];
            return a.filter(x => x.getAttribute('aria-current') === 'location').length === 1 &&
                   a.every(x => (x.getAttribute('aria-current') === 'location') === x.classList.contains('active')); }""", timeout=WAIT)
        assert errors == []
    finally:
        ctx.close()


def test_a_strip_wider_than_the_sheet_says_so(browser):
    ctx, page, errors = _open(browser, 1280, 900)
    try:
        _sheet(page, "EXPE")
        page.wait_for_function("""() => { const n = document.querySelector('#stock-modal .modal-nav');
            return n.scrollWidth > n.clientWidth + 1; }""", timeout=WAIT)
        page.wait_for_function("document.querySelector('#stock-modal .mnav-wrap').classList.contains('more-r')", timeout=WAIT)
        assert page.is_visible("#stock-modal .mnav-r") and not page.is_visible("#stock-modal .mnav-l")
        page.click("#stock-modal .mnav-r")
        page.wait_for_function("document.querySelector('#stock-modal .mnav-wrap').classList.contains('more-l')", timeout=WAIT)
        # the arrows are not tab stops: the keyboard reaches every tab directly
        assert page.evaluate("[...document.querySelectorAll('#stock-modal .mnav-scroll')].every(b => b.tabIndex === -1)")
        page.evaluate("document.querySelector('#stock-modal .modal-nav').scrollLeft = 1e6")
        page.wait_for_function("!document.querySelector('#stock-modal .mnav-wrap').classList.contains('more-r')", timeout=WAIT)
        assert errors == []
    finally:
        ctx.close()


def test_stepping_keeps_focus_and_says_which_stock(browser):
    ctx, page, errors = _open(browser)
    try:
        first = page.evaluate("D.table_data.slice().sort((a, b) => a.Rank - b.Rank)[0].Ticker")
        second = page.evaluate("D.table_data.slice().sort((a, b) => a.Rank - b.Rank)[1].Ticker")
        _sheet(page, first)
        page.focus("#mt-next")
        page.keyboard.press("Enter")
        page.wait_for_function(f"UX.current === {second!r}", timeout=WAIT)
        assert page.evaluate("document.activeElement.id") == "mt-next"
        ann = page.inner_text("#modal-announce")
        s = page.evaluate(f"D.stock_detail[{second!r}]")
        assert ann.startswith(second + ", ") and f"Rank {s['rank']} of " in ann, ann
        assert page.get_attribute("#modal-announce", "aria-live") == "polite"
        # back to the first stock: Previous becomes disabled, so focus moves to Next, never to Close
        page.focus("#mt-prev")
        page.keyboard.press("Enter")
        page.wait_for_function(f"UX.current === {first!r}", timeout=WAIT)
        assert page.evaluate("document.activeElement.id") == "mt-next"
        assert errors == []
    finally:
        ctx.close()


def test_the_research_under_a_reading_opens_from_a_button(browser):
    ctx, page, errors = _open(browser)
    try:
        page.wait_for_function("!!window.SCREENER_CONTEXT", timeout=30000)
        _sheet(page, "EXPE")
        page.wait_for_function("document.querySelectorAll('#modal-context .ctx-why-more').length > 0", timeout=WAIT)
        b = "#modal-context .ctx-why-more"
        pid = page.get_attribute(b, "aria-controls")
        assert page.get_attribute(b, "aria-expanded") == "false" and page.inner_text(b) == "Read the research"
        page.focus(b)
        page.keyboard.press("Enter")
        page.wait_for_function(f"document.getElementById({pid!r}).classList.contains('open')", timeout=WAIT)
        assert page.get_attribute(b, "aria-expanded") == "true" and page.inner_text(b) == "Show less"
        # the paragraph's own click (kept) moves the button with it
        page.click(f"#{pid}")
        page.wait_for_function(f"!document.getElementById({pid!r}).classList.contains('open')", timeout=WAIT)
        assert page.get_attribute(b, "aria-expanded") == "false"
        assert errors == []
    finally:
        ctx.close()


def test_a_phone_labels_the_workings_figures(browser):
    ctx, page, errors = _open(browser, 375, 812, touch=True)
    try:
        _sheet(page, "EXPE")
        page.evaluate("openWorkings('valuation')")
        page.wait_for_function("!document.getElementById('cat-detail-valuation').classList.contains('collapsed')", timeout=WAIT)
        got = page.evaluate("""() => { const r = document.querySelector('#cat-detail-valuation tr.metric-row:not(.wk-nodata)');
            const a = sel => getComputedStyle(r.querySelector(sel), '::after').content;
            return { pct: a('.metric-pct-label'), w: a('.metric-weight'), pts: a('.wk-points'),
                     tap: Math.round(r.querySelector('.wk-more').getBoundingClientRect().height),
                     text: r.querySelector('.wk-points').textContent }; }""")
        assert got["pct"] == '" pctile"' and got["w"] == '" weight"' and got["pts"] == '" pts"', got
        assert got["tap"] >= 44, got
        # a label is drawn, never written: the figure's own text is unchanged
        assert re.fullmatch(r"\d+\.\d\d", got["text"].strip()), got["text"]
        assert page.evaluate("document.documentElement.scrollWidth") <= 376
        assert errors == []
    finally:
        ctx.close()


def test_printing_with_a_sheet_open_prints_the_sheet_and_puts_it_back(browser):
    ctx, page, errors = _open(browser)
    try:
        _sheet(page, "EXPE")
        before = page.evaluate("[...document.querySelectorAll('#stock-modal .collapsible.collapsed')].map(s => s.id)")
        assert before, "some sections start folded"
        page.evaluate("window.print = () => {}; printSheet()")
        page.emulate_media(media="print")
        page.wait_for_function("document.documentElement.classList.contains('print-sheet')", timeout=WAIT)
        got = page.evaluate("""() => ({
            sheet: getComputedStyle(document.getElementById('stock-modal')).display,
            others: [...document.querySelectorAll('.dashboard-container > :not(#stock-modal), body > :not(.dashboard-container)')]
                .filter(e => getComputedStyle(e).display !== 'none').map(e => e.id || e.className),
            folded: document.querySelectorAll('#stock-modal .collapsible.collapsed').length,
            tools: getComputedStyle(document.querySelector('#stock-modal .modal-tools')).display,
            tabs: getComputedStyle(document.querySelector('#stock-modal .mnav-wrap')).display,
            body: getComputedStyle(document.querySelector('#stock-modal .modal-body')).overflowY,
            bg: getComputedStyle(document.querySelector('#stock-modal .modal-content')).backgroundColor })""")
        assert got["sheet"] == "block" and got["others"] == [], got
        assert got["folded"] == 0 and got["tools"] == "none" and got["tabs"] == "none", got
        assert got["body"] == "visible" and got["bg"] == "rgb(255, 255, 255)", got
        page.emulate_media(media="screen")
        page.evaluate("window.dispatchEvent(new Event('afterprint'))")
        page.wait_for_function("!document.documentElement.classList.contains('print-sheet')", timeout=WAIT)
        after = page.evaluate("[...document.querySelectorAll('#stock-modal .collapsible.collapsed')].map(s => s.id)")
        assert after == before
        # with no sheet open, printing is the page's own business
        page.evaluate("closeModal(); window.dispatchEvent(new Event('beforeprint'))")
        assert page.evaluate("document.documentElement.classList.contains('print-sheet')") is False
        assert errors == []
    finally:
        ctx.close()
