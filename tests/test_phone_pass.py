"""The phone pass: on a phone the page is short on arrival and everything long opens on request.

WHY THIS EXISTS - 2026-11-22 (owner, after reading the dashboard on his phone): *"there is too
much info and it got way too long and hard to know where to go."* Measured at 375px that day:

- the ranking began **4,514px** down the home page (five and a half screens);
- one stock's sheet was **9,819px** long, every one of its sections open;
- Market backdrop opened to **6,760px**;
- the Methodology was one **73,020px** column with no contents list (its rail hides below 900px).

The fix is one rule applied everywhere: **on a phone, a long part is a heading that says what is
inside it, and it opens with one tap.** Nothing leaves the page, no number or sentence changes,
and the desktop page is untouched - every rule is inside ``@media screen and (max-width: 760px)``
(``generate_dashboard._css_phone``) or behind ``isPhone()`` (``_js_phone``).

What these tests hold:

- the folding is **phone only** and **screen only** (a printed page shows everything);
- everything folded is **reachable**: by its heading, by any link that targets it, by the tabs;
- a folded filter is never a silent filter - the Filters button counts the ones in use;
- "Read with care" - the trap flags and the count of metrics behind the score - is never folded.

**The ranking's rows, 2026-11-23** (owner: "Yes, simplify rows"). A phone row carried nine numbers -
the composite and eight labelled scores, 450 figures in the first fifty rows, 80px a row. It is now
rank, move, ticker, the whole company name, any trap flag and the composite, 58px; the eight scores
are on the stock, or back on every row with one tap ("Show the 8 scores", which restores the earlier
row exactly). A list sorted by a category shows that category's score on each row, because an order
the row cannot explain is worse than a long row.

Heights are asserted as budgets per folded part, not as page totals: a total moves with the day's
data (how many chips a leader has, how long a headline runs) and a part's folded height does not.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import generate_dashboard as gd  # noqa: E402
from tests import examples  # noqa: E402 - the stock a browser test opens

try:
    from playwright import sync_api
except Exception:  # noqa: BLE001
    sync_api = None

PAGE = ROOT / "index.html"
PAYLOAD = ROOT / "dashboard_data.js"
EX = examples.stock("EXPE")  # a stock with every section of the sheet
FOLDED = ["section-contribution", "section-categories", "section-context", "section-history", "section-price-targets"]
ALWAYS_FOLDED = ["section-snapshot", "section-peers", "section-provenance"]


# --------------------------------------------------------------------------- the source


def test_every_phone_rule_is_phone_only_and_screen_only():
    """The pass may not reach the desktop page or a printed one. Outside the one media block the
    stylesheet does a single thing: hide the three phone-only controls everywhere else."""
    css = gd._css_phone()
    assert css.count("@media") == 1 and "@media screen and (max-width: 760px)" in css
    outside = css.split("@media screen and (max-width: 760px)")[0]
    rules = re.findall(r"([^{}/]+)\{([^{}]*)\}", re.sub(r"/\*.*?\*/", "", outside, flags=re.S))
    assert [(s.strip(), b.strip()) for s, b in rules] == [
        (".guide-open-m, .filters-more-m, .summary-more-m, .rows-scores-m", "display: none;")]


def test_the_phone_styles_and_script_are_emitted_last():
    """Emitted after every other block, so a phone rule wins a tie without ``!important``."""
    html = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    assert html.index("THE STOCK SHEET (UI elevation, 2026-10-27)") < html.index("THE PHONE PASS (2026-11-22)")
    assert html.count("THE PHONE PASS (2026-11-22)") == 2  # the stylesheet and the script
    assert html.rindex("</style>") > html.index("THE PHONE PASS (2026-11-22)")


def test_every_folding_sheet_section_says_what_is_inside():
    """A folded section is a heading plus one line naming its contents; a heading alone sends a
    reader in to find out. The line is the heading's ``data-hint``, shown by CSS on a phone."""
    html = gd.generate_html(methodology_html="<p>m</p>", data_timestamp="t", data_version="v")
    for sid in FOLDED + ALWAYS_FOLDED:
        block = html[html.index(f'<div class="collapsible' , html.index(f'id="{sid}"') - 60):]
        m = re.search(r'<span class="collapsible-t" data-hint="([^"]+)">([^<]+)</span>', block[:900])
        assert m, f"{sid}: no hint on its heading"
        assert 15 <= len(m.group(1)) <= 60, f"{sid}: hint should be one short line, got {m.group(1)!r}"
        assert not re.search(r"\d", m.group(1)), f"{sid}: a hint describes, it never states a figure"


# --------------------------------------------------------------------------- the browser

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


def _open(browser, width=375, height=812, hash_=""):
    phone = width <= 760
    ctx = browser.new_context(viewport={"width": width, "height": height}, is_mobile=phone, has_touch=phone)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri() + hash_, wait_until="load", timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
    return ctx, page, errors


def _open_sheet(page, ticker=EX):
    page.evaluate("t => openStockDetail(t)", ticker)
    page.wait_for_function("document.getElementById('stock-modal').style.display !== 'none' "
                           "&& document.querySelectorAll('#modal-summary-body p').length > 0", timeout=15000)


def _shown(page, selector):
    """Is the first match laid out (not display:none, not inside something that is)?"""
    return page.evaluate("s => { const e = document.querySelector(s); return !!e && e.getClientRects().length > 0; }", selector)


def _no_sideways_scroll(page):
    return page.evaluate("document.documentElement.scrollWidth <= innerWidth")


@needs_browser
def test_on_a_phone_the_sheet_opens_short_and_every_section_is_one_tap_away(browser):
    ctx, page, errors = _open(browser)
    try:
        _open_sheet(page)
        state = page.evaluate("""ids => ids.map(id => { const e = document.getElementById(id);
            const t = e.querySelector('.collapsible-t');
            return { id, shown: e.style.display !== 'none', folded: e.classList.contains('collapsed'), h: e.offsetHeight,
                     hint: getComputedStyle(t, '::after').content, expanded: e.querySelector('.collapsible-header').getAttribute('aria-expanded') }; })""",
                              FOLDED + ALWAYS_FOLDED)
        for s in state:
            if not s["shown"]:
                continue  # Rank history is absent for a stock with no comparable run
            assert s["folded"], f"{s['id']} is open on arrival"
            assert s["expanded"] == "false", f"{s['id']}: aria-expanded should say it is folded"
            assert s["hint"] not in ("none", '""', "normal"), f"{s['id']}: the folded heading shows no hint"
            assert s["h"] <= 110, f"{s['id']}: a folded section is a heading and a line, not {s['h']}px"
        # every route in opens what it targets: the tabs (goToModal), a score tile, a case-against row
        page.evaluate("goToModal('section-context')")
        assert not page.evaluate("document.getElementById('section-context').classList.contains('collapsed')")
        assert _shown(page, "#modal-context .ctx-card")
        page.evaluate("openWorkings('valuation')")
        assert not page.evaluate("document.getElementById('section-categories').classList.contains('collapsed')")
        go = page.evaluate("""() => { const row = document.querySelector('#modal-case-list [onclick*="goToModal"], #modal-case-list [data-go]');
            if (!row) return null; row.click(); const m = (row.getAttribute('onclick') || '').match(/goToModal\\('([^']+)'\\)/);
            return m ? m[1] : (row.dataset.go || null); }""")
        if go:
            assert not page.evaluate("id => document.getElementById(id).classList.contains('collapsed')", go), go
        # the reader's choice stands from stock to stock, as it does on a wide screen
        page.evaluate("stepStock(1)")
        assert not page.evaluate("document.getElementById('section-context').classList.contains('collapsed')")
        assert _no_sideways_scroll(page)
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_on_a_wide_screen_nothing_is_folded_by_the_phone_pass(browser):
    ctx, page, errors = _open(browser, 1440, 900)
    try:
        _open_sheet(page)
        folded = page.evaluate("ids => ids.filter(id => document.getElementById(id).classList.contains('collapsed'))", FOLDED)
        assert folded == [], f"the phone pass folded {folded} at 1440px"
        for sel in ("#guide-open-m", "#filters-more-m", "#summary-more-m", "#rows-scores-m"):
            assert not _shown(page, sel), f"{sel} is a phone control and shows at 1440px"
        assert page.evaluate("document.getElementById('dist').className") == "dist" and _shown(page, "#dist-plot svg")
        hidden = page.evaluate("[...document.querySelectorAll('#modal-summary-body .summary-fold')].filter(e => e.getClientRects().length === 0).length")
        assert hidden == 0, "a summary sentence is folded at 1440px"
        assert page.evaluate("getComputedStyle(document.querySelector('#stock-modal .collapsible-t'), '::after').content") in ("none", "normal")
        row = page.evaluate("""() => { const r = document.querySelector('#universe-tbody tr[data-t]');
            return { h: r.getBoundingClientRect().height, sc: [...r.querySelectorAll('td.sc')].filter(c => c.getClientRects().length > 0).length }; }""")
        assert row == {"h": 40, "sc": 8}, f"the wide table's row changed: {row}"
        page.evaluate("closeModal(); openMethodology()")
        assert page.evaluate("document.querySelectorAll('#methodology-modal .m-folded, #methodology-modal h2.m-acc').length") == 0
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_on_a_phone_the_summary_folds_two_groups_and_never_read_with_care(browser):
    ctx, page, errors = _open(browser)
    try:
        _open_sheet(page)
        got = page.evaluate("""() => { const b = document.getElementById('modal-summary-body');
            const vis = e => e.getClientRects().length > 0;
            const labels = [...b.querySelectorAll('.summary-group-label')].map(e => [e.textContent.trim(), vis(e), e.classList.contains('summary-fold')]);
            return { labels, lead: vis(b.querySelector('.summary-lead')),
                     care: [...b.querySelectorAll('.summary-flags, .summary-confidence')].map(vis),
                     folded: [...b.querySelectorAll('p.summary-fold')].map(vis),
                     btn: document.getElementById('summary-more-m').hidden ? null : document.getElementById('summary-more-m').textContent }; }""")
        assert got["lead"], "the lead sentence is hidden"
        assert got["care"] and all(got["care"]), "Read with care must never be folded: it carries the trap flags"
        for name, visible, fold in got["labels"]:
            assert fold == (name in ("What changed", "Context")), f"{name}: wrong fold membership"
            assert visible != fold, f"{name}: should be {'folded' if fold else 'shown'} on arrival"
        if not got["folded"]:
            assert got["btn"] is None, "a button with nothing behind it"
            return
        assert not any(got["folded"]) and got["btn"] == "Show what changed and context"
        page.click("#summary-more-m")
        after = page.evaluate("""() => [[...document.querySelectorAll('#modal-summary-body p.summary-fold')].every(e => e.getClientRects().length > 0),
            document.getElementById('summary-more-m').textContent, document.getElementById('summary-more-m').getAttribute('aria-expanded')]""")
        assert after == [True, "Show less", "true"]
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_on_a_phone_the_home_page_is_headings_until_asked(browser):
    ctx, page, errors = _open(browser)
    try:
        # the first-visit guide: one line, opened by its own button and by "How to read this"
        assert _shown(page, "#guide-open-m") and not _shown(page, "#guide .guide-steps")
        assert page.evaluate("document.getElementById('guide').offsetHeight") <= 70
        page.click("#guide-open-m")
        assert _shown(page, "#guide .guide-steps") and page.get_attribute("#guide-open-m", "aria-expanded") == "true"
        page.click("#guide-open-m")
        assert not _shown(page, "#guide .guide-steps")
        page.evaluate("showGuide()")
        assert _shown(page, "#guide .guide-steps"), "How to read this must show the guide open, not its one line"
        # the dot chart: its title is the line that opens it, and it opens at the width it was drawn at
        page.wait_for_function("document.querySelectorAll('#dist-plot svg').length === 1", timeout=15000)
        dist = page.evaluate("""() => { const d = document.getElementById('dist'), p = document.getElementById('dist-plot');
            return { h: d.offsetHeight, plot: p.offsetHeight, hidden: getComputedStyle(p).visibility,
                     expanded: d.querySelector('.dist-head').getAttribute('aria-expanded'), w: p.querySelector('svg').getBoundingClientRect().width }; }""")
        assert dist["plot"] == 0 and dist["hidden"] == "hidden" and dist["expanded"] == "false" and dist["h"] <= 80, dist
        page.click("#dist .dist-head")
        opened = page.evaluate("""() => { const p = document.getElementById('dist-plot');
            return [p.offsetHeight > 100, getComputedStyle(p).visibility, p.querySelector('svg').getBoundingClientRect().width,
                    document.querySelector('#dist .dist-head').getAttribute('aria-expanded')]; }""")
        assert opened[0] and opened[1] == "visible" and opened[3] == "true", opened
        assert opened[2] == dist["w"] and 200 < opened[2] <= 375, "the chart must open at the width it was drawn at"
        # a leader's card: name, composite and reasons; the bars and run strip are in its sheet
        page.wait_for_function("document.querySelectorAll('#top5-row .top5-card .top5-ticker').length > 0", timeout=15000)
        assert _shown(page, "#top5-row .top5-card .traits") and not _shown(page, "#top5-row .top5-card .t5-fp")
        # the section tiles: a title and the headline, each a short row
        tiles = page.evaluate("""() => [...document.querySelectorAll('#sec-grid > .collapsible-section.collapsed')]
            .filter(e => e.offsetHeight > 0).map(e => [e.id, e.offsetHeight,
                [...e.querySelectorAll('.sec-blurb, .tile-viz')].filter(x => x.getClientRects().length > 0).length])""")
        assert len(tiles) >= 4, tiles
        for sid, h, extra in tiles:
            assert extra == 0, f"{sid}: its blurb or mini chart still shows on a phone"
            assert h <= 150, f"{sid}: a folded tile is {h}px"
        # opening a tile still shows the whole section
        page.evaluate("toggleSection('sec-changed')")
        assert page.evaluate("document.getElementById('sec-changed').offsetHeight") > 400
        assert _no_sideways_scroll(page)
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_on_a_phone_a_folded_filter_is_never_a_silent_one(browser):
    ctx, page, errors = _open(browser)
    try:
        vis = lambda sel: _shown(page, sel)  # noqa: E731
        assert vis("#filter-search") and vis("#sort-by") and vis("#filters-more-m")
        for sel in ("#filter-sector", "#filter-vt", "#filter-ctx", "#filter-profile", "#filter-comp-min"):
            assert not vis(sel), f"{sel} shows before Filters is opened"
        assert page.evaluate("document.getElementById('filters-more-n').hidden")
        # a filter set from elsewhere on the page (a link, a tile) while the group is folded
        total = page.evaluate("SCREENER_DATA.table_data.length")
        page.evaluate("""() => { const s = document.getElementById('filter-vt'); s.value = 'any';
            s.dispatchEvent(new Event('change', { bubbles: true })); }""")
        page.wait_for_function("!document.getElementById('filters-more-n').hidden", timeout=5000)
        assert page.inner_text("#filters-more-n") == "1"
        assert "1 in use" in page.get_attribute("#filters-more-m", "aria-label")
        assert vis("#filter-clear"), "Clear filters must stay in view while a folded filter is on"
        shown = page.evaluate("document.querySelectorAll('#universe-tbody tr[data-t]').length")
        assert 0 < shown < total and vis("#result-count")
        page.click("#filters-more-m")
        assert vis("#filter-sector") and page.get_attribute("#filters-more-m", "aria-expanded") == "true"
        page.click("#filter-clear")
        page.wait_for_function("document.getElementById('filters-more-n').hidden", timeout=5000)
        assert _no_sideways_scroll(page)
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_on_a_phone_market_backdrop_opens_to_labels_and_gauges(browser):
    ctx, page, errors = _open(browser)
    try:
        # the section draws itself when it is opened, once the context file has landed
        page.wait_for_function("!!window.SCREENER_CONTEXT", timeout=30000)
        if not page.evaluate("!!(window.SCREENER_CONTEXT.market && (window.SCREENER_CONTEXT.market.readings || []).length)"):
            pytest.skip("MISSING DATA: this run published no market backdrop")
        page.evaluate("if (document.getElementById('sec-market').classList.contains('collapsed')) toggleSection('sec-market')")
        page.wait_for_function("document.querySelectorAll('#sec-market .mk-reading').length > 0", timeout=20000)
        page.wait_for_function("document.querySelectorAll('#sec-market .mk-reading.m-clamped .m-more').length > 0", timeout=5000)
        got = page.evaluate("""() => { const host = document.getElementById('sec-market');
            const rs = [...host.querySelectorAll('.mk-reading')], gs = [...host.querySelectorAll('.mk-group')];
            const p = rs[0].querySelector('p:not(.mk-gloss)');
            return { readings: rs.length, buttons: host.querySelectorAll('.mk-reading > .m-more').length,
                     gauges: rs.filter(r => r.querySelector('.mk-r-h').getClientRects().length > 0).length,
                     clipped: p.scrollHeight > p.clientHeight + 2, groups: gs.length,
                     grids: gs.filter(g => g.querySelector('.mk-grid').getClientRects().length > 0).length,
                     heads: gs.filter(g => g.querySelector('.mk-group-h').getAttribute('aria-expanded') === 'false').length }; }""")
        assert got["buttons"] == got["readings"] == got["gauges"], got
        assert got["groups"] >= 1 and got["grids"] == 0 and got["heads"] == got["groups"], got
        page.click("#sec-market .mk-reading .m-more")
        assert page.evaluate("""() => { const r = document.querySelector('#sec-market .mk-reading'), p = r.querySelector('p:not(.mk-gloss)');
            return r.classList.contains('m-open') && p.scrollHeight <= p.clientHeight + 2 && r.querySelector('.m-more').textContent === 'Show less'; }""")
        page.click("#sec-market .mk-group .mk-group-h")
        assert page.evaluate("""() => { const g = document.querySelector('#sec-market .mk-group');
            return g.querySelector('.mk-grid').getClientRects().length > 0 && g.querySelector('.mk-group-h').getAttribute('aria-expanded') === 'true'; }""")
        assert _no_sideways_scroll(page)
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_on_a_phone_the_methodology_is_its_section_headings_with_overview_open(browser):
    ctx, page, errors = _open(browser)
    try:
        page.evaluate("openMethodology()")
        page.wait_for_function("document.querySelectorAll('#methodology-modal .methodology-body h2').length > 0", timeout=15000)
        got = page.evaluate("""() => { const b = document.querySelector('#methodology-modal .methodology-body');
            const hs = [...b.children].filter(e => e.tagName === 'H2');
            const folded = b.scrollHeight;
            hs.forEach((h, i) => methSetOpen(b, i, true));
            const open = b.scrollHeight;
            hs.forEach((h, i) => methSetOpen(b, i, i === 0));
            return { n: hs.length, acc: hs.filter(h => h.classList.contains('m-acc')).length,
                     expanded: hs.map(h => h.getAttribute('aria-expanded')), folded, open,
                     headsShown: hs.filter(h => h.getClientRects().length > 0).length,
                     first: b.querySelector('[data-msec="0"]').getClientRects().length > 0,
                     last: b.querySelector('[data-msec="' + (hs.length - 1) + '"]').getClientRects().length > 0 }; }""")
        if got["n"] < 4:
            pytest.skip("MISSING DATA: the methodology has fewer than four sections; nothing to fold")
        assert got["acc"] == got["headsShown"] == got["n"], "every section heading stays in view as the contents list"
        assert got["expanded"] == ["true"] + ["false"] * (got["n"] - 1)
        assert got["first"] and not got["last"]
        assert got["folded"] < got["open"] / 3, f"folded {got['folded']}px of {got['open']}px: the fold is not folding"
        # a tap opens a section; a cross-reference into a folded section opens it before the jump
        page.click("#methodology-modal .methodology-body h2[data-msec-head='1']")
        assert page.get_attribute("#methodology-modal .methodology-body h2[data-msec-head='1']", "aria-expanded") == "true"
        assert _shown(page, "#methodology-modal .methodology-body [data-msec='1']")
        target = page.evaluate("""() => { const b = document.querySelector('#methodology-modal .methodology-body');
            const hs = [...b.children].filter(e => e.tagName === 'H2'), h = hs[hs.length - 1];
            const a = document.createElement('a'); a.href = '#' + methodSlug(h.textContent); a.textContent = 'x'; a.id = 'phone-test-link';
            b.querySelector('[data-msec="0"]').appendChild(a); a.click();
            return h.getAttribute('aria-expanded'); }""")
        assert target == "true", "a link into a folded section did not open it"
        assert _no_sideways_scroll(page)
        assert not errors, errors
    finally:
        ctx.close()


ROW_JS = """() => {
    const t = document.getElementById('universe-table'), r = t.querySelector('tbody tr[data-t]');
    const seen = s => [...r.querySelectorAll(s)].filter(c => c.getClientRects().length > 0);
    const co = r.querySelector('td.company');
    return { h: r.getBoundingClientRect().height, rowH: parseFloat(getComputedStyle(t).getPropertyValue('--row-h')),
             scores: seen('td.sc').map(c => c.dataset.l), rank: seen('td.rank').length, ticker: seen('td.ticker').length,
             comp: seen('td.comp').length, company: seen('td.company').length,
             nameCut: co.scrollWidth > co.clientWidth + 1, nameW: co.getBoundingClientRect().width,
             full: t.classList.contains('rows-full'), sortl: t.getAttribute('data-sortl') };
}"""


@needs_browser
def test_on_a_phone_a_ranking_row_is_identity_and_composite_and_the_scores_are_one_tap_away(browser):
    ctx, page, errors = _open(browser)
    try:
        row = page.evaluate(ROW_JS)
        assert row["scores"] == [], f"a phone row arrives with category scores on it: {row['scores']}"
        assert (row["rank"], row["ticker"], row["company"], row["comp"]) == (1, 1, 1, 1), row
        assert row["h"] == row["rowH"] <= 60, f"the compact row is {row['h']}px (it was 80)"
        assert row["nameW"] >= 200, f"the company name has {row['nameW']:.0f}px - it had ~140 beside the ticker"
        # the windowed table places its spacers by --row-h: the rows drawn must be that tall, all of them
        heights = page.evaluate("[...new Set([...document.querySelectorAll('#universe-tbody tr[data-t]')].map(r => r.getBoundingClientRect().height))]")
        assert heights == [row["rowH"]], heights

        btn = page.locator("#rows-scores-m")
        assert btn.is_visible() and btn.get_attribute("aria-pressed") == "false"
        assert btn.evaluate("e => e.getBoundingClientRect().height") >= 44
        btn.tap()
        full = page.evaluate(ROW_JS)
        assert full["full"] and full["scores"] == ["Val", "Qual", "Grow", "Mom", "Risk", "Rev", "Size", "Inv"]
        assert full["h"] == full["rowH"] == 80, "Show the 8 scores puts the earlier row back, at its own height"
        assert btn.get_attribute("aria-pressed") == "true" and "Hide" in btn.inner_text()
        assert _no_sideways_scroll(page)
        btn.tap()
        assert page.evaluate(ROW_JS)["scores"] == [] and btn.get_attribute("aria-pressed") == "false"
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_on_a_phone_the_choice_to_see_scores_is_remembered(browser):
    ctx, page, errors = _open(browser)
    try:
        page.locator("#rows-scores-m").tap()
        page.reload(wait_until="load")
        page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
        row = page.evaluate(ROW_JS)
        assert row["full"] and len(row["scores"]) == 8 and row["h"] == row["rowH"] == 80
        assert page.locator("#rows-scores-m").get_attribute("aria-pressed") == "true"
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_on_a_phone_a_list_sorted_by_a_category_shows_that_score_on_the_row(browser):
    """Sorted by Momentum, rank 467 can head the list: the row has to show the number it is sorted by.
    The score read off the row is the payload's own, for whichever stock the day's data puts first."""
    ctx, page, errors = _open(browser)
    try:
        for col, label in (("momentum_score", "Mom"), ("valuation_score", "Val"), ("investment_score", "Inv")):
            page.select_option("#sort-by", col)
            page.wait_for_function("l => document.getElementById('universe-table').getAttribute('data-sortl') === l", arg=label)
            row = page.evaluate(ROW_JS)
            assert row["scores"] == [label], f"sorted by {col}, the row shows {row['scores']}"
            assert row["h"] == row["rowH"] <= 60
            shown, want = page.evaluate("""([col, label]) => {
                const r = document.querySelector('#universe-tbody tr[data-t]');
                const d = window.SCREENER_DATA.table_data.find(x => x.Ticker === r.dataset.t);
                return [parseFloat(r.querySelector('td.sc[data-l="' + label + '"]').textContent), d[col]]; }""", [col, label])
            assert abs(shown - want) < 0.051, f"{col}: the row says {shown}, the payload {want}"
        page.select_option("#sort-by", "Rank")
        page.wait_for_function("!document.getElementById('universe-table').hasAttribute('data-sortl')")
        assert page.evaluate(ROW_JS)["scores"] == []
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_on_a_phone_a_trap_flag_stays_on_the_compact_row(browser):
    """A flag is the one thing on a row that says "read the detail first"; the compact row keeps it,
    spelled out, beside the ticker - and the company name keeps its whole line."""
    ctx, page, errors = _open(browser)
    try:
        from tests import phone as phone_taps
        phone_taps.open_filters(page)
        page.select_option("#filter-vt", "any")
        page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0")
        n = page.evaluate("document.querySelectorAll('#universe-tbody tr[data-t]').length")
        flagged = page.evaluate("""() => [...document.querySelectorAll('#universe-tbody tr[data-t]')].map(r => {
            const f = [...r.querySelectorAll('.vt-cell .flag')], c = r.querySelector('td.company').getBoundingClientRect();
            const k = r.querySelector('td.ticker').getBoundingClientRect();
            return { n: f.length, seen: f.every(e => e.getClientRects().length > 0 && e.getBoundingClientRect().right <= innerWidth),
                     word: f.every(e => getComputedStyle(e, '::after').content.includes('trap')),
                     beside: f.every(e => e.getBoundingClientRect().left >= k.right - 1 && e.getBoundingClientRect().bottom <= c.top + 2),
                     nameW: c.width, h: r.getBoundingClientRect().height }; })""")
        assert n > 0 and all(r["n"] >= 1 for r in flagged), "the premise: the trap filter lists flagged stocks"
        bad = [r for r in flagged if not (r["seen"] and r["word"] and r["beside"] and r["nameW"] >= 200 and r["h"] <= 60)]
        assert bad == [], bad[:3]
        assert not errors, errors
    finally:
        ctx.close()


@needs_browser
def test_the_phone_page_still_fits_a_small_phone(browser):
    """320px: the new controls - the guide line, the Filters row, the folded headings and their
    hints - must not push the page or the sheet sideways."""
    ctx, page, errors = _open(browser, 320, 640)
    try:
        assert _no_sideways_scroll(page)
        over = page.evaluate("""() => ['#guide-open-m', '#filters-more-m', '.filter-sort-m', '#filter-search', '#rows-scores-m', '#universe-tbody tr[data-t]']
            .filter(s => { const e = document.querySelector(s); return e && e.getBoundingClientRect().right > innerWidth + 0.5; })""")
        assert over == [], over
        _open_sheet(page)
        wide = page.evaluate("""() => [...document.querySelectorAll('#stock-modal .collapsible-header, #stock-modal .modal-score-card, #summary-more-m')]
            .filter(e => e.getClientRects().length > 0 && e.getBoundingClientRect().right > innerWidth + 0.5).length""")
        assert wide == 0
        assert not errors, errors
    finally:
        ctx.close()
