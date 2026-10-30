"""The Factor Lab read by hand, by keyboard and on paper (2026-10-27, plan/ui-elevation.md second pass).

An audit of ``lab.html`` as a student would use it found the bar charts were the one kind a phone
could not read (their values print only where a bar has room, so eight of the size effect's ten
bars carried no figure at 375 px) and no keyboard could read them at all; a series that failed to
arrive replaced the whole experiment with one line and no way to try again; opening an experiment
left the tab titled "Factor Lab" and the keyboard focus on a menu entry; the picker's list of
eighteen entries answered only Tab; several controls were under 44 px on a phone; the equations hid
their right half in a box that scrolled sideways; and a printed experiment came out white on white.

Every assertion waits on a condition (8 s), never on a fixed sleep.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_lab as fl  # noqa: E402

PAGE = ROOT / "lab.html"
WAIT = 8000
BY_ID = {x["id"]: x for x in fl.EXPERIMENTS}
# one experiment of every kind that draws bars, and what its first bar's reading starts with
BARS = {
    "size-effect": lambda page: "Size group 1 of 10, the smallest:",
    "low-volatility": lambda page: BY_ID["low-volatility"]["sort"]["groups"][0] + ":",
    "big-and-small": lambda page: page.evaluate("LAB_DATA.split_names[LAB_DATA.split_keys[0]]") + ":",
    "replication-crisis": lambda page: page.evaluate("LAB_STATE().result.themes[0].theme") + ":",
    "size-given-profitability": lambda page: BY_ID["size-given-profitability"]["double"]["rows"][0] + ":",
}
BAR_IDLE = "Point at a bar, or tap it, to read its figures."


def _hit(i) -> str:
    """True when the bars of slot i, and only they, are marked as being read."""
    return (f"(() => {{ const h = [...document.querySelectorAll('#ch-bars rect.hit')]; "
            f"return h.length > 0 && h.every(r => r.dataset.i === '{i}'); }})()")


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")
    with sync_api.sync_playwright() as pw:
        try:
            b = pw.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _page(browser, width=375, touch=False, init=None):
    ctx = browser.new_context(viewport={"width": width, "height": 900}, has_touch=touch, is_mobile=touch)
    if init:
        ctx.add_init_script(init)
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    return ctx, page, errors


def _goto(page, scene: str) -> None:
    page.goto(PAGE.as_uri() + "#" + scene, wait_until="load", timeout=60000)
    page.wait_for_function(f"window.LAB_STATE && window.LAB_STATE() && window.LAB_STATE().id === '{scene}'", timeout=WAIT)
    # and the opening scene fully drawn: the 153 factors' chart waits on a second part (round three)
    page.wait_for_function("!document.querySelector('#lab [aria-busy=\"true\"]')", timeout=WAIT)


def test_every_bar_chart_can_be_read_from_the_keyboard(browser):
    ctx, page, errors = _page(browser)
    try:
        for scene, first in BARS.items():
            _goto(page, scene)
            svg, read = "#ch-bars svg", "#ch-bars + .chart-read"
            assert page.get_attribute(svg, "tabindex") == "0" and page.get_attribute(svg, "role") == "img", scene
            assert "arrow keys" in page.get_attribute(svg, "aria-label"), scene
            assert page.get_attribute(read, "aria-live") == "polite"
            assert page.inner_text(read) == BAR_IDLE
            n = page.evaluate("new Set([...document.querySelectorAll('#ch-bars rect[data-i]')].map(r => r.dataset.i)).size")
            page.focus(svg)
            page.keyboard.press("Home")
            page.wait_for_function(f"document.querySelector('{read}').textContent.startsWith({first(page)!r})", timeout=WAIT)
            # the bar being read is marked on the chart, and only that bar (or that pair)
            assert page.evaluate(_hit(0))
            page.keyboard.press("ArrowRight")
            page.wait_for_function(_hit(1), timeout=WAIT)
            page.keyboard.press("End")
            page.wait_for_function(_hit(n - 1), timeout=WAIT)
            page.keyboard.press("ArrowRight")                                     # held at the last bar
            assert page.evaluate(_hit(n - 1))
            page.focus("#c-from")                                                  # leaving the chart clears it
            page.wait_for_function(f"document.querySelector('{read}').textContent === {BAR_IDLE!r}", timeout=WAIT)
            assert page.evaluate("document.querySelectorAll('#ch-bars rect.hit').length") == 0
        # the size chart's reading is the figure the build computed for that group
        _goto(page, "size-effect")
        page.focus("#ch-bars svg")
        page.keyboard.press("End")
        mean = page.evaluate("LAB_STATE().result.deciles[9].mean.toFixed(2)").replace("-", "−")
        page.wait_for_function("document.querySelector('#ch-bars + .chart-read').textContent.includes('the largest')", timeout=WAIT)
        assert f": {mean}% a month on average." in page.inner_text("#ch-bars + .chart-read")
        assert errors == []
    finally:
        ctx.close()


def test_a_bar_reading_stays_after_a_tap_on_a_touch_screen(browser):
    ctx, page, errors = _page(browser, touch=True)
    try:
        _goto(page, "size-effect")
        page.locator("#ch-bars svg").scroll_into_view_if_needed()
        box = page.locator("#ch-bars svg").bounding_box()
        page.touchscreen.tap(box["x"] + box["width"] * 0.25, box["y"] + box["height"] * 0.6)
        page.wait_for_function(f"document.querySelector('#ch-bars + .chart-read').textContent !== {BAR_IDLE!r}", timeout=WAIT)
        assert page.inner_text("#ch-bars + .chart-read").startswith("Size group ")
        assert page.evaluate("document.querySelectorAll('#ch-bars rect.hit').length") == 1
        assert errors == []
    finally:
        ctx.close()


def test_a_line_chart_s_label_ends_on_its_last_month(browser):
    ctx, page, errors = _page(browser, width=1200)
    try:
        _goto(page, "risk-managed-momentum")
        last = page.evaluate("LAB_STATE().result.months.slice(-1)[0]")
        mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][int(last[4:]) - 1] + " " + last[:4]
        label = page.get_attribute("#ch-growth svg", "aria-label")
        assert f"Last month shown, {mon}: $1 had become $" in label and "arrow keys" in label
        assert errors == []
    finally:
        ctx.close()


SABOTAGE = """
window.__failOnce = true;
const __append = Node.prototype.appendChild;
Node.prototype.appendChild = function (n) {
    if (window.__failOnce && n && n.tagName === 'SCRIPT' && /lab_data_size\\.js/.test(n.src)) {
        window.__failOnce = false; n.src = 'lab_data_not_a_file.js';
    }
    return __append.call(this, n);
};
"""


def test_a_part_that_fails_to_load_keeps_the_experiment_and_can_be_tried_again(browser):
    ctx, page, errors = _page(browser, init=SABOTAGE)
    try:
        page.goto(PAGE.as_uri() + "#size-effect", wait_until="load", timeout=60000)
        page.wait_for_selector("#x-retry", timeout=WAIT)
        # what did arrive stays: the experiment's title and verdict, from lab_data.js
        assert page.inner_text("#x-title") == BY_ID["size-effect"]["title"]
        assert page.get_attribute("#x-failed", "role") == "alert"
        assert "did not load" in page.inner_text("#x-failed")
        assert page.evaluate("document.querySelector('.exp').hasAttribute('aria-busy')") is False
        page.click("#x-retry")
        page.wait_for_function("window.LAB_STATE().id === 'size-effect'", timeout=WAIT)
        assert page.evaluate("document.querySelectorAll('#ch-bars rect').length") == 10
        assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
        assert errors == []
    finally:
        ctx.close()


def test_opening_an_experiment_names_it_in_the_tab_and_moves_focus_to_its_title(browser):
    ctx, page, errors = _page(browser, width=1200)
    try:
        _goto(page, "overview")
        assert page.title() == "Factor Lab · Multi-Factor Screener"
        assert page.evaluate("document.activeElement === document.body")          # a first load moves nothing
        x = fl.EXPERIMENTS[3]
        page.click(".card >> nth=3")
        page.wait_for_function(f"window.LAB_STATE().id === '{x['id']}'", timeout=WAIT)
        page.wait_for_function("document.activeElement && document.activeElement.id === 'x-title'", timeout=WAIT)
        assert page.title() == (x.get("tab") or x["title"]) + " · Factor Lab"
        # the pager does the same, and so does a part that has to be fetched first
        nxt = fl.EXPERIMENTS[4]
        page.click("#pager a.nx")
        page.wait_for_function(f"window.LAB_STATE().id === '{nxt['id']}'", timeout=WAIT)
        page.wait_for_function("document.activeElement && document.activeElement.id === 'x-title'", timeout=WAIT)
        assert page.title() == (nxt.get("tab") or nxt["title"]) + " · Factor Lab"
        page.click("#tabs a[data-id='overview']")
        page.wait_for_function("window.LAB_STATE().id === 'overview'", timeout=WAIT)
        page.wait_for_function("document.activeElement && document.activeElement.id === 'o-exp'", timeout=WAIT)
        assert page.title() == "Factor Lab · Multi-Factor Screener"
        assert errors == []
    finally:
        ctx.close()


def test_the_experiment_list_answers_the_arrow_keys(browser):
    ctx, page, errors = _page(browser)
    try:
        _goto(page, "overview")
        page.focus("#picker")
        page.keyboard.press("ArrowDown")                                           # opens, on the entry on screen
        page.wait_for_function("document.getElementById('picker').getAttribute('aria-expanded') === 'true'", timeout=WAIT)
        page.wait_for_function("document.activeElement.dataset.id === 'overview'", timeout=WAIT)
        page.keyboard.press("ArrowDown")
        page.wait_for_function("document.activeElement.dataset.id === document.querySelectorAll('#tabs a')[1].dataset.id", timeout=WAIT)
        page.keyboard.press("End")
        page.wait_for_function("document.activeElement === [...document.querySelectorAll('#tabs a')].pop()", timeout=WAIT)
        page.keyboard.press("Home")
        page.wait_for_function("document.activeElement.dataset.id === 'overview'", timeout=WAIT)
        page.keyboard.press("ArrowUp")                                             # held at the first entry
        assert page.evaluate("document.activeElement.dataset.id") == "overview"
        page.keyboard.press("Escape")
        page.wait_for_function("document.activeElement.id === 'picker'", timeout=WAIT)
        assert page.get_attribute("#picker", "aria-expanded") == "false"
        assert errors == []
    finally:
        ctx.close()


TARGETS = (".bar nav a, #picker, .controls button, .controls select, .controls input[type=range], "
           ".verdict > a, .exp summary, .pager a, .warn .retry, .next-nav a")


def test_on_a_phone_every_control_is_at_least_44_px_tall(browser):
    ctx, page, errors = _page(browser)
    try:
        for scene in ("overview", "risk-managed-momentum", "momentum-crashes", "size-effect", "replication-crisis"):
            _goto(page, scene)
            short = page.evaluate(f"""[...document.querySelectorAll({TARGETS!r})]
                .filter(e => e.offsetParent !== null && getComputedStyle(e).display !== 'none' && !e.hidden)
                .map(e => [e.id || e.textContent.trim().slice(0, 30), e.getBoundingClientRect().height])
                .filter(x => x[1] < 43.5)""")
            assert short == [], (scene, short)
            assert page.evaluate(f"document.querySelectorAll({TARGETS!r}).length") >= 3, scene
        page.click("#picker")
        page.wait_for_function("document.getElementById('tabs').classList.contains('open')", timeout=WAIT)
        assert page.evaluate("Math.min(...[...document.querySelectorAll('#tabs a')].map(a => a.getBoundingClientRect().height))") >= 43.5
        assert errors == []
    finally:
        ctx.close()


def test_on_a_phone_an_equation_wraps_instead_of_scrolling_sideways(browser):
    ctx, page, errors = _page(browser)
    try:
        for x in fl.EXPERIMENTS:
            _goto(page, x["id"])
            wide = page.evaluate("[...document.querySelectorAll('.eq')].filter(e => e.scrollWidth > e.clientWidth + 1).length")
            assert wide == 0, x["id"]
        assert errors == []
    finally:
        ctx.close()


def test_the_no_effect_line_starts_its_own_line(browser):
    """Run on after a long verdict note, the line's tick sat alone at the end of a line and read as a
    stray capital I (seen at 375 px on the size effect)."""
    ctx, page, errors = _page(browser)
    try:
        _goto(page, "size-effect")
        assert page.evaluate("getComputedStyle(document.querySelector('.verdict .null')).display") == "block"
        tick, line = page.evaluate("""(() => { const t = document.querySelector('.verdict .null .nl').getBoundingClientRect(),
            v = document.querySelector('.verdict .null').getBoundingClientRect(); return [t.left, v.left]; })()""")
        assert abs(tick - line) < 2
        assert errors == []
    finally:
        ctx.close()


def test_a_printed_experiment_is_ink_on_paper_without_its_controls(browser):
    ctx, page, errors = _page(browser, width=1200)
    try:
        _goto(page, "size-effect")
        page.emulate_media(media="print")
        assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(255, 255, 255)"
        assert page.evaluate("getComputedStyle(document.getElementById('x-title')).color") == "rgb(17, 17, 17)"
        for sel in (".bar", ".controls", "#pager", "#index"):
            assert page.evaluate(f"getComputedStyle(document.querySelector({sel!r})).display") == "none", sel
        # the result, its chart and the comparison with the paper are what is printed
        for sel in ("#x-read", "#ch-bars svg", "#t-ref"):
            assert page.evaluate(f"getComputedStyle(document.querySelector({sel!r})).display") != "none", sel
        assert errors == []
    finally:
        ctx.close()


def test_the_verdict_leads_to_what_it_means_for_the_screener(browser):
    """A student reading a verdict asks what it means for the screener; the answer was the fourth
    section down. The verdict now links to it, and the link keeps the experiment and its address."""
    ctx, page, errors = _page(browser, width=1200)
    try:
        for x in fl.EXPERIMENTS[:4]:
            _goto(page, x["id"])
            page.click(".verdict a[href='#x-scr']")
            page.wait_for_function("Math.abs(document.getElementById('x-scr').getBoundingClientRect().top) < 300", timeout=WAIT)
            assert page.evaluate("window.LAB_STATE().id") == x["id"]
            assert page.evaluate("location.hash") == "#" + x["id"]
        assert errors == []
    finally:
        ctx.close()


def test_the_overview_keys_its_meters_and_the_foot_leads_on(browser):
    ctx, page, errors = _page(browser)
    try:
        _goto(page, "overview")
        key = page.inner_text(".meter-key")
        for words in ("within its tolerance", "outside it", "no effect at all would match"):
            assert words in key, words
        hrefs = page.evaluate("[...document.querySelectorAll('#foot .next-nav a')].map(a => a.getAttribute('href'))")
        assert hrefs == ["#overview", "research.html", "index.html"]
        assert all((ROOT / h).exists() for h in hrefs[1:])
        # a count such as "17 of 19" stays on one line on a phone, and no table slides sideways for it
        _goto(page, "replication-crisis")
        tall = page.evaluate("""[...document.querySelectorAll('#t-live tbody td:nth-child(2)')]
            .filter(td => { const r = document.createRange(); r.selectNodeContents(td);
                return new Set([...r.getClientRects()].map(x => Math.round(x.top))).size > 1; }).length""")
        assert tall == 0
        assert page.evaluate("[...document.querySelectorAll('.tablewrap')].every(w => w.scrollWidth <= w.clientWidth)")
        assert errors == []
    finally:
        ctx.close()


def test_without_javascript_the_page_says_why_and_where_to_go():
    """It used to say "Loading the experiment..." for ever."""
    html = PAGE.read_text(encoding="utf-8")
    ns = html[html.index("<noscript><p"):html.index("</noscript>", html.index("<noscript><p"))]
    assert "needs JavaScript" in ns and 'href="research.html"' in ns


# --------------------------------------------------------------------------- follow-up pass, 2026-10-27
def test_on_a_phone_the_overview_scans_as_a_list_with_every_experiment_on_it(browser):
    """At 375 px the seventeen cards ran to about 7,000 px of an overview 19,000 px long. Each card is
    now its title and verdict meter, with the rest under "What it found"; no experiment is hidden."""
    ctx, page, errors = _page(browser)
    try:
        _goto(page, "overview")
        n = len(fl.EXPERIMENTS)
        assert page.evaluate("[...document.querySelectorAll('.card .card-a strong')].filter(e => e.offsetParent).length") == n
        assert page.evaluate("[...document.querySelectorAll('.card .meter svg')].filter(e => e.getBoundingClientRect().height > 0).length") == n
        assert page.evaluate("[...document.querySelectorAll('.card-body')].filter(e => e.offsetParent).length") == 0
        assert page.evaluate("document.querySelector('#o-exp').closest('section').offsetHeight") < 4500
        # "What it found" opens one card's detail and says so; it does not open the experiment
        x = fl.EXPERIMENTS[2]
        page.click(f"#o-exp ~ .cards .card-more[aria-controls='cb-{x['id']}']")
        page.wait_for_function(f"document.getElementById('cb-{x['id']}').offsetParent !== null", timeout=WAIT)
        assert page.get_attribute(f".card-more[aria-controls='cb-{x['id']}']", "aria-expanded") == "true"
        assert page.evaluate("window.LAB_STATE().id") == "overview"
        body = page.inner_text(f"#cb-{x['id']}")
        assert str(x["paper"]["year"]) in body and "no effect at all would match" in body
        page.click(f".card-more[aria-controls='cb-{x['id']}']")
        page.wait_for_function(f"document.getElementById('cb-{x['id']}').offsetParent === null", timeout=WAIT)
        # the card itself still opens its experiment, from its title as from anywhere else on it
        page.click(".card >> nth=5")
        page.wait_for_function(f"window.LAB_STATE().id === '{fl.EXPERIMENTS[5]['id']}'", timeout=WAIT)
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        assert errors == []
    finally:
        ctx.close()


def test_on_a_wide_screen_every_card_shows_all_of_itself(browser):
    ctx, page, errors = _page(browser, width=1200)
    try:
        _goto(page, "overview")
        assert page.evaluate("[...document.querySelectorAll('.card-body')].filter(e => e.offsetParent).length") == len(fl.EXPERIMENTS)
        assert page.evaluate("[...document.querySelectorAll('.card-more')].filter(e => e.offsetParent).length") == 0
        assert errors == []
    finally:
        ctx.close()


def test_every_bar_chart_axis_carries_its_unit_and_fits(browser):
    ctx, page, errors = _page(browser)
    try:
        for scene in BARS:                                                      # the y axis's labels sit at L - 8
            _goto(page, scene)
            # (the 153 factors on a phone are a row per theme, round three: their scale runs along the foot, `text.tick`)
            ticks = page.evaluate("[...document.querySelectorAll('#ch-bars text')].filter(t => (t.getAttribute('text-anchor') === 'end'"
                                  " && t.getAttribute('x') === '44') || t.classList.contains('tick')).map(t => t.textContent)")
            assert len(ticks) >= 3 and all(t.endswith("%") for t in ticks), (scene, ticks)
            assert not any(t.startswith("-") for t in ticks), (scene, ticks)               # a true minus, as elsewhere
            spill = page.evaluate("""(() => { const s = document.querySelector('#ch-bars svg').getBoundingClientRect().left;
                return Math.min(...[...document.querySelectorAll('#ch-bars text')].map(t => t.getBoundingClientRect().left)) - s; })()""")
            assert spill >= -1, (scene, spill)
        assert errors == []
    finally:
        ctx.close()


CLIPBOARD = """
Object.defineProperty(navigator, 'clipboard', { configurable: true,
    value: { writeText: t => (window.__copied = t, window.__clipFails ? Promise.reject(new Error('no')) : Promise.resolve()) } });
"""


def test_copy_a_link_confirms_through_a_status_region(browser):
    ctx, page, errors = _page(browser, init=CLIPBOARD)
    try:
        page.on("dialog", lambda d: d.dismiss())
        _goto(page, "size-effect")
        status = "#c-copy-status"
        assert page.get_attribute(status, "role") == "status" and page.get_attribute(status, "aria-live") == "polite"
        assert page.inner_text(status) == ""                                     # present, and empty, before anything is said
        page.click("#c-weighting button[data-v='vw']")
        page.click("#c-copy")
        page.wait_for_function("document.getElementById('c-copy-status').textContent.includes('on the clipboard')", timeout=WAIT)
        assert page.evaluate("window.__copied").endswith("#size-effect?weighting=vw")
        page.evaluate("window.__clipFails = true")
        page.click("#c-copy")
        page.wait_for_function("document.getElementById('c-copy-status').textContent.includes('could not be reached')", timeout=WAIT)
        assert errors == []
    finally:
        ctx.close()


def test_on_a_phone_the_hero_figures_sit_beside_their_labels(browser):
    """Three 110 px columns set each label five words to a line over five lines."""
    ctx, page, errors = _page(browser)
    try:
        _goto(page, "overview")
        rows = page.evaluate("""[...document.querySelectorAll('#kpis .kpi')].map(k => {
            const v = k.querySelector('.v').getBoundingClientRect(), l = k.querySelector('.k'), b = l.getBoundingClientRect();
            return { vRight: v.right, kLeft: b.left, kWidth: b.width, lines: Math.round(b.height / parseFloat(getComputedStyle(l).lineHeight)),
                     overlapY: v.top < b.bottom && b.top < v.bottom }; })""")
        assert len(rows) == 3
        for r in rows:
            assert r["overlapY"] and r["vRight"] <= r["kLeft"], r            # the figure beside its label
            assert r["kWidth"] >= 150 and r["lines"] <= 3, r
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        assert errors == []
    finally:
        ctx.close()



# --------------------------------------------------------------------------- round two, 2026-10-27
# plan/ui-elevation.md "round two": the overview's three tables on a phone, the page's landmarks and
# underlined links (an axe-core audit, run by hand and not by this suite - no test may need the
# network), and the two speed changes: the menu's files asked for from the head, and an experiment's
# series asked for at the first sign that the reader is choosing it.
def test_on_a_phone_the_three_overview_tables_read_as_rows_with_every_row_kept(browser):
    """At 375 px the three tables after the cards were 8,313 px (Inter loaded). After the paper is a
    row per result - its name across the width, its two figures under the two column heads, the
    drift drawn beside them; the scorecard and the category map keep every category, its weight,
    what the lab measured or covers and its experiments, and fold only the third column behind a
    button named for it. Measured 5,615 px; the bound here holds with or without the web font."""
    ctx, page, errors = _page(browser)
    try:
        _goto(page, "overview")
        n_after = page.evaluate("LAB_DATA.after_the_paper.length")
        rows = page.evaluate("""[...document.querySelectorAll('#t-after tbody tr')].map(tr => {
            const c = [...tr.children].map(td => td.getBoundingClientRect()), t = tr.getBoundingClientRect();
            return { full: c[0].width >= t.width - 1, sideBySide: Math.abs(c[1].top - c[2].top) < 1 && c[1].right <= c[2].left,
                     under: c[1].top >= c[0].bottom - 1, drift: c[3].width > 0,
                     hasDrift: !!tr.querySelector('svg.drift') }; })""")
        assert len(rows) == n_after
        for r in rows:
            assert r["full"] and r["sideBySide"] and r["under"], r
            assert r["drift"] or not r["hasDrift"], r
        assert page.is_visible("#t-after thead th:nth-child(2)") and page.is_visible("#t-after thead th:nth-child(3)")
        for tid, n, label in (("t-score", 8, "Left open"), ("t-map", 8, "Not covered")):
            assert page.evaluate(f"[...document.querySelectorAll('#{tid} tbody tr')].filter(tr => tr.offsetHeight > 0).length") == n
            assert page.evaluate(f"[...document.querySelectorAll('#{tid} .cat')].filter(e => e.offsetParent).length") == n
            assert page.evaluate(f"[...document.querySelectorAll('#{tid} tbody td:nth-child(2)')].filter(e => e.offsetParent && e.textContent.trim()).length") == n
            assert page.evaluate(f"[...document.querySelectorAll('#{tid} td.folds')].filter(e => e.offsetParent).length") == 0
            btn = f"#{tid} tbody tr:nth-child(3) .row-more"
            assert page.inner_text(btn) == label and page.get_attribute(btn, "aria-expanded") == "false"
            cell = page.get_attribute(btn, "aria-controls")
            box = page.evaluate(f"document.querySelector('{btn}').getBoundingClientRect().height")
            assert box >= 44
            page.click(btn)
            page.wait_for_function(f"document.getElementById('{cell}').offsetParent !== null", timeout=WAIT)
            assert page.get_attribute(btn, "aria-expanded") == "true"
            assert page.evaluate(f"getComputedStyle(document.getElementById('{cell}'), '::before').content") == f'"{label}: "'
            assert page.evaluate("window.LAB_STATE().id") == "overview"
            page.click(btn)
            page.wait_for_function(f"document.getElementById('{cell}').offsetParent === null", timeout=WAIT)
        total = page.evaluate("['t-after', 't-score', 't-map'].reduce((a, id) => a + document.getElementById(id).offsetHeight, 0)")
        assert total < 6000, total
        assert page.evaluate("[...document.querySelectorAll('.tablewrap')].every(w => w.scrollWidth <= w.clientWidth)")
        page.emulate_media(media="print")                       # on paper every cell is printed, no button
        assert page.evaluate("[...document.querySelectorAll('td.folds')].every(e => e.offsetParent)")
        assert page.evaluate("[...document.querySelectorAll('.row-more')].every(e => !e.offsetParent)")
        assert errors == []
    finally:
        ctx.close()


def test_on_a_wide_screen_the_three_tables_are_unchanged_tables(browser):
    ctx, page, errors = _page(browser, width=1280)
    try:
        _goto(page, "overview")
        for tid in ("t-after", "t-score", "t-map"):
            assert page.evaluate(f"getComputedStyle(document.getElementById('{tid}')).display") == "table"
            assert page.is_visible(f"#{tid} thead")
        assert page.evaluate("[...document.querySelectorAll('td.folds')].every(e => e.offsetParent)")
        assert page.evaluate("[...document.querySelectorAll('.row-more')].every(e => !e.offsetParent)")
        assert errors == []
    finally:
        ctx.close()


def test_one_banner_one_main_and_links_inside_sentences_are_underlined(browser):
    """axe-core 4.10.2 on the overview and three experiments, 375 and 1280 px, found the wordmark,
    the introduction and the closing note outside every landmark, the figures an aside nested in the
    banner, and links in the citations and the foot marked by colour alone (1.12:1 against the grey)."""
    html = PAGE.read_text(encoding="utf-8")
    body = html[html.index("<body>"):]
    assert body.count("<header") == 1 and body.count("<main") == 1 and body.count("<footer") == 1
    assert "<aside" not in body
    assert body.index('<header class="bar"') < body.index("<main") < body.index("<h1>") \
        < body.index('id="lab"') < body.index("</main>") < body.index('<footer id="foot"')
    ctx, page, errors = _page(browser, width=1280)
    try:
        for scene in ("overview", "risk-managed-momentum"):
            _goto(page, scene)
            bare = page.evaluate("""[...document.querySelectorAll('p a, .cite a, .note a, dd a, li a')]
                .filter(a => !a.closest('.next-nav, .pager, .tabs') && a.offsetParent)
                .filter(a => !getComputedStyle(a).textDecorationLine.includes('underline')).map(a => a.textContent)""")
            assert bare == [], (scene, bare)
            outside = page.evaluate("""[...document.querySelectorAll('h1, h2, h3, p, table, .mark')]
                .filter(e => !e.closest('main, header, footer, nav')).map(e => e.tagName + ' ' + e.textContent.slice(0, 40))""")
            assert outside == [], (scene, outside)
        assert errors == []
    finally:
        ctx.close()


def test_the_head_asks_for_the_menu_s_two_files_by_the_script_tags_own_urls():
    html = PAGE.read_text(encoding="utf-8")
    head = html[:html.index("</head>")]
    pre = re.findall(r'<link rel="preload" href="([^"]+)" as="script">', head)
    assert pre == re.findall(r'<script src="([^"]+)"', html) == ["lab_data.js?v=" + str(fl.ASSET_VERSION), "lab_engine.js?v=" + str(fl.ASSET_VERSION)]
    assert head.index('as="script"') < head.index("<style>")


def test_a_mouse_button_on_a_card_asks_for_its_series_before_the_click(browser):
    """The series of an experiment whose data is not in lab_data.js is asked for when a mouse
    button comes down on its card, so the press's own length is not spent waiting; a pointer that
    merely passes over cards asks for nothing, and neither does a finger coming down - on a phone
    that is as often a scroll as a tap, and a scroll past the 153 factors must not fetch them."""
    ctx, page, errors = _page(browser)
    try:
        _goto(page, "overview")
        assert page.evaluate("'size' in LAB_DATA") is False
        # a pointer moved straight across a lazy card and away: no dwell, no request.
        # The over and the out are dispatched in one task, so the dwell really is zero. Two
        # page.mouse.move calls are two round trips, and on a loaded machine the gap between them
        # passed the page's 150 ms dwell: the pointer *had* rested, the page rightly asked for the
        # series, and this failed (2026-10-27, research/2026-10-27-flaky-tests.md). The page's
        # listener reads only mouseover/mouseout here, which is what a real pass fires.
        page.mouse.move(2, 2)
        other = page.query_selector(".card-a[href='#low-volatility']")
        other.scroll_into_view_if_needed()
        other.evaluate("""el => { el.dispatchEvent(new MouseEvent('mouseover', { bubbles: true }));
            el.dispatchEvent(new MouseEvent('mouseout', { bubbles: true })); }""")
        page.wait_for_timeout(400)                            # longer than the 150 ms dwell it never had
        # (each sort is its own part since round three, put into the empty `sorts` lab_data.js carries)
        assert page.evaluate("'low-volatility' in LAB_DATA.sorts || 'sorts_low_volatility' in LAB_DATA") is False
        assert page.evaluate("document.querySelectorAll('script[src^=\"lab_data_\"]').length") == 0
        page.evaluate("""document.querySelector(".card-a[href='#replication-crisis']")
            .dispatchEvent(new PointerEvent('pointerdown', { bubbles: true, pointerType: 'touch' }))""")
        page.wait_for_timeout(400)
        assert page.evaluate("'jkp' in LAB_DATA") is False
        assert page.evaluate("document.querySelectorAll('script[src^=\"lab_data_\"]').length") == 0
        el = page.query_selector(".card-a[href='#size-effect']")
        el.scroll_into_view_if_needed()
        b = el.bounding_box()
        page.mouse.move(b["x"] + 20, b["y"] + 12)
        page.mouse.down()
        page.wait_for_function("'size' in LAB_DATA", timeout=WAIT)      # arrived while the finger is still down
        assert page.evaluate("window.LAB_STATE().id") == "overview"
        page.mouse.up()
        page.wait_for_function("window.LAB_STATE().id === 'size-effect' && document.querySelector('#ch-bars svg rect')", timeout=WAIT)
        assert page.evaluate("document.querySelectorAll('script[src^=\"lab_data_size.js\"]').length") == 1
        assert errors == []
    finally:
        ctx.close()


# --------------------------------------------------------------------------- round three: the 153 factors by universe
# The 153 factors' part carried all three universes (716 KB gzipped) and the page checked all three
# against the build before showing anything: 4.9-5.1 s to open on the phone profile. Each size group is
# now its own part (factor_lab.LAZY_INTO). The experiment opens on all stocks, fetches the mega caps
# because its opening chart draws them, and fetches the large caps only when chosen. A figure whose
# series has not arrived is never dropped and never passed off as checked: it stands as the build's,
# says so, and is re-run when its series lands.
PARTS_ON_PAGE = "[...document.querySelectorAll('script[src^=\"lab_data_\"]')].map(s => s.getAttribute('src').split('?')[0])"
PEND = "[...document.querySelectorAll('#t-ref tr.pend')].map(tr => tr.cells[0].textContent)"


def test_the_153_factors_open_on_all_stocks_and_check_each_group_when_it_arrives(browser):
    ctx, page, errors = _page(browser)
    try:
        _goto(page, "replication-crisis")
        assert page.evaluate(PARTS_ON_PAGE) == ["lab_data_jkp.js", "lab_data_jkp_mega.js"]
        assert page.evaluate("Object.keys(LAB_DATA.jkp.series)") == ["all", "mega"]
        assert page.evaluate("document.querySelectorAll('#t-ref tbody tr').length") == 10        # every figure still listed
        pend = page.evaluate(PEND)
        assert len(pend) == 1 and pend[0].startswith("Large caps only") and "not yet re-run in this browser" in pend[0]
        # the pending row prints the build's own figure, so nothing on the page is a number the page made up
        built = next(r for r in next(x for x in fl.load_published()["experiments"] if x["id"] == "replication-crisis")["result"]["rows"]
                     if r["key"] == "rate_capm_large")
        assert page.evaluate("document.querySelector('#t-ref tr.pend').cells[2].textContent") == f"{built['here']:.2f}"
        said = page.inner_text("#x-checked")
        assert "9 of the 10 figures above were re-run in your browser" in said and "each agrees with the build" in said
        assert "large-cap figure is the build’s until its series loads" in said
        assert page.get_attribute("#x-checked", "role") == "status"
        assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
        assert page.evaluate("document.getElementById('x-check-now').getBoundingClientRect().height") >= 43.5
        page.click("#x-check-now")
        page.wait_for_function("!document.querySelector('#t-ref tr.pend')", timeout=WAIT)
        assert page.evaluate(PARTS_ON_PAGE)[-1] == "lab_data_jkp_large.js"
        assert page.inner_text("#x-checked").startswith("All 10 figures above were re-run in your browser")
        assert page.evaluate("document.getElementById('x-check-now')") is None
        assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
        assert errors == []
    finally:
        ctx.close()


def test_a_link_to_the_large_caps_opens_on_them_and_leaves_the_mega_figure_pending(browser):
    ctx, page, errors = _page(browser, width=1280)
    try:
        page.goto(PAGE.as_uri() + "#replication-crisis?universe=large", wait_until="load", timeout=60000)
        page.wait_for_function("window.LAB_STATE && window.LAB_STATE() && window.LAB_STATE().id === 'replication-crisis'"
                               " && window.LAB_STATE().result && !document.querySelector('#lab [aria-busy=\"true\"]')", timeout=WAIT)
        assert sorted(page.evaluate(PARTS_ON_PAGE)) == ["lab_data_jkp.js", "lab_data_jkp_large.js"]
        assert "among large caps" in page.inner_text("#x-read")
        assert page.evaluate("document.querySelectorAll('#ch-bars rect').length") == 26
        pend = page.evaluate(PEND)
        assert len(pend) == 1 and pend[0].startswith("Mega caps only")
        assert "mega-cap figure is the build’s" in page.inner_text("#x-checked")
        # choosing the mega caps fetches their series, redraws on them, and the last figure is checked
        page.click("#c-universe button[data-v='mega']")
        page.wait_for_function("window.LAB_STATE().result && !document.querySelector('#lab [aria-busy=\"true\"]')", timeout=WAIT)
        assert "among mega caps" in page.inner_text("#x-read")
        assert page.evaluate(PEND) == []
        assert page.evaluate("document.getElementById('x-mismatch').hidden") is True
        assert errors == []
    finally:
        ctx.close()


SABOTAGE_LARGE = SABOTAGE.replace("lab_data_size\\.js", "lab_data_jkp_large\\.js")


def test_a_group_whose_series_fails_to_load_says_so_and_can_be_tried_again(browser):
    assert SABOTAGE_LARGE != SABOTAGE
    ctx, page, errors = _page(browser, init=SABOTAGE_LARGE)
    try:
        _goto(page, "replication-crisis")
        page.click("#c-universe button[data-v='large']")
        page.wait_for_selector("#x-read .warn .retry", timeout=WAIT)
        assert "did not load" in page.inner_text("#x-read") and page.get_attribute("#x-read .warn", "role") == "alert"
        assert "factors hold up" in page.inner_text("#x-title")                               # the experiment stays
        assert "it did not load" in page.inner_text("#x-checked")
        assert len(page.evaluate(PEND)) == 1                                                 # still listed, still pending
        page.click("#x-read .retry")
        page.wait_for_function("window.LAB_STATE().result && !document.querySelector('#lab [aria-busy=\"true\"]')", timeout=WAIT)
        assert "among large caps" in page.inner_text("#x-read")
        assert page.evaluate(PEND) == []
        assert errors == []
    finally:
        ctx.close()


OVERLAPS = """(() => { const b = [...document.querySelectorAll('#ch-bars text')].filter(t => !t.classList.contains('val'))
    .map(t => t.getBoundingClientRect()), hit = [];
  for (let i = 0; i < b.length; i++) for (let j = i + 1; j < b.length; j++)
    if (b[i].left < b[j].right - 1 && b[j].left < b[i].right - 1 && b[i].top < b[j].bottom - 1 && b[j].top < b[i].bottom - 1) hit.push([i, j]);
  return hit.length; })()"""


def test_on_a_phone_the_153_factors_chart_is_a_row_per_theme_and_no_label_runs_into_another(browser):
    """At 375 px the thirteen theme names, set at an angle under twenty-six bars, overlapped."""
    ctx, page, errors = _page(browser)
    try:
        _goto(page, "replication-crisis")
        themes = page.evaluate("LAB_STATE().result.themes.map(t => t.theme)")
        assert page.evaluate("[...document.querySelectorAll('#ch-bars text.rl')].map(t => t.textContent)") == themes
        assert page.evaluate("document.querySelectorAll('#ch-bars rect[data-i]').length") == 26
        assert page.evaluate(OVERLAPS) == 0
        assert page.evaluate("document.querySelector('#ch-bars svg').getBoundingClientRect().width") <= 375
        # read down the rows: ArrowDown steps to the next theme, as ArrowRight does
        page.focus("#ch-bars svg")
        page.keyboard.press("Home")
        page.keyboard.press("ArrowDown")
        page.wait_for_function(_hit(1), timeout=WAIT)
        assert page.inner_text("#ch-bars + .chart-read").startswith(themes[1] + ":")
        assert "up and down" in page.get_attribute("#ch-bars svg", "aria-label")
        assert errors == []
    finally:
        ctx.close()
    ctx, page, errors = _page(browser, width=1280)
    try:
        _goto(page, "replication-crisis")                                   # a wide screen keeps its columns
        assert page.evaluate("document.querySelectorAll('#ch-bars text.rl').length") == 0
        assert page.evaluate("document.querySelectorAll('#ch-bars rect[data-i]').length") == 26
        assert errors == []
    finally:
        ctx.close()


def test_every_card_puts_its_count_under_its_meter_so_the_counts_line_up(browser):
    """Round three: "6 of 6" and "1 of 4" sat beside their short meters, "13 of 14" under its long one."""
    for width in (375, 1280):
        ctx, page, errors = _page(browser, width=width)
        try:
            _goto(page, "overview")
            beside = page.evaluate("""[...document.querySelectorAll('.card .meter')].filter(m => m.offsetParent)
                .filter(m => m.querySelector('.m-l').getBoundingClientRect().top < m.querySelector('svg').getBoundingClientRect().bottom - 1)
                .map(m => m.querySelector('.m-l').textContent)""")
            assert beside == [], (width, beside)
            assert page.evaluate("[...document.querySelectorAll('.card .meter')].filter(m => m.offsetParent).length") == len(fl.EXPERIMENTS)
            assert errors == []
        finally:
            ctx.close()


# A figure that rounds to zero carries no sign (2026-10-29). A sweep of every experiment through
# every setting found "−0.00" in 12 of 17: "Sharpe ratio −0.00", "Momentum averaged −0.00% a
# month", decile tables, and four experiments opened on a "−0.00" difference beside "within".
SIGNED_ZERO = re.compile("[\u2212-]0\\.0+(?![0-9])")


@pytest.mark.parametrize("scene", ["asset-growth", "five-factors", "big-and-small", "replication-crisis"])
def test_an_experiment_opens_without_a_signed_zero(browser, scene):
    ctx, page, errors = _page(browser, width=1280)
    try:
        _goto(page, scene)
        bad = [ln for ln in page.inner_text("#lab").split("\n") if SIGNED_ZERO.search(ln)]
        assert bad == [], (scene, bad[:5])
        assert errors == []
    finally:
        ctx.close()


def test_the_formatters_sign_the_printed_figure_not_the_raw_value(browser):
    src = PAGE.read_text(encoding="utf-8")
    f2 = re.search(r"^\s*const f2 = .*$", src, re.M).group(0)
    sf2 = re.search(r"^\s*const sf2 = .*$", src, re.M).group(0)
    ctx, page, _ = _page(browser)
    try:
        got = page.evaluate(f"""() => {{ {f2} {sf2}
            return [[-0.001, 0, 0.004, -0.006, 1.234, -1.234].map(f2),
                    [-0.001, 0, 0.004, -0.006, 1.234, -1.234].map(sf2)]; }}""")
    finally:
        ctx.close()
    assert got[0] == ["0.00", "0.00", "0.00", "\u22120.01", "1.23", "\u22121.23"]
    assert got[1] == ["0.00", "0.00", "0.00", "\u22120.01", "+1.23", "\u22121.23"]
    assert not any(SIGNED_ZERO.search(s) for s in got[0] + got[1])
