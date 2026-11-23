"""The research page on a phone (2026-11-22, owner: "there is too much info and it got way too long
and hard to know where to go").

At 375 px ``research.html`` was 15,518 px: the answer, then eight sections read one after another.
On a screen 760 px wide or narrower the answer keeps its headline and its plain-terms sentence with
the statistics and the chart one tap under it, and six of the eight sections fold to their headings
(``research_page.PHONE_CSS`` / ``PHONE_JS``). What these tests hold:

* the page is short on load and the answer is on the first screen;
* **the caveats are never folded** - "What this means", "Limits of the test" and the footer's "not
  investment advice" are on the page without a tap;
* nothing is removed: every folded part is in the document, and a tap on its heading, a contents
  link, or an address ending in its ``#id`` opens it;
* paper shows everything;
* a wide screen is left exactly as the generator wrote it.

Every wait is on a condition, never a fixed sleep. The page under test is the generator's own output
written to a temporary file, so a stale committed ``research.html`` is ``test_research_page.py``'s
failure and not this module's.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import research_page as rp  # noqa: E402
import v2_experiments as X  # noqa: E402

WAIT = 8000
PHONE_BUDGET = 5000          # px at 375 wide on load; it was 15,518 before and is about 3,500 now
FOLDED = ["how-tested", "alternatives", "best-backtest", "trading-rule", "how-long", "screen-and-horizon"]
ALWAYS_OPEN = ["what-it-means", "limits"]
ARMED = "document.querySelectorAll('main > section.fold').length > 0"


@pytest.fixture(scope="module")
def page_url(tmp_path_factory):
    latest = {}
    for line in X.LEDGER.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            latest[r["id"]] = r
    reg = X.load_registry()
    out = tmp_path_factory.mktemp("research_phone") / "research.html"
    out.write_text(rp.build(reg, latest, X.summary_for_page(reg, latest), X.detectable, X.load_evaluations()),
                   encoding="utf-8")
    return out.as_uri()


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


def _open(browser, url, width=375, height=812):
    ctx = browser.new_context(viewport={"width": width, "height": height})
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(url, wait_until="load", timeout=60000)
    page.wait_for_selector("main > section#limits", timeout=WAIT)
    return ctx, page, errors


def _shut(page) -> list[str]:
    return page.evaluate("[...document.querySelectorAll('main > section.shut')].map(s => s.id)")


def test_the_sections_the_page_folds_are_the_sections_it_has():
    """The fold list is read off the page, so a section added to the generator is folded too - and
    this module's two lists must then be brought level with it."""
    import re
    html = rp.PAGE.read_text(encoding="utf-8")
    assert re.findall(r'<section id="([a-z-]+)">', html) == FOLDED + ALWAYS_OPEN
    assert "KEEP = { 'what-it-means': 1, 'limits': 1 }" in rp.PHONE_JS


@pytest.mark.parametrize("width", [375, 320])
def test_on_a_phone_the_page_is_short_and_the_answer_is_on_the_first_screen(browser, page_url, width):
    ctx, page, errors = _open(browser, page_url, width, 812 if width == 375 else 700)
    try:
        page.wait_for_function(ARMED, timeout=WAIT)
        assert _shut(page) == FOLDED
        height = page.evaluate("document.documentElement.scrollHeight")
        assert height < PHONE_BUDGET, height
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        if width == 375:
            # the verdict and its plain-terms sentence, whole, before any scrolling
            r = page.evaluate("""() => { const li = document.querySelector('.answer > li'), s = li.querySelector('strong'),
                p = li.querySelector('.plain') || s; return [s.textContent, s.getBoundingClientRect().top, p.getBoundingClientRect().bottom]; }""")
            assert r[0] in ("Not measurably.", "Not reliably.") and 0 < r[1] < r[2] <= 812, r
        # each fold is a heading that is a button of a comfortable size, saying whether it is open
        rows = page.evaluate("""[...document.querySelectorAll('main > section.fold > h2 > button.sec-t')].map(b =>
            [b.getBoundingClientRect().height, b.getAttribute('aria-expanded'), !!document.getElementById(b.getAttribute('aria-controls'))])""")
        assert len(rows) == len(FOLDED) and all(h >= 44 and e == "false" and c for h, e, c in rows), rows
        assert page.evaluate("document.querySelector('.ans-t').getBoundingClientRect().height") >= 40
        assert errors == []
    finally:
        ctx.close()


def test_on_a_phone_the_caveats_are_on_the_page_without_a_tap(browser, page_url):
    ctx, page, _ = _open(browser, page_url)
    try:
        page.wait_for_function(ARMED, timeout=WAIT)
        for sid in ALWAYS_OPEN:
            r = page.evaluate(f"""() => {{ const s = document.getElementById({sid!r});
                return [s.classList.contains('fold'), !!s.querySelector('button'), s.getBoundingClientRect().height,
                        [...s.querySelectorAll('p, li')].every(e => e.getBoundingClientRect().height > 0)]; }}""")
            assert r[0] is False and r[1] is False and r[2] > 150 and r[3] is True, (sid, r)
        shown = page.evaluate("""[...document.querySelectorAll('main, footer')].map(e => e.innerText).join(' ')""")
        for words in ("Not supported", "It is not a forecast.", "It is not investment advice.",
                      "No alternative did better."):
            assert words.lower() in shown.lower(), words
    finally:
        ctx.close()


def test_nothing_is_removed_and_a_tap_opens_and_closes_a_fold(browser, page_url):
    """The folded page holds every word the wide page shows."""
    wide_ctx, wide, _ = _open(browser, page_url, 1280, 900)
    ctx, page, errors = _open(browser, page_url)
    try:
        page.wait_for_function(ARMED, timeout=WAIT)
        text = "document.querySelector('main').textContent.replace(/\\s+/g, ' ')"
        added = "The figures behind this"
        assert page.evaluate(text).replace(added, "").replace(" ", "") == wide.evaluate(text).replace(" ", "")
        # the answer's statistics and chart: one tap
        assert page.evaluate("document.querySelector('.ans-more').hidden") is True
        assert page.evaluate("document.querySelector('.ans-more figure.fp') !== null")
        page.click(".ans-t")
        page.wait_for_function("document.querySelector('.ans-more figure.fp').getBoundingClientRect().height > 100", timeout=WAIT)
        assert page.get_attribute(".ans-t", "aria-expanded") == "true"
        # a section: tap its heading to open it, again to close it
        page.click("#trading-rule .sec-t")
        page.wait_for_function("document.getElementById('trading-rule-body').getBoundingClientRect().height > 500", timeout=WAIT)
        assert page.get_attribute("#trading-rule .sec-t", "aria-expanded") == "true"
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        page.click("#trading-rule .sec-t")
        page.wait_for_function("document.getElementById('trading-rule-body').getBoundingClientRect().height === 0", timeout=WAIT)
        assert errors == []
    finally:
        ctx.close()
        wide_ctx.close()


def test_a_contents_link_and_an_address_open_the_section_they_name(browser, page_url):
    ctx, page, _ = _open(browser, page_url)
    try:
        page.wait_for_function(ARMED, timeout=WAIT)
        page.click(".toc a[href='#how-long']")
        page.wait_for_function("!document.getElementById('how-long').classList.contains('shut')", timeout=WAIT)
        page.wait_for_function("Math.abs(document.getElementById('how-long').getBoundingClientRect().top) < 200", timeout=WAIT)
        assert _shut(page) == [s for s in FOLDED if s != "how-long"]
        # typed into the address bar on the open page
        page.evaluate("location.hash = '#best-backtest'")
        page.wait_for_function("!document.getElementById('best-backtest').classList.contains('shut')", timeout=WAIT)
    finally:
        ctx.close()
    # and arriving on it from elsewhere
    ctx, page, _ = _open(browser, page_url + "#screen-and-horizon")
    try:
        page.wait_for_function(ARMED, timeout=WAIT)
        assert _shut(page) == [s for s in FOLDED if s != "screen-and-horizon"]
        page.wait_for_function("Math.abs(document.getElementById('screen-and-horizon').getBoundingClientRect().top) < 200", timeout=WAIT)
    finally:
        ctx.close()


def test_paper_shows_every_fold(browser, page_url):
    ctx, page, _ = _open(browser, page_url)
    try:
        page.wait_for_function(ARMED, timeout=WAIT)
        page.emulate_media(media="print")
        hidden = page.evaluate("""[...document.querySelectorAll('.sec-body, .ans-more')]
            .filter(e => getComputedStyle(e).display === 'none').map(e => e.id)""")
        assert hidden == []
        assert page.evaluate("getComputedStyle(document.querySelector('.ans-t')).display") == "none"
    finally:
        ctx.close()


def test_on_a_wide_screen_nothing_is_folded_or_rearranged(browser, page_url):
    """At 1440 px the script touches nothing; and a phone turned to a wide window gets the page back."""
    probe = "document.querySelectorAll('.sec-body, .sec-t, .ans-more, .ans-t, .fold, .shut').length"
    ctx, page, errors = _open(browser, page_url, 1440, 900)
    try:
        assert page.evaluate(probe) == 0
        heights = page.evaluate("[...document.querySelectorAll('main > section')].map(s => s.getBoundingClientRect().height)")
        assert len(heights) == len(FOLDED) + len(ALWAYS_OPEN) and min(heights) > 150, heights
        wide_html = page.evaluate("document.querySelector('main').innerHTML")
        page.set_viewport_size({"width": 375, "height": 812})
        page.wait_for_function(ARMED, timeout=WAIT)
        page.set_viewport_size({"width": 1440, "height": 900})
        page.wait_for_function(probe + " === 0", timeout=WAIT)
        assert page.evaluate("document.querySelector('main').innerHTML") == wide_html
        assert errors == []
    finally:
        ctx.close()
