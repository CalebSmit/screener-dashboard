"""The Factor Lab's overview on a phone (2026-11-22, owner: "there is too much info and it got way too
long and hard to know where to go").

At 375 px the overview was 14,249 px: the introduction, a card per experiment at about 213 px each,
then three tables and a guide read one after another (about 6,500 px between them). Now the
introduction is set tighter, each card is a row with "What it found" beside its count, and the four
sections after the cards fold under their headings - each heading and the sentence introducing it
left in place, the table behind a tap. What these tests hold:

* the overview is short on load, with every experiment still on it;
* the sentences that qualify the lab are on the page without a tap: "not investment advice", how
  many results were weaker after their paper, and that the lab sets no weight;
* nothing is removed - a tap on a heading, or an address naming it, opens its fold; paper shows all;
* a wide screen is left as it was written.

The bounds are per card and for the page's fixed parts, never one ceiling on a list that grows by an
experiment at a time (``test_lab_reading_and_access.py`` says why). Every wait is on a condition.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import factor_lab as fl  # noqa: E402

PAGE = ROOT / "lab.html"
WAIT = 8000
FOLDS = ["o-after", "o-score", "o-map", "o-how"]
READY = "window.LAB_STATE && window.LAB_STATE() && window.LAB_STATE().id === 'overview'"
PROBE = "document.querySelectorAll('.sec-body, .sec-t, section.fold, section.shut').length"


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


def _open(browser, width=375, height=812, hash_="#overview"):
    ctx = browser.new_context(viewport={"width": width, "height": height})
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri() + hash_, wait_until="load", timeout=60000)
    page.wait_for_function(READY, timeout=WAIT)
    return ctx, page, errors


def _shut(page) -> list[str]:
    return page.evaluate("[...document.querySelectorAll('section.fold.shut > h3')].map(h => h.id)")


@pytest.mark.parametrize("width", [375, 320])
def test_on_a_phone_the_overview_is_short_with_every_experiment_on_it(browser, width):
    ctx, page, errors = _open(browser, width)
    try:
        page.wait_for_function("document.querySelectorAll('section.fold.shut').length === 4", timeout=WAIT)
        assert _shut(page) == FOLDS
        n = len(fl.EXPERIMENTS)
        assert page.evaluate("[...document.querySelectorAll('.card .card-a strong')].filter(e => e.offsetParent).length") == n
        r = page.evaluate("""() => { const cards = [...document.querySelectorAll('.card')].map(c => c.offsetHeight);
            return { page: document.documentElement.scrollHeight, cards: cards.reduce((a, b) => a + b, 0),
                     hero: document.querySelector('.hero').offsetHeight,
                     folds: [...document.querySelectorAll('section.fold')].reduce((a, s) => a + s.offsetHeight, 0),
                     sideways: document.documentElement.scrollWidth > document.documentElement.clientWidth }; }""")
        assert not r["sideways"], r
        # per card where the list grows, a ceiling only on what does not: 146 px a card at 375 px
        # (188 at 320, where the button sits under the count), 620 px of introduction, 700 of folds
        assert r["cards"] / n < (175 if width == 375 else 215), (r, n)
        assert r["hero"] < (700 if width == 375 else 820), r
        assert r["folds"] < 900, r
        assert r["page"] - r["cards"] < 3600, r                 # everything but the cards; it was 10,200
        # each fold's heading is a button of a comfortable size that says it is closed
        rows = page.evaluate("""[...document.querySelectorAll('section.fold > h3 > button.sec-t')].map(b =>
            [b.getBoundingClientRect().height, b.getAttribute('aria-expanded'), !!document.getElementById(b.getAttribute('aria-controls'))])""")
        assert len(rows) == 4 and all(h >= 44 and e == "false" and c for h, e, c in rows), rows
        assert errors == []
    finally:
        ctx.close()


def test_on_a_phone_what_it_found_sits_beside_the_count_and_never_on_it(browser):
    """From 372 px the button shares the count's line; below that it goes under it. Either way the two
    never overlap, and the button keeps its 44 px."""
    for width in (320, 360, 372, 375, 390):
        ctx, page, errors = _open(browser, width)
        try:
            page.wait_for_function("document.querySelectorAll('.card .card-more').length > 0", timeout=WAIT)
            rows = page.evaluate("""[...document.querySelectorAll('.card')].map(c => {
                const g = document.createRange(); g.selectNodeContents(c.querySelector('.m-l'));
                const l = g.getBoundingClientRect(), b = c.querySelector('.card-more').getBoundingClientRect();
                const sameLine = b.top < l.bottom && l.top < b.bottom;
                return { sameLine, gap: sameLine ? b.left - l.right : b.top - l.bottom, h: b.height,
                         inside: b.right <= c.getBoundingClientRect().right && b.bottom <= c.getBoundingClientRect().bottom + 1 }; })""")
            assert len(rows) == len(fl.EXPERIMENTS)
            assert all(r["h"] >= 44 and r["inside"] for r in rows), (width, rows)
            assert all(r["sameLine"] for r in rows) if width >= 372 else not any(r["sameLine"] for r in rows), (width, rows)
            assert min(r["gap"] for r in rows) >= (8 if width >= 372 else 0), (width, rows)
            assert errors == []
        finally:
            ctx.close()


def test_on_a_phone_the_sentences_that_qualify_the_lab_need_no_tap(browser):
    ctx, page, _ = _open(browser)
    try:
        page.wait_for_function("document.querySelectorAll('section.fold.shut').length === 4", timeout=WAIT)
        shown = page.evaluate("document.body.innerText").replace("’", "'")
        for words in ("A teaching tool, not investment advice.", "were weaker once their paper's sample ended",
                      "The lab does not set the screener's weights", "Not investment advice."):
            assert words in shown, words
        # the introducing sentence is still the heading's next sibling, in view
        assert page.evaluate("document.querySelector('#o-after + p').getBoundingClientRect().height") > 0
        assert page.evaluate("document.querySelector('#o-score + p').getBoundingClientRect().height") > 0
    finally:
        ctx.close()


def test_a_tap_or_an_address_opens_a_fold_and_nothing_is_removed(browser):
    wide_ctx, wide, _ = _open(browser, 1280, 900)
    ctx, page, errors = _open(browser)
    try:
        page.wait_for_function("document.querySelectorAll('section.fold.shut').length === 4", timeout=WAIT)
        text = "document.getElementById('lab').textContent.replace(/\\s+/g, '')"
        assert page.evaluate(text) == wide.evaluate(text)                   # every word, folded or not
        n_after = page.evaluate("LAB_DATA.after_the_paper.length")
        assert page.evaluate("document.querySelectorAll('#o-after-body #t-after tbody tr').length") == n_after
        assert page.evaluate("document.getElementById('o-after-body').offsetHeight") == 0
        page.click("#o-after .sec-t")
        page.wait_for_function("document.getElementById('t-after').offsetHeight > 500", timeout=WAIT)
        assert page.get_attribute("#o-after .sec-t", "aria-expanded") == "true"
        assert page.evaluate("window.LAB_STATE().id") == "overview"
        assert page.evaluate("document.documentElement.scrollWidth <= document.documentElement.clientWidth")
        page.click("#o-after .sec-t")
        page.wait_for_function("document.getElementById('o-after-body').offsetHeight === 0", timeout=WAIT)
        # an address that names a section opens it, and the scene's own address is kept
        page.evaluate("location.hash = '#o-map'")
        page.wait_for_function("!document.getElementById('o-map').closest('section').classList.contains('shut')", timeout=WAIT)
        page.wait_for_function("document.getElementById('t-map').offsetHeight > 200", timeout=WAIT)
        assert page.evaluate("window.LAB_STATE().id") == "overview"
        assert _shut(page) == ["o-after", "o-score", "o-how"]
        page.emulate_media(media="print")                                   # paper shows every fold
        assert page.evaluate("[...document.querySelectorAll('.sec-body')].filter(e => getComputedStyle(e).display === 'none').length") == 0
        assert errors == []
    finally:
        ctx.close()
        wide_ctx.close()


def test_on_a_wide_screen_nothing_is_folded_and_a_widened_window_gets_the_sections_back(browser):
    ctx, page, errors = _open(browser, 1440, 900)
    try:
        page.wait_for_selector("#t-map tbody tr", timeout=WAIT)
        assert page.evaluate(PROBE) == 0
        for tid in ("t-after", "t-score", "t-map"):
            assert page.evaluate(f"document.getElementById('{tid}').offsetHeight") > 200, tid
        wide_html = page.evaluate("document.getElementById('lab').innerHTML")
        page.set_viewport_size({"width": 375, "height": 812})
        page.wait_for_function("document.querySelectorAll('section.fold.shut').length === 4", timeout=WAIT)
        page.set_viewport_size({"width": 1440, "height": 900})
        page.wait_for_function(PROBE + " === 0", timeout=WAIT)
        assert page.evaluate("document.getElementById('lab').innerHTML") == wide_html
        assert errors == []
    finally:
        ctx.close()
