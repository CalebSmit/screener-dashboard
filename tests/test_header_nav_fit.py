"""The header's section links are never cut off (owner, 2026-10-27).

The owner photographed the live page on a wide monitor with "Market" clipped to "Marl" and every
later link gone: the bar carried the brand, a "Last updated" line, nine links and a labelled search,
and its link strip was a scroll box with its scrollbar hidden. Measured that day on the generator's
own page: links were clipped at 1024, 1152, 1536, 1680 and 1920 px wide - five of eight widths.

The fix is the standard "priority plus" bar: every link is shown in full or listed, in order, under
"More". These tests open the committed page (which `tests/test_rebuild_page.py` holds equal to the
generator's output) at phone, tablet, laptop and wide widths and assert, for each:

- no visible link in the bar is clipped by any ancestor or by the viewport;
- every section link is in exactly one place - the bar or the More menu - in the original order;
- the More menu opens, lists exactly the hidden links, and choosing one closes it.

Skips loudly where Playwright or a browser is not installed, like the other browser modules.
"""

from __future__ import annotations

from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api", reason="playwright not installed")

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "index.html"

pytestmark = pytest.mark.skipif(not PAGE.exists(), reason="generated dashboard not present")

WIDTHS = (375, 768, 1024, 1280, 1536, 1680, 1920)

STATE = """() => {
  const nav = document.querySelector('.header-nav');
  const links = [...nav.querySelectorAll(':scope > a')].filter(a => a.style.display !== 'none');
  const shown = links.filter(a => !a.classList.contains('nav-hidden'));
  const clipped = shown.filter(a => {
    const b = a.getBoundingClientRect();
    if (b.right > innerWidth + 1 || b.left < -1) return true;
    for (let p = a.parentElement; p && p !== document.body; p = p.parentElement) {
      if (getComputedStyle(p).overflowX !== 'visible') {
        const pb = p.getBoundingClientRect();
        if (b.right > pb.right + 1 || b.left < pb.left - 1) return true;
      }
    }
    return false;
  }).map(a => a.textContent.trim());
  const menu = [...document.querySelectorAll('#nav-more-menu a')].map(a => a.textContent.trim());
  return {
    all: links.map(a => a.textContent.trim()),
    shown: shown.map(a => a.textContent.trim()),
    clipped, menu,
    moreShown: !document.getElementById('nav-more').hidden,
  };
}"""


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        try:
            b = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"no usable chromium: {exc}")
        yield b
        b.close()


def _settled(browser, width):
    ctx = browser.new_context(viewport={"width": width, "height": 900})
    page = ctx.new_page()
    page.goto(PAGE.as_uri())
    page.wait_for_function("window.SCREENER_DATA && document.querySelector('.header-nav')", timeout=60000)
    page.evaluate("document.fonts ? document.fonts.ready : null")
    page.wait_for_timeout(400)
    return ctx, page


@pytest.mark.parametrize("width", WIDTHS)
def test_every_section_link_is_whole_in_the_bar_or_listed_under_more(browser, width):
    ctx, page = _settled(browser, width)
    try:
        s = page.evaluate(STATE)
        assert s["clipped"] == [], f"{width}px: links cut off in the header: {s['clipped']}"
        assert s["shown"] + s["menu"] == s["all"], (
            f"{width}px: bar {s['shown']} + More {s['menu']} is not every link in order {s['all']}")
        assert s["moreShown"] == bool(s["menu"]), f"{width}px: More is shown={s['moreShown']} with {s['menu']}"
    finally:
        ctx.close()


def test_the_bar_refits_when_the_window_is_resized(browser):
    ctx, page = _settled(browser, 1920)
    try:
        assert page.evaluate(STATE)["menu"] == []
        page.set_viewport_size({"width": 1024, "height": 900})
        page.wait_for_timeout(300)
        s = page.evaluate(STATE)
        assert s["clipped"] == [] and s["menu"], f"after resizing to 1024px: {s}"
        assert s["shown"] + s["menu"] == s["all"]
    finally:
        ctx.close()


def test_more_opens_lists_the_hidden_links_and_closes_on_choice(browser):
    ctx, page = _settled(browser, 1024)
    try:
        s = page.evaluate(STATE)
        assert s["menu"], "1024px should need the More menu"
        page.click("#nav-more-btn")
        assert page.evaluate("!document.getElementById('nav-more-menu').hidden")
        assert page.get_attribute("#nav-more-btn", "aria-expanded") == "true"
        page.click(f"#nav-more-menu a:has-text('{s['menu'][-1]}')")
        assert page.evaluate("document.getElementById('nav-more-menu').hidden")
        page.click("#nav-more-btn")
        page.keyboard.press("Escape")
        assert page.evaluate("document.getElementById('nav-more-menu').hidden")
    finally:
        ctx.close()
