"""The dashboard page's UI elevation (2026-10-27, ``plan/ui-elevation.md``), in a real browser.

Two things this pass made true that nothing else checks:

1. **An open section shows all of its content.** The collapse rule capped every open section body
   at 5,000px with ``overflow: hidden``. Reporting soon's 7-day list is 8,857px tall and its 14-day
   list 12,874px (measured 2026-10-27), so about a hundred companies - and every later day - could
   not be reached. Nothing failed; the list simply ended. This asserts the body is as tall as its
   content, for the longest list on the page.

2. **The ranking picture is the published ranking.** One dot per scored company, drawn from
   ``table_data``'s own Composite; hovering a dot names that company with its published rank and
   composite, and selecting it opens that company's sheet. A picture that re-derived or dropped a
   number would be the defect row 0.8 of CLAUDE.md forbids.

Skips loudly where Playwright or a browser is not installed, like ``test_dashboard_browser.py``.
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


def _open(browser, width=1440, height=900):
    ctx = browser.new_context(viewport={"width": width, "height": height})
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.goto(PAGE.as_uri(), wait_until="load", timeout=60000)
    page.wait_for_function("document.querySelectorAll('#universe-tbody tr[data-t]').length > 0", timeout=30000)
    return ctx, page, errors


@pytest.mark.parametrize("width", [1440, 375])
def test_an_open_section_is_not_cut_off(browser, width):
    ctx, page, errors = _open(browser, width, 900)
    try:
        page.wait_for_function("window.SCREENER_CONTEXT !== undefined", timeout=30000)
        page.evaluate("goToSection('sec-reporting')")
        page.wait_for_timeout(400)
        for days in (7, 14):
            page.evaluate(f"setReporting('days', {days})")
            page.wait_for_timeout(250)
            m = page.evaluate("""() => { const b = document.getElementById('sec-reporting-body');
                return { scroll: b.scrollHeight, client: b.clientHeight,
                         rows: b.querySelectorAll('tr.rp-row').length }; }""")
            assert m["rows"] > 0, m
            assert m["client"] >= m["scroll"] - 1, (
                f"{days}-day list: {m['scroll']}px of content in a {m['client']}px body - the rest is unreachable")
        assert errors == []
    finally:
        ctx.close()


@pytest.mark.parametrize("width", [1440, 375])
def test_the_ranking_picture_is_one_dot_per_published_composite(browser, width):
    ctx, page, errors = _open(browser, width, 900)
    try:
        info = page.evaluate("""() => {
            const scored = D.table_data.filter(r => typeof r.Composite === 'number');
            const placed = [].concat(...DIST.cols);
            const same = placed.length === scored.length && scored.every(r => placed.indexOf(r) !== -1);
            // every dot sits in the bin its own published composite falls in
            const binned = DIST.cols.every((c, i) => c.every(r => {
                const lo = DIST.lo + i * DIST.bw, hi = lo + DIST.bw;
                return r.Composite >= lo - 1e-9 && (r.Composite < hi + 1e-9 || i === DIST.cols.length - 1);
            }));
            return { n: scored.length, placed: placed.length, same: same, binned: binned,
                     svgs: document.querySelectorAll('#dist-plot svg').length,
                     nodes: document.querySelectorAll('#dist-plot *').length };
        }""")
        assert info["same"] and info["binned"], info
        assert info["svgs"] == 1 and info["nodes"] < 40, info  # one picture, not one node per company
        # hover the published #1 and select it
        pos = page.evaluate("""() => { const host = document.getElementById('dist-plot');
            host.scrollIntoView({block: 'center'});
            const b = host.getBoundingClientRect(); let at = null;
            DIST.cols.forEach((c, i) => c.forEach((r, k) => { if (r.Rank === 1) at = {i: i, k: k, t: r.Ticker, c: r.Composite}; }));
            return { x: b.left + DIST.pl + DIST.x0 + at.i * DIST.pitch + DIST.pitch / 2,
                     y: b.top + DIST.pt + DIST.base - at.k * DIST.pitch - DIST.pitch / 2, t: at.t, c: at.c }; }""")
        page.mouse.move(pos["x"], pos["y"])
        page.wait_for_timeout(150)
        tip = page.evaluate("DIST.tip.hidden ? '' : DIST.tip.textContent")
        assert pos["t"] in tip and "#1" in tip and f"{pos['c']:.1f}" in tip, tip
        page.mouse.click(pos["x"], pos["y"])
        page.wait_for_selector("#stock-modal .modal-body", state="visible")
        assert pos["t"] in page.inner_text("#modal-ticker")
        assert errors == []
    finally:
        ctx.close()


def test_leader_cards_print_every_category_score(browser):
    """The fingerprint rows print the payload's own score, rounded, for all eight categories."""
    ctx, page, errors = _open(browser)
    try:
        bad = page.evaluate("""() => {
            const keys = ['valuation_score','quality_score','growth_score','momentum_score','risk_score',
                          'revisions_score','size_score','investment_score'];
            const out = [];
            document.querySelectorAll('#top5-row .top5-card').forEach(card => {
                const t = card.querySelector('.top5-ticker').textContent.trim();
                const row = D.table_data.find(r => r.Ticker === t);
                const vals = [...card.querySelectorAll('.fp-v')].map(e => e.textContent.trim());
                if (vals.length !== 8) out.push(t + ': ' + vals.length + ' rows');
                keys.forEach((k, j) => { const want = row[k] == null ? '—' : String(Math.round(row[k]));
                    if (vals[j] !== want) out.push(t + ' ' + k + ' ' + vals[j] + ' != ' + want); });
            });
            return out; }""")
        assert bad == []
        assert errors == []
    finally:
        ctx.close()
