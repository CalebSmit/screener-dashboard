"""Render og-card.png, the image a link to the site shows when it is pasted into a message or a deck.

WHY: the owner wants to show this to people (2026-10-15); a bare link previews as text. The card
carries the product's statement only - no ranking, score or return - so it can never go stale or say
something the page no longer says. Rerun only if the statement changes:

    python scripts/make_og_card.py

Needs Playwright's Chromium (the test suite already uses it). Writes og-card.png (1200x630) at the
repository root, where GitHub Pages serves it.
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "og-card.png"

CARD = """<!doctype html><html><head><meta charset="utf-8">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  html, body { margin: 0; width: 1200px; height: 630px; background: #0d0d0d; }
  body { font-family: Inter, system-ui, sans-serif; color: #fff; display: flex; flex-direction: column;
         justify-content: space-between; padding: 72px 84px; box-sizing: border-box; }
  .brand { display: flex; align-items: center; gap: 16px; font-size: 26px; font-weight: 600; letter-spacing: -.01em; }
  .mark { width: 48px; height: 48px; }
  h1 { margin: 0; font-size: 76px; line-height: 1.04; letter-spacing: -.035em; font-weight: 700; max-width: 900px; }
  p { margin: 22px 0 0; font-size: 28px; line-height: 1.45; color: #c3c2b7; max-width: 900px; }
  .foot { display: flex; gap: 34px; font-size: 21px; color: #8e8c86; }
  .foot b { color: #5598e7; font-weight: 500; }
</style></head><body>
  <div class="brand"><svg class="mark" viewBox="0 0 32 32"><rect width="32" height="32" rx="8" fill="#1a1a19"/>
    <rect x="7" y="17" width="4" height="8" rx="1.5" fill="#3987e5" opacity=".55"/><rect x="14" y="12" width="4" height="13" rx="1.5" fill="#3987e5" opacity=".8"/>
    <rect x="21" y="7" width="4" height="18" rx="1.5" fill="#3987e5"/></svg>Multi-Factor Screener</div>
  <div><h1>The S&amp;P 500, ranked on eight factors</h1>
    <p>Every company scored against its own sector, with every number behind every score one click away.</p></div>
  <div class="foot"><span><b>Rankings</b> every weekday</span><span><b>Tested</b> on 2012&ndash;2026 history</span><span><b>Research</b> rerun in the Factor Lab</span></div>
</body></html>"""


def main() -> int:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 630})
        page.set_content(CARD, wait_until="networkidle")
        page.screenshot(path=str(OUT))
        browser.close()
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
