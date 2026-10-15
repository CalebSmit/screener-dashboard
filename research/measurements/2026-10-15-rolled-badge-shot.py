"""Look at the two provenance sentences on a real page, at 1440 and 375px.

Both flags only enter the payload after a fresh fetch, so a committed page cannot carry them.
This loads a page, sets BRK-B's `ltm_rolled` and DD's `q_rejected` to what the next data run
will publish, opens each drilldown, and reports and photographs the Data Provenance section.

The second half of the check is why this exists: `renderFlagsWarnings` turned out to be dead
code - never called, no container - so a badge added to it could never appear. The sentences
live in `renderProvenance` now, and this is the look that confirms it.

Usage:  python research/measurements/2026-10-15-rolled-badge-shot.py [path-to-dashboard.html]
Writes logs/design/provenance-{desktop,mobile}.png (gitignored).
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "logs" / "design"
OUT.mkdir(parents=True, exist_ok=True)

LABELS = ["Capital Expenditure", "EBIT", "Net Income", "Operating Cash Flow", "Total Revenue"]
Q_WHY = "quarterly revenue of -1.871bn for 2025-12-31 - a derived quarter, not a filed one"

INJECT = """
([labels, qwhy]) => {
  const sd = window.SCREENER_DATA.stock_detail;
  if (sd['BRK-B']) { sd['BRK-B'].flags.ltm_rolled = true;
                     sd['BRK-B'].flags.ltm_rolled_labels = labels; }
  if (sd['DD']) { sd['DD'].flags.q_rejected = qwhy; }
  return {brk: !!sd['BRK-B'], dd: !!sd['DD']};
}
"""

READ = """
(id) => {
  const el = document.getElementById(id);
  if (!el) return {found: false};
  const r = el.getBoundingClientRect();
  return {found: true, text: el.innerText.replace(/\\s+/g, ' ').trim(),
          overflow_px: Math.round(r.right - document.documentElement.clientWidth),
          lines: Math.round(r.height / parseFloat(getComputedStyle(el).lineHeight || 20))};
}
"""


def main():
    page_path = (Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else ROOT / "index.html")
    with sync_playwright() as p:
        b = p.chromium.launch()
        for label, w, h in (("desktop", 1440, 900), ("mobile", 375, 812)):
            pg = b.new_page(viewport={"width": w, "height": h})
            errors: list[str] = []
            pg.on("pageerror", lambda e: errors.append(str(e)))
            pg.goto(page_path.as_uri())
            pg.wait_for_function("() => window.SCREENER_DATA && window.SCREENER_DATA.stock_detail")
            print(f"\n--- {label} {w}x{h}  ({page_path.name})")
            print("   ", pg.evaluate(INJECT, [LABELS, Q_WHY]))
            for ticker, eid in (("BRK-B", "provenance-ltm-rolled"), ("DD", "provenance-q-rejected")):
                pg.evaluate("(t) => openStockDetail(t)", ticker)
                pg.wait_for_timeout(500)
                # open the section if it is closed; a second click would close it again
                pg.evaluate("""(id) => {
                    const el = document.getElementById(id);
                    if (el && el.getBoundingClientRect().height > 0) return;
                    const h = [...document.querySelectorAll('.collapsible-header, button, h3')]
                      .find(e => e.textContent.trim().startsWith('Data Provenance'));
                    if (h) h.click();
                }""", eid)
                pg.wait_for_timeout(400)
                got = pg.evaluate(READ, eid)
                print(f"    {ticker:<6} {eid}: {got}")
                if got.get("found"):
                    pg.evaluate("(id) => document.getElementById(id)"
                                ".scrollIntoView({block: 'center'})", eid)
                    pg.wait_for_timeout(250)
                    pg.screenshot(path=str(OUT / f"provenance-{ticker}-{label}.png"))
                pg.evaluate("() => (typeof closeModal === 'function') && closeModal()")
                pg.wait_for_timeout(200)
            print("    page errors:", errors or "none")
            pg.close()
        b.close()
    print(f"\nscreenshots in {OUT}")


if __name__ == "__main__":
    main()
