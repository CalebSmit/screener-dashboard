"""Screenshot the options half of 'Before you decide' at 1440 and 375px, and print its sentence.

The rule for any dashboard change is to look at the page rather than judge from the source.
Used 2026-10-12 to check that the options panel names Friday's session after the quote-date
repair, at both widths.

    python research/measurements/2026-10-12-shot-options-panel.py AAPL
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs" / "design"


def main() -> int:
    ticker = (sys.argv[1] if len(sys.argv) > 1 else "AAPL").upper()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch()
        for label, w, h in (("desktop", 1440, 1000), ("mobile", 375, 800)):
            pg = b.new_page(viewport={"width": w, "height": h})
            pg.goto((ROOT / "index.html").as_uri())
            # CTX_LOADED is closure-scoped; the payload itself is the observable signal
            pg.wait_for_function("() => !!window.SCREENER_CONTEXT", timeout=60_000)
            # the first-visit guide is an overlay and will sit on top of the panel
            for sel in (".guide-dismiss", "#guide-got-it", "button:has-text('Got it')"):
                el = pg.query_selector(sel)
                if el:
                    el.click()
                    pg.wait_for_timeout(300)
                    break
            pg.evaluate(f"openStockDetail('{ticker}')")
            pg.wait_for_timeout(800)
            pg.evaluate("goToModal('section-context')")
            pg.wait_for_timeout(800)
            # the options block is the one whose copy mentions the expected move
            box = pg.query_selector("#section-context")
            if box is None:
                print(f"{label}: #section-context not found")
                continue
            # shoot the options sub-block, so the image is the thing that changed
            opt = pg.query_selector("#section-context .ctx-block:has-text('Expected move')") or box
            opt.scroll_into_view_if_needed()
            pg.wait_for_timeout(300)
            opt.screenshot(path=str(OUT / f"2026-10-12-options-{ticker}-{label}.png"))
            text = box.inner_text()
            print(f"\n--- {label} {w}x{h} ---")
            m = re.search(r"Expected move.*", text)
            if m is None:
                print("    (no expected-move copy on this stock)")
            else:
                for chunk in re.findall(r".{1,105}(?:\s|$)", m.group(0)):
                    if chunk.strip():
                        print("    ", chunk.rstrip())
            pg.close()
        b.close()
    print(f"\nwrote {OUT}/2026-10-12-options-{ticker}-*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
