"""Screenshot the two panels this morning's fix changed, at 1440 and 375.

NVR's analyst-target card (the signed zero that turned gate 1 red) and ESS's "Before you decide"
trend line (the unsigned zero that was still live in `cPct`). Writes into the gitignored
logs/design/ and prints the text of each, so the figure is in the log as well as the picture.

    python -X utf8 research/measurements/2026-11-03-shot-fixed-cards.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "logs" / "design"
OUT.mkdir(parents=True, exist_ok=True)

TARGETS = [("NVR", "#modal-price-targets"), ("ESS", "#section-context")]


def main() -> int:
    with sync_playwright() as p:
        b = p.chromium.launch()
        for label, w, h in (("desktop", 1440, 900), ("mobile", 375, 812)):
            pg = b.new_page(viewport={"width": w, "height": h})
            pg.goto((ROOT / "index.html").as_uri(), wait_until="load", timeout=60000)
            pg.wait_for_function(
                "typeof openStockDetail === 'function' && typeof D !== 'undefined' && "
                "Object.values(D.stock_detail).some(s => s.ctx)", timeout=60000)
            for t, sel in TARGETS:
                pg.evaluate(f"() => openStockDetail('{t}')")
                pg.wait_for_timeout(250)
                el = pg.query_selector(sel)
                if el is None:
                    print(f"{t} {label}: no {sel}")
                    continue
                el.scroll_into_view_if_needed()
                pg.wait_for_timeout(150)
                path = OUT / f"2026-11-03-{t}-{label}.png"
                el.screenshot(path=str(path))
                txt = " | ".join(x.strip() for x in el.inner_text().split("\n") if x.strip())
                print(f"{t} {label}: {txt[:400]}")
                print(f"   -> {path}")
                pg.evaluate("() => closeModal()")
            pg.close()
        b.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
