"""Screenshot one section of the generated ``research.html`` at desktop and 375px.

    python scripts/shot_research.py <label> [heading-text]

Writes ``logs/design/<label>-research-{desktop,mobile}.png`` (gitignored via ``logs/``) and prints
whether the section overflows its viewport horizontally - the one defect a CSS read cannot see
(CLAUDE.md: look at the page at desktop and 375px and say what you saw). Companion to
``shot_dashboard.py``, which only knows about ``index.html``.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "design"


def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else "research"
    heading = sys.argv[2] if len(sys.argv) > 2 else "stop-loss"
    OUT.mkdir(parents=True, exist_ok=True)
    url = (ROOT / "research.html").as_uri()
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        for name, w, h in (("desktop", 1440, 1000), ("mobile", 375, 820)):
            page = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=2)
            page.goto(url, wait_until="load")
            el = page.locator(f"h3:has-text('{heading}')").first
            el.scroll_into_view_if_needed()
            section = page.locator("section", has=el).first
            section.screenshot(path=str(OUT / f"{label}-research-{name}.png"))
            over = page.evaluate("() => document.documentElement.scrollWidth - window.innerWidth")
            box = section.bounding_box()
            print(f"{name} {w}x{h}: horizontal overflow {over}px, section {box['width']:.0f}x{box['height']:.0f}")
            page.close()
        b.close()
    print(f"wrote {OUT / (label + '-research-desktop.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
