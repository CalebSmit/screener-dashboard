"""Look at the market backdrop panel with the corrected sentences, at 1440 and 375.

The rule (CLAUDE.md, "How it looks and feels is part of whether it is correct"): look at the
page before and after any change that shows on it, at both widths, and say what you saw. The
corrected readings do not reach the published payload until the next full run rewrites
``data/market_context.json``, so this script injects them into ``window.SCREENER_CONTEXT`` in
the live page before the panel renders - which is exactly the layout the next run will produce,
with no published artifact touched.

Writes logs/design/backdrop-{before,after}-{desktop,mobile}.png and prints, for each reading,
the rendered height and whether the panel or any card overflows its column.

Usage:  python scripts/../research/measurements/2026-10-17-shot-backdrop.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs" / "design"
sys.path.insert(0, str(ROOT))

MEASURE_JS = """
() => {
  const host = document.getElementById('market-body');
  const panel = host ? host.querySelector('.mk-readings') : null;
  if (!panel) return {error: 'no .mk-readings'};
  const cards = [...panel.children];
  return {
    panel: {w: panel.clientWidth, h: panel.clientHeight,
            overflowX: panel.scrollWidth > panel.clientWidth + 1},
    doc_overflow: document.documentElement.scrollWidth > window.innerWidth + 1,
    cards: cards.map(c => ({
      title: (c.querySelector('.mk-title, h3, strong') || c).textContent.slice(0, 22).trim(),
      h: Math.round(c.getBoundingClientRect().height),
      w: Math.round(c.getBoundingClientRect().width),
      clipped: c.scrollHeight > c.clientHeight + 1,
    })),
  };
}
"""


def main() -> int:
    import market_context
    from playwright.sync_api import sync_playwright

    stored = json.loads((ROOT / "data" / "market_context.json").read_text(encoding="utf-8"))
    after = market_context.readings(stored["series"], stored["sahm"], stored.get("sahm_basis"))
    before = stored["readings"]
    print(f"before: {sum(len(r['text']) for r in before)} chars over {len(before)} readings")
    print(f"after:  {sum(len(r['text']) for r in after)} chars over {len(after)} readings")
    for b, a in zip(before, after):
        if b["text"] != a["text"]:
            print(f"  {a['k']:<10} {len(b['text']):>4} -> {len(a['text']):>4} chars")

    OUT.mkdir(parents=True, exist_ok=True)
    url = (ROOT / "index.html").as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for which, readings in (("before", None), ("after", after)):
            for name, w, h in (("desktop", 1440, 1000), ("mobile", 375, 812)):
                page = browser.new_page(viewport={"width": w, "height": h})
                page.goto(url, wait_until="networkidle", timeout=60000)
                page.wait_for_function("() => !!window.SCREENER_CONTEXT", timeout=30000)
                if readings is not None:
                    page.evaluate("r => { window.SCREENER_CONTEXT.market.readings = r; }", readings)
                page.evaluate("goToSection('sec-market')")
                page.wait_for_timeout(900)
                el = page.query_selector("#sec-market")
                (el or page).screenshot(path=str(OUT / f"backdrop-{which}-{name}.png"))
                print(f"\n[{which} {name}] " + json.dumps(page.evaluate(MEASURE_JS), indent=1))
                page.close()
        browser.close()
    print(f"\nwrote backdrop-before/after-desktop/mobile to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
