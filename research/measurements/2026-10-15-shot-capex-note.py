"""Open GM's FCF-yield workings on the regenerated page and photograph them, 1440 and 375px.

For the capex-note review: the standing rule is to look at the page before and after a change
and say what was seen. Points at the rehearsal site by default (the regenerated page the next
data run would publish), since the published files are restored byte for byte.

    python research/measurements/2026-10-15-shot-capex-note.py [PAGE_DIR] [TICKER]
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
PAGE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "logs" / "rehearse-publish" / "site"
TICKER = (sys.argv[2] if len(sys.argv) > 2 else "GM").upper()
OUT = ROOT / "logs" / "design"
OUT.mkdir(parents=True, exist_ok=True)
HTML = (PAGE / "dashboard.html") if (PAGE / "dashboard.html").exists() else (PAGE / "index.html")

OPEN_JS = """
(t) => {
  openStockDetail(t);
  openWorkings('valuation');
  const sec = document.querySelector('.cat-detail-section[data-cat="valuation"]');
  if (!sec) return 'no valuation section';
  const tr = sec.querySelector('tr[data-metric="fcf_yield"]');
  if (!tr) return 'no fcf_yield row';
  toggleMetricDetail(tr);
  const body = tr.nextElementSibling;
  const cav = body && body.querySelector('.wk-caveat-stock');
  return cav ? cav.innerText : 'no stock caveat rendered';
}
"""

with sync_playwright() as p:
    b = p.chromium.launch()
    for label, w, h in (("desktop", 1440, 900), ("mobile", 375, 812)):
        pg = b.new_page(viewport={"width": w, "height": h})
        pg.goto(HTML.as_uri())
        pg.wait_for_function("() => window.SCREENER_DATA && document.querySelectorAll('#universe-tbody tr').length > 0")
        print(f"--- {label} {w}x{h}: {pg.evaluate(OPEN_JS, TICKER)}")
        pg.wait_for_timeout(400)
        pg.eval_on_selector(".wk-caveat-stock", "el => el.scrollIntoView({block:'center'})")
        pg.wait_for_timeout(400)
        shot = OUT / f"capex-note-{label}.png"
        pg.screenshot(path=str(shot))
        print(f"    {shot}")
        pg.close()
    b.close()
