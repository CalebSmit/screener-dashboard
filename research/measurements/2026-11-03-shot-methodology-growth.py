"""Read the Growth weight paragraph back out of the Methodology dialog, as a reader opens it.

The paragraph is generated (`run_screener.build_screener_overview`) and reaches the page through
the methodology template, so "it is right in the markdown" is not the same fact as "it renders".
The dialog is filled on first open (`hydrateMethodology`), so this clicks it open first.

    python -X utf8 research/measurements/2026-11-03-shot-methodology-growth.py
"""
from __future__ import annotations

from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "logs" / "design"
OUT.mkdir(parents=True, exist_ok=True)

FIND = """() => {
  openMethodology();
  const body = document.querySelector('#methodology-modal .methodology-body');
  const p = [...body.querySelectorAll('p')].find(e => e.innerText.includes('Why the weight stays at'));
  if (!p) return null;
  p.scrollIntoView({block: 'center'});
  return p.innerText;
}"""

with sync_playwright() as pw:
    b = pw.chromium.launch()
    for label, w, h in (("desktop", 1440, 900), ("mobile", 375, 812)):
        pg = b.new_page(viewport={"width": w, "height": h})
        pg.goto((ROOT / "index.html").as_uri(), wait_until="load", timeout=60000)
        pg.wait_for_function("typeof openMethodology === 'function'", timeout=60000)
        txt = pg.evaluate(FIND)
        pg.wait_for_timeout(200)
        path = OUT / f"2026-11-03-methodology-growth-{label}.png"
        pg.screenshot(path=str(path))
        print(f"--- {label} ---\n{txt}\n-> {path}\n")
        pg.close()
    b.close()
