"""Screenshot index.html at desktop and mobile widths, and report feel metrics.

Usage: python scripts/shot_dashboard.py <label>
Writes logs/design/<label>-desktop.png, <label>-mobile.png (gitignored via logs/)
and prints DOM node count, elements with CSS transitions, and font census.
"""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "design"
OUT.mkdir(parents=True, exist_ok=True)

METRICS_JS = """
() => {
  const all = [...document.querySelectorAll('*')];
  const fonts = {};
  const sizes = {};
  let transitions = 0;
  for (const el of all) {
    const cs = getComputedStyle(el);
    const fam = cs.fontFamily.split(',')[0].replace(/['"]/g, '').trim();
    fonts[fam] = (fonts[fam] || 0) + 1;
    sizes[cs.fontSize] = (sizes[cs.fontSize] || 0) + 1;
    if (cs.transitionDuration && cs.transitionDuration.split(',').some(d => parseFloat(d) > 0)) transitions++;
  }
  return {
    nodes: all.length,
    transitions,
    fonts: Object.fromEntries(Object.entries(fonts).sort((a,b)=>b[1]-a[1]).slice(0,6)),
    sizes: Object.fromEntries(Object.entries(sizes).sort((a,b)=>b[1]-a[1]).slice(0,6)),
  };
}
"""


FEEL_JS = """
() => new Promise(res => {
  const out = {};
  const rows = document.querySelectorAll('#universe-tbody tr[data-t]').length;
  out.rendered_table_rows = rows;
  const shell = document.querySelectorAll('*').length;
  out.nodes_first_paint = shell;
  let t0 = performance.now();
  sortTable('momentum_score');
  requestAnimationFrame(() => requestAnimationFrame(() => {
    out.sort_to_paint_ms = Math.round(performance.now() - t0);
    t0 = performance.now();
    const tr = document.querySelector('#universe-tbody tr[data-t]');
    if (tr) tr.click();
    requestAnimationFrame(() => requestAnimationFrame(() => {
      out.row_click_to_paint_ms = Math.round(performance.now() - t0);
      let cls = 0;
      try {
        const po = new PerformanceObserver(l => { for (const e of l.getEntries()) if (!e.hadRecentInput) cls += e.value; });
        po.observe({ type: 'layout-shift', buffered: true });
      } catch (e) {}
      setTimeout(() => { out.layout_shift = Math.round(cls * 1000) / 1000; closeModal(); res(out); }, 200);
    }));
  }));
})
"""


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "shot"
    url = (ROOT / "index.html").as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, w, h in [("desktop", 1440, 1000), ("mobile", 375, 812)]:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT / f"{label}-{name}.png"))
            if name == "desktop":
                print(json.dumps(page.evaluate(METRICS_JS), indent=1))
                # The budgets in plan/dashboard-redesign-master.md: nodes, click-to-paint, shift.
                print("feel:", json.dumps(page.evaluate(FEEL_JS)))
            # drilldown shot on desktop only
            if name == "desktop":
                page.evaluate("openStockDetail(SCREENER_DATA.table_data[0].ticker)")
                page.wait_for_timeout(600)
                page.screenshot(path=str(OUT / f"{label}-drilldown.png"))
            page.close()
        browser.close()
    print(f"wrote {label}-desktop/mobile/drilldown to {OUT}")


if __name__ == "__main__":
    main()
