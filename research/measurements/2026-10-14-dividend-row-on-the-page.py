"""What the Dividend Yield row actually says on the page, at both widths.

Two false statements were live on the 2026-10-14 02:00 build:

  * "Dividend Yield: None" for 495 of 501 stocks - including Coca-Cola, J&J and
    Procter & Gamble - because the metric's only input vanished and absent was
    rendered as a measured zero;
  * a spurious input-churn caveat on 87 stocks, telling a reader that two of the
    metrics behind the comparison had changed availability when one of the two
    changed for all 501.

This reads the rendered row and the rendered summary, rather than trusting the
generator, and shoots both widths. The row lives in the Data section, which is
collapsed on open.

Usage: python research/measurements/2026-10-14-dividend-row-on-the-page.py [TICKER ...]
Opens the committed index.html in Chromium. No network, writes screenshots to logs/design/.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "logs" / "design"

# The row is a label/value pair in the financials grid. Read every pair whose
# label mentions a dividend or a payout, plus its sub-line.
ROW_JS = """
() => {
  const out = [];
  document.querySelectorAll('*').forEach(el => {
    if (el.children.length) return;
    const t = (el.textContent || '').trim();
    if (t === 'Dividend Yield' || t === 'Payout Ratio') {
      const box = el.parentElement;
      out.push((box ? box.innerText : t).replace(/\\n/g, ' | '));
    }
  });
  return out;
}
"""

CHURN_JS = """
() => {
  const el = document.querySelector('#summaryList, .summary-list, #whyRanks');
  const txt = el ? el.innerText : document.body.innerText;
  return txt.split('\\n').filter(l => /Caution on that comparison|changed availability|could be computed/.test(l));
}
"""


def main() -> None:
    tickers = sys.argv[1:] or ["KO", "JNJ", "PG", "PYPL", "GOOGL"]
    OUT.mkdir(parents=True, exist_ok=True)
    url = (ROOT / "index.html").as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for view, w, h in [("desktop", 1440, 1000), ("mobile", 375, 812)]:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1200)
            for ticker in tickers:
                page.evaluate(f"openStockDetail({ticker!r})")
                page.wait_for_timeout(600)
                # Open every collapsed section so the Data grid renders.
                page.evaluate("""
                  () => document.querySelectorAll('details').forEach(d => d.open = true)
                """)
                page.evaluate("""
                  () => document.querySelectorAll('[aria-expanded="false"]').forEach(b => b.click())
                """)
                page.wait_for_timeout(500)
                rows = page.evaluate(ROW_JS)
                churn = page.evaluate(CHURN_JS)
                print(f"\n== {ticker} ({view}) ==")
                for r in rows:
                    print(f"   {r}")
                print(f"   input-churn sentences: {churn if churn else 'none'}")
                page.screenshot(path=str(OUT / f"dividend-row-{ticker}-{view}.png"),
                                full_page=(view == "mobile"))
            page.close()
        browser.close()
    print(f"\nshots in {OUT}")


if __name__ == "__main__":
    main()
