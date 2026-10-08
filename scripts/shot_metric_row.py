"""Open one metric's row inside a stock's workings and screenshot what it says.

The rule in CLAUDE.md 0.8c is that a scoring change ships with its frontend, and the way to
check that is to *look at the row*, at both widths, and read it. This opens a stock's
drilldown, jumps to a category's workings, clicks the named metric's row and shoots it.

Usage: python scripts/shot_metric_row.py <label> <category> <metric> TICKER [TICKER ...]
e.g.   python scripts/shot_metric_row.py mdd risk max_drawdown_1y SNPS AAPL
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "design"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    label, cat, metric, tickers = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    url = (ROOT / "index.html").as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for view, w, h in [("desktop", 1440, 1000), ("mobile", 375, 812)]:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1200)
            for ticker in tickers:
                page.evaluate(f"openStockDetail({ticker!r})")
                page.wait_for_timeout(500)
                page.evaluate(f"openWorkings({cat!r})")
                page.wait_for_timeout(500)
                row = page.query_selector(f'tr.metric-row[data-metric="{metric}"]')
                if row is None:
                    print(f"[{view}] {ticker}: no row for {metric}")
                    continue
                row.scroll_into_view_if_needed()
                row.click()
                page.wait_for_timeout(500)
                text = page.evaluate(
                    """(m) => {
                        const tr = document.querySelector(`tr.metric-row[data-metric="${m}"]`);
                        if (!tr) return '(gone)';
                        const parts = [tr.innerText];
                        let n = tr.nextElementSibling;
                        while (n && !n.classList.contains('metric-row')) {
                            parts.push(n.innerText); n = n.nextElementSibling;
                        }
                        return parts.join('\\n---\\n');
                    }""", metric)
                print(f"\n===== [{view}] {ticker} / {metric} =====")
                print(text.strip()[:1800])
                path = OUT / f"{label}-{ticker}-{view}.png"
                page.screenshot(path=str(path))
                print(f"  -> {path.relative_to(ROOT)}")
            page.close()
        browser.close()


if __name__ == "__main__":
    main()
