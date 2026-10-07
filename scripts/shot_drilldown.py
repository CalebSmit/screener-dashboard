"""Open a stock's drilldown and screenshot it, at desktop and 375px.

`shot_dashboard.py` takes one drilldown shot of the top-ranked name as a
by-product of its page metrics. This opens *named* tickers, which is what
`plan/calculation-transparency.md`'s verification step 4 asks for: look at a
bank, a Piotroski-adjusted name, a thin-coverage name and an ordinary one,
because choosing them on purpose is what exercises the paths that were wrong.

It also prints the rendered "Why it ranks here" text, so a session can quote
what the page actually says rather than what the generator intended.

Usage: python scripts/shot_drilldown.py <label> TICKER [TICKER ...]
"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "design"
OUT.mkdir(parents=True, exist_ok=True)

SUMMARY_JS = """
() => {
  const el = document.querySelector('#modal-summary, .summary-facts, #stock-modal');
  return el ? el.innerText.slice(0, 1200) : '(no summary element found)';
}
"""


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    label, tickers = sys.argv[1], sys.argv[2:]
    url = (ROOT / "index.html").as_uri()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for view, w, h in [("desktop", 1440, 1000), ("mobile", 375, 812)]:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(url, wait_until="networkidle", timeout=60000)
            page.wait_for_timeout(1200)
            for ticker in tickers:
                page.evaluate(f"openStockDetail({ticker!r})")
                page.wait_for_timeout(700)
                page.screenshot(path=str(OUT / f"{label}-{ticker}-{view}.png"))
                if view == "desktop":
                    print(f"\n===== {ticker} =====")
                    print(page.evaluate(SUMMARY_JS))
            page.close()
        browser.close()
    print(f"\nwrote shots to {OUT}")


if __name__ == "__main__":
    main()
