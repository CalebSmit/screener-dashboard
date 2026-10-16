"""Look at the earnings-yield row's one-off-charge note, at 1440 and 375 - 2026-10-16

CLAUDE.md: a scoring or page change is looked at in a real browser at both widths before it ships,
and the log says what was seen. The note is built into the payload, so it appears on the page the
next data run publishes - this reads the regenerated page that `scripts/rehearse_publish.py` leaves
in ``logs/rehearse-publish/site/``, which is that page.

Three stocks, chosen because they exercise the two branches of the sentence and the worst case
for the layout:
  SBUX - a profit, so the share of net income is printed;
  KHC  - a loss, so it is called a loss and no share is printed;
  GILD - the longest tag in the set, ImpairmentOfIntangibleAssetsIndefinitelivedExcludingGoodwill
         at 59 characters, which is one unbreakable token at 375px.

Usage: python research/measurements/2026-10-16-shot-special-item-note.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SITE = ROOT / "logs" / "rehearse-publish" / "site" / "dashboard.html"
OUT = ROOT / "logs" / "design"
WIDTHS = {"desktop": (1440, 1000), "mobile": (375, 812)}


def main():
    from playwright.sync_api import sync_playwright
    if not SITE.exists():
        sys.exit(f"no regenerated page at {SITE} - run scripts/rehearse_publish.py first")
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for label, (w, h) in WIDTHS.items():
            for ticker in ("SBUX", "KHC", "GILD"):
                page = browser.new_context(viewport={"width": w, "height": h}).new_page()
                errors = []
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.goto(SITE.as_uri(), wait_until="load", timeout=60_000)
                page.wait_for_function(
                    "document.querySelectorAll('#universe-tbody tr[data-t]').length > 0",
                    timeout=60_000)
                page.evaluate(f"openStockDetail('{ticker}')")
                page.wait_for_selector("#stock-modal .modal-body", state="visible")
                page.evaluate("openWorkings('valuation')")
                page.wait_for_timeout(300)
                row = "#cat-detail-valuation tr[data-metric='earnings_yield']"
                page.click(f"{row} .metric-raw")
                page.wait_for_timeout(250)
                card = page.inner_text("#cat-detail-valuation tr.wk-detail .wk-card")
                page.evaluate("""() => { const e = document.querySelector(
                    '#cat-detail-valuation tr.wk-detail .wk-caveat-stock');
                    if (e) e.scrollIntoView({block: 'center'}); }""")
                page.wait_for_timeout(200)
                shot = OUT / f"2026-10-16-special-item-{ticker}-{label}.png"
                page.screenshot(path=str(shot))
                notes = page.evaluate("""() => [...document.querySelectorAll(
                    '#cat-detail-valuation tr.wk-detail .wk-caveat-stock')].map(e => e.innerText)""")
                overflow = page.evaluate("document.documentElement.scrollWidth")
                print(f"\n=== {ticker} at {label} ({w}px) -> {shot.name}")
                print(f"    page scrollWidth {overflow} (viewport {w})")
                print(f"    row shows: {page.inner_text(row + ' .metric-raw').strip()[:90]}")
                for n in notes:
                    print(f"    note: {n}")
                if errors:
                    print(f"    JS ERRORS: {errors}")
                assert any("own filings report for the same twelve months" in n for n in notes), \
                    f"{ticker}: the filings note is not on the row"
                assert overflow <= w + 1, f"{ticker} at {label}: overflows by {overflow - w}px"
                page.context.close()
        browser.close()
    print(f"\nshots in {OUT}")


if __name__ == "__main__":
    main()
