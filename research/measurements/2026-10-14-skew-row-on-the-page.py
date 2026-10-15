"""Look at the options panel after the skew change, at 1440 and 375px.

Two states have to be checked, because the data and the code move on different clocks:

A. **As the next page will actually render today.** `_ctx_opt_skew` is computed at *fetch*
   time, so the newest full run carries the superseded 0.90-moneyness-minus-mean measure and
   none of the new legs. The generator withholds the row in that case rather than printing a
   number the sentence beside it does not describe. This confirms the panel still reads well
   with the row absent - no orphaned label, no dangling sentence.

B. **As it will render after the next full fetch**, with the legs present. There is no way to
   get real new-construction values onto a page before the 02:00 run, so two tickers' legs are
   injected into a **local throwaway copy** of the regenerated context file. That copy is never
   published and is deleted at the end; it exists only to look at the row.

Run `python scripts/rehearse_publish.py` first - this reads its output in
`logs/rehearse-publish/site/`.

Usage: python -I -X utf8 research/measurements/2026-10-14-skew-row-on-the-page.py
Writes PNGs to logs/design/ (gitignored). Publishes nothing.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent.parent
SITE = ROOT / "logs" / "rehearse-publish" / "site"
OUT = ROOT / "logs" / "design"

# Chosen so the window clause is exercised in **both** directions with the stock's *real*
# days-to-expiry, not a flag invented independently of it - measured from the 2026-10-14 cache,
# where expiries outside the paper's 10-60 day range run 9 days (51 stocks) and 65-69 days.
# ``oskw`` below is therefore what ``_xzz_skew`` would actually compute for that stock's ``od``.
#   AAPL  23 days - inside
#   ABT    9 days - outside, short (an expiry chosen to span a report)
#   A     65 days - outside, long  (this is the case the first wording got backwards)
INJECT = {
    "AAPL": {"osk": 0.0499, "oskp": 0.3455, "oskc": 0.2956, "oskps": 305, "oskcs": 340,
             "oskw": True},
    "ABT":  {"osk": -0.0212, "oskp": 0.1988, "oskc": 0.2200, "oskps": 125, "oskcs": 132.5,
             "oskw": False},
    "A":    {"osk": 0.0314, "oskp": 0.4520, "oskc": 0.4206, "oskps": 155, "oskcs": 170,
             "oskw": False},
}

PANEL_JS = """
(t) => {
  openStockDetail(t);
  const cards = [...document.querySelectorAll('.ctx-card')];
  const card = cards.find(c => (c.querySelector('.ctx-h') || {}).textContent
                               && c.querySelector('.ctx-h').textContent.includes('options'));
  if (!card) return {found: false, cards: cards.length};
  const rows = [...card.querySelectorAll('tr')].map(r =>
      [...r.querySelectorAll('td')].map(td => td.textContent.trim()).join('  |  '));
  const why = (card.querySelector('.ctx-why') || {}).textContent || '';
  const r = card.getBoundingClientRect();
  return {found: true, rows, why, width: Math.round(r.width),
          overflows: r.right > window.innerWidth + 1};
}
"""


def _shoot(label: str, site: Path, tickers: list[str]) -> None:
    url = (site / "dashboard.html").as_uri()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, w, h in [("desktop", 1440, 1000), ("mobile", 375, 812)]:
            page = browser.new_page(viewport={"width": w, "height": h})
            page.goto(url, wait_until="networkidle", timeout=90000)
            page.wait_for_function("() => !!window.SCREENER_CONTEXT", timeout=60000)
            for t in tickers:
                info = page.evaluate(PANEL_JS, t)
                print(f"\n--- {label} / {name} / {t} ---")
                if not info.get("found"):
                    print("  options card NOT FOUND", info)
                    continue
                for row in info["rows"]:
                    print("  row:", row)
                print(f"  card width {info['width']}px, overflows viewport: {info['overflows']}")
                print("  why:", " ".join(info["why"].split())[:700])
                # The sheet animates in; shooting immediately double-exposes it over the page
                # behind. Settle, then put the options card in frame - otherwise the screenshot
                # shows the top of the drilldown and not the row being checked.
                page.wait_for_timeout(700)
                page.evaluate(
                    "() => { const c = [...document.querySelectorAll('.ctx-card')]"
                    ".find(x => (x.querySelector('.ctx-h')||{}).textContent"
                    "           && x.querySelector('.ctx-h').textContent.includes('options'));"
                    "  if (c) c.scrollIntoView({block: 'center'}); }")
                page.wait_for_timeout(400)
                page.screenshot(path=str(OUT / f"{label}-{name}-{t}.png"))
                page.evaluate("closeModal()")
                page.wait_for_timeout(300)
            page.close()
        browser.close()


def main() -> int:
    if not (SITE / "dashboard.html").exists():
        print(f"missing {SITE}; run python scripts/rehearse_publish.py first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("A. the page as the next render produces it - run predates the change, row withheld")
    print("=" * 78)
    _shoot("skew-withheld", SITE, ["AAPL", "ABT"])

    mock = ROOT / "logs" / "design" / "skew-mock-site"
    try:
        shutil.rmtree(mock, ignore_errors=True)
        # ``dirs_exist_ok`` because rmtree is best-effort under OneDrive, which can hold a
        # directory open after the files in it are gone - so the script stays re-runnable.
        shutil.copytree(SITE, mock, dirs_exist_ok=True)
        ctx_path = mock / "dashboard_context.js"
        text = ctx_path.read_text(encoding="utf-8")
        obj = json.loads(text[text.index("{"):text.rindex("}") + 1])
        for t, legs in INJECT.items():
            entry = obj["ctx"].get(t)
            if entry is None:
                print(f"  {t} has no context entry; skipping")
                continue
            entry.update(legs)
        head = text[:text.index("{")]
        ctx_path.write_text(head + json.dumps(obj, separators=(",", ":")) + ";\n",
                            encoding="utf-8")
        print()
        print("=" * 78)
        print("B. a local throwaway copy with the legs injected - never published")
        print("=" * 78)
        _shoot("skew-new", mock, list(INJECT))
    finally:
        shutil.rmtree(mock, ignore_errors=True)
        print("\nthrowaway copy deleted")

    # Nothing in the repo may have moved.
    assert (ROOT / "dashboard_context.js").exists()
    return 0


if __name__ == "__main__":
    sys.exit(main())
