"""Screenshot one section of the Methodology dialog at desktop and phone widths.

The four ship gates read the *committed* index.html, which only a full run rewrites, so a
generator change to the methodology text cannot be looked at there. Point this at the page
`scripts/rehearse_publish.py` regenerates instead:

    python scripts/rehearse_publish.py            # writes logs/rehearse-publish/site/
    python scripts/shot_methodology.py <label> [--page <path to dashboard.html>]
                                               [--section "8. Investment"]
                                               [--marker "most practitioners file it"]

Defaults to the rehearsal's page if it exists, else the published index.html. Writes
logs/design/<label>-methodology-{desktop,mobile}.png (gitignored) and prints the chosen
section's text as the dialog actually renders it, so the session can say what it saw.

`--section` is the "### n. <Category>" heading to print; `--marker` is a regex for the
paragraph to scroll to and to measure on the phone. Both default to the Growth section's
weight paragraph, which is what this script was written for on 2026-11-03; they became
arguments on 2026-11-04, when a second session needed a second section and the alternative
was a hand-edit nobody would have kept.

The dialog is filled on first open (`hydrateMethodology`); `.methodology-body` is empty before
that by design (CLAUDE.md, ship gates). So it is opened before anything is read.
"""
from __future__ import annotations

import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "logs" / "design"
REHEARSAL = ROOT / "logs" / "rehearse-publish" / "site" / "dashboard.html"

# The text of one '### n. <Category>' block, as the dialog renders it.
SECTION_JS = """
(heading) => {
  const body = document.querySelector('#methodology-modal .methodology-body');
  if (!body) return {error: 'no .methodology-body'};
  const text = body.innerText || '';
  const start = text.indexOf(heading);
  if (start < 0) return {error: 'heading not found', length: text.length};
  const rest = text.slice(start + heading.length);
  const next = rest.search(/\\n\\s*\\d+\\.\\s/);
  return {text: heading + (next > 0 ? rest.slice(0, next) : rest.slice(0, 4000))};
}
"""


def main() -> int:
    args = [a for a in sys.argv[1:]]
    label = args[0] if args and not args[0].startswith("--") else "methodology"
    page_path = REHEARSAL if REHEARSAL.exists() else ROOT / "index.html"
    if "--page" in args:
        page_path = Path(args[args.index("--page") + 1]).resolve()
    section = args[args.index("--section") + 1] if "--section" in args else "3. Growth"
    marker = args[args.index("--marker") + 1] if "--marker" in args else "Why the weight stays"
    if not page_path.exists():
        print(f"no page at {page_path}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"page: {page_path}")
    print(f"section: {section!r}  marker: /{marker}/")

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, width, height in (("desktop", 1440, 900), ("mobile", 375, 812)):
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(page_path.as_uri())
            page.wait_for_function("() => window.SCREENER_DATA && window.SCREENER_DATA.table_data")
            page.evaluate("() => openMethodology()")
            page.wait_for_function(
                "() => { const b = document.querySelector('#methodology-modal .methodology-body');"
                " return b && b.innerText.length > 2000; }"
            )
            # scroll the chosen paragraph into view so the shot shows it, not the top of the doc
            page.evaluate(
                """(marker) => {
                  const re = new RegExp(marker);
                  const body = document.querySelector('#methodology-modal .methodology-body');
                  const hit = [...body.querySelectorAll('h3, h2, strong')]
                    .find(e => re.test(e.textContent));
                  if (hit) hit.scrollIntoView({block: 'center'});
                }""",
                marker,
            )
            page.wait_for_timeout(250)
            shot = OUT / f"{label}-methodology-{name}.png"
            page.screenshot(path=str(shot))
            print(f"  {name:7} {width}x{height} -> {shot.name}")
            if name == "desktop":
                got = page.evaluate(SECTION_JS, section)
                print(f"\n--- {section} as the dialog renders it ---")
                print(got.get("text") or got)
                print("--- end ---\n")
            else:
                # on a phone, confirm the paragraph is present and not clipped out of the layout
                box = page.evaluate(
                    """(marker) => {
                      const re = new RegExp(marker);
                      const body = document.querySelector('#methodology-modal .methodology-body');
                      const hit = [...body.querySelectorAll('p, strong')]
                        .find(e => re.test(e.textContent));
                      if (!hit) return null;
                      const r = hit.getBoundingClientRect();
                      return {w: Math.round(r.width), overflowsRight: r.right > window.innerWidth + 1,
                              visible: getComputedStyle(hit).display !== 'none'};
                    }""",
                    marker,
                )
                print(f"  phone paragraph box: {box}")
            page.close()
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
