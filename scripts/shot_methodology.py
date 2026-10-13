"""Screenshot the Methodology dialog's Growth section at desktop and phone widths.

The four ship gates read the *committed* index.html, which only a full run rewrites, so a
generator change to the methodology text cannot be looked at there. Point this at the page
`scripts/rehearse_publish.py` regenerates instead:

    python scripts/rehearse_publish.py            # writes logs/rehearse-publish/site/
    python scripts/shot_methodology.py <label> [--page <path to dashboard.html>]

Defaults to the rehearsal's page if it exists, else the published index.html. Writes
logs/design/<label>-methodology-{desktop,mobile}.png (gitignored) and prints the Growth
section's text as the dialog actually renders it, so the session can say what it saw.

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
    if not page_path.exists():
        print(f"no page at {page_path}")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"page: {page_path}")

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
            # scroll the Growth section into view so the shot shows the paragraph, not the top
            page.evaluate(
                """() => {
                  const body = document.querySelector('#methodology-modal .methodology-body');
                  const hit = [...body.querySelectorAll('h3, h2, strong')]
                    .find(e => /Why the weight stays/.test(e.textContent));
                  if (hit) hit.scrollIntoView({block: 'center'});
                }"""
            )
            page.wait_for_timeout(250)
            shot = OUT / f"{label}-methodology-{name}.png"
            page.screenshot(path=str(shot))
            print(f"  {name:7} {width}x{height} -> {shot.name}")
            if name == "desktop":
                got = page.evaluate(SECTION_JS, "3. Growth")
                print("\n--- Growth section as the dialog renders it ---")
                print(got.get("text") or got)
                print("--- end ---\n")
            else:
                # on a phone, confirm the paragraph is present and not clipped out of the layout
                box = page.evaluate(
                    """() => {
                      const body = document.querySelector('#methodology-modal .methodology-body');
                      const hit = [...body.querySelectorAll('p, strong')]
                        .find(e => /Why the weight stays/.test(e.textContent));
                      if (!hit) return null;
                      const r = hit.getBoundingClientRect();
                      return {w: Math.round(r.width), overflowsRight: r.right > window.innerWidth + 1,
                              visible: getComputedStyle(hit).display !== 'none'};
                    }"""
                )
                print(f"  phone paragraph box: {box}")
            page.close()
        browser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
