"""Rebuild the published page from the published payload, for presentation-only work.

    python scripts/rebuild_page.py           # rewrite index.html and dashboard.html
    python scripts/rebuild_page.py --check   # exit 1 if they differ from what the code builds

A design change to `generate_dashboard.py` alters the page the 02:00 loop will publish, but the
committed `index.html` only changes on a full run. Running `generate_dashboard.py` to see the new
page rewrites `dashboard_data.js` from the newest run, which breaks the CLAUDE.md rule that
presentation work leaves every number, rank and sentence byte-identical. This script calls
`generate_html` alone, with the timestamp and the two cache-busting versions recovered from the
committed files, so **the payload cannot change by construction**: it never opens either payload
file for writing, and it checks both hashes before and after.

On an unchanged generator the output is byte-identical to the committed page, which `--check`
proves. Run `python scripts/rehearse_publish.py` as well whenever the generator changes - this
script shows you the page; that one proves the next data run will publish it.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PAYLOADS = ("dashboard_data.js", "dashboard_context.js")
PAGES = ("index.html", "dashboard.html")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _version(path: Path, prefix: str) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.startswith(prefix) and text.endswith(";"), f"{path.name}: unexpected framing"
    return hashlib.md5(text[len(prefix):-1].encode("utf-8")).hexdigest()[:12]


def build() -> str:
    import generate_dashboard as gd

    page = (ROOT / "index.html").read_text(encoding="utf-8")
    m = re.search(r'<span id="gen-time">([^<]*)</span>', page)
    if not m:
        raise SystemExit("index.html has no #gen-time - cannot recover the data timestamp")
    data_version = _version(ROOT / "dashboard_data.js", "window.SCREENER_DATA = ")
    context_version = _version(ROOT / gd.CONTEXT_FILE, "window.SCREENER_CONTEXT = ")
    if f"dashboard_data.js?v={data_version}" not in page:
        raise SystemExit("index.html does not reference the committed payload's version - "
                         "the page and payload are from different runs; rebuild with a full run")
    return gd.generate_html(methodology_html=gd._load_methodology_html(), data_timestamp=m.group(1),
                            data_version=data_version, context_version=context_version)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare only; write nothing")
    args = ap.parse_args(argv)

    before = {n: _sha(ROOT / n) for n in PAYLOADS}
    html = build()
    if args.check:
        stale = [n for n in PAGES if (ROOT / n).read_text(encoding="utf-8") != html]
        print("PAGE: up to date" if not stale else f"PAGE: differs from the generator: {', '.join(stale)}")
        return 1 if stale else 0
    for n in PAGES:
        (ROOT / n).write_text(html, encoding="utf-8")
    after = {n: _sha(ROOT / n) for n in PAYLOADS}
    if before != after:
        raise SystemExit("PAYLOAD CHANGED - this must never happen; restore it from git")
    print(f"Rebuilt {', '.join(PAGES)} ({len(html) / 1024:.0f} KB); payload byte-identical.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
