"""Download the papers and methodology documents a note cites, to the gitignored `data/papers/`.

The companion to `_read_pdf.py` and `_pdf_pages.py`: `prompts/nightly.md` says to fetch the PDF
and read the tables rather than cite from memory, and that a publisher's bot check is not to be
worked around - so this prints the byte count and the file's first bytes, which is what makes a
403 or an HTML error page visible instead of silently becoming an unreadable "paper".

Usage:  python research/measurements/_fetch_pdfs.py <slug>=<url> [...]
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")
DEST = Path(__file__).resolve().parents[2] / "data" / "papers"


def main(args: list[str]) -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for a in args:
        slug, _, url = a.partition("=")
        out = DEST / f"{slug}.pdf"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            out.write_bytes(data)
            print(f"{slug:34s} {len(data):>9,d} bytes  head={data[:5]!r}  {url}")
        except Exception as e:                                    # noqa: BLE001
            print(f"{slug:34s} FAILED {type(e).__name__}: {e}  {url}")


if __name__ == "__main__":
    main(sys.argv[1:])
