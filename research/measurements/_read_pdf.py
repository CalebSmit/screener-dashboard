"""Print the pages of a downloaded paper that match a regex - the reading step in
prompts/nightly.md "How to read a paper", as a script because the shell refuses heredocs.

Usage:  python -I -X utf8 research/measurements/_read_pdf.py <pdf> <regex> [chars]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> int:
    path = Path(sys.argv[1])
    pattern = sys.argv[2] if len(sys.argv) > 2 else "."
    chars = int(sys.argv[3]) if len(sys.argv) > 3 else 2600
    reader = PdfReader(str(path))
    print(f"{path.name}: {len(reader.pages)} pages, matching /{pattern}/i\n")
    hits = 0
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if re.search(pattern, text, re.I):
            hits += 1
            print(f"--- page {i} (pdf page {i + 1}) ---")
            print(text[:chars])
            print()
    print(f"[{hits} page(s) matched]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
