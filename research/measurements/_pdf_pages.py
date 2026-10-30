"""Print a page range of a downloaded PDF.

Usage:  python -I -X utf8 research/measurements/_pdf_pages.py <pdf> <first> <last> [chars]
Page numbers are 1-based as the PDF viewer shows them.
"""
from __future__ import annotations

import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> int:
    path = Path(sys.argv[1])
    first, last = int(sys.argv[2]), int(sys.argv[3])
    chars = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
    reader = PdfReader(str(path))
    for i in range(first - 1, min(last, len(reader.pages))):
        print(f"--- pdf page {i + 1} ---")
        print((reader.pages[i].extract_text() or "")[:chars])
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
