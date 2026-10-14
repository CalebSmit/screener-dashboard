"""Download Xing, Zhang & Zhao (2010) so its SKEW definition can be read from the paper.

The 2026-10-12 note confirmed the definition second-hand (via a survey). The page cites the
paper's ~11%/yr effect size for a measure this screener computes differently, so before the
computation is changed to match, the definition is read from the paper itself.

Usage: python -I -X utf8 research/measurements/2026-10-14-fetch-xzz.py
Writes to data/papers/ (gitignored).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/131.0.0.0 Safari/537.36")

CANDIDATES = [
    # Rice University - Yuhang Xing's own copy
    ("xzz-rice.pdf", "https://www.ruf.rice.edu/~yxing/smirk_jfqa.pdf"),
    ("xzz-rice2.pdf", "https://www.ruf.rice.edu/~yxing/Smirk.pdf"),
    # Author copies / mirrors
    ("xzz-mirror1.pdf", "https://www0.gsb.columbia.edu/mygsb/faculty/research/pubfiles/3221/smirk.pdf"),
    ("xzz-wayback.pdf",
     "https://web.archive.org/web/2018/https://www.ruf.rice.edu/~yxing/smirk_jfqa.pdf"),
]

OUT = Path("data/papers")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    got = []
    for name, url in CANDIDATES:
        dest = OUT / name
        cmd = ["curl", "-sL", "--max-time", "60", "-A", UA, "-o", str(dest), url]
        subprocess.run(cmd, check=False)
        if not dest.exists():
            print(f"  MISS  {name}  (no file)  {url}")
            continue
        head = dest.open("rb").read(5)
        size = dest.stat().st_size
        if head.startswith(b"%PDF"):
            print(f"  PDF   {name}  {size:,} bytes  {url}")
            got.append(dest)
        else:
            print(f"  NOT A PDF  {name}  {size:,} bytes, starts {head!r}  {url}")
            dest.unlink()
    if not got:
        print("\nNo PDF obtained. Say so in the note rather than citing from memory.")
        return 1
    print(f"\n{len(got)} PDF(s) in {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
