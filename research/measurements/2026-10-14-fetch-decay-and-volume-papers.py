"""Download the two papers the options-panel sentence corrections need.

(b) needs a post-publication decay figure from the primary source: McLean & Pontiff (2016),
    "Does Academic Research Destroy Stock Return Predictability?", Journal of Finance 71(1).
(c) needs Pan & Poteshman (2006), "The Information in Option Volume for Future Stock Prices",
    Review of Financial Studies 19(3), for what the *publicly observable* (unsigned) put-call
    ratio predicts - the page's put/call open-interest row has no sentence at all.

The 2026-10-12 note cites both second-hand. Per prompts/nightly.md, a figure that goes on a
public page is read from the paper.

Usage: python -I -X utf8 research/measurements/2026-10-14-fetch-decay-and-volume-papers.py
Writes to data/papers/ (gitignored).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/131.0.0.0 Safari/537.36")

CANDIDATES = [
    # McLean & Pontiff (2016) - working-paper and conference copies found by search
    ("mp-2016-a.pdf", "https://tevgeniou.github.io/EquityRiskFactors/bibliography/AcademicReviewFactor.pdf"),
    ("mp-2016-b.pdf", "https://www.hec.ca/finance/Fichier/McLean.pdf"),
    ("mp-2016-c.pdf", "https://www.fmg.ac.uk/sites/default/files/2020-08/Jeffrey-Pontiff.pdf"),
    ("mp-2016-d.pdf", "https://abfer.org/media/abfer-events-2013/annual-conference/corporate-finance/track2-does-academic-research-destroy-stock-return-predictability.pdf"),
    ("mp-2016-e.pdf", "https://www.ivey.uwo.ca/media/2827306/bengraham3rdsymposium-pontiff-paper-2014.pdf"),
    # Pan & Poteshman (2006)
    ("pp-2006-a.pdf", "https://www.mit.edu/~junpan/volume.pdf"),
    ("pp-2006-b.pdf", "https://www.nber.org/system/files/working_papers/w10925/w10925.pdf"),
]

OUT = Path("data/papers")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    got = []
    for name, url in CANDIDATES:
        dest = OUT / name
        subprocess.run(["curl", "-sL", "--max-time", "90", "-A", UA, "-o", str(dest), url],
                       check=False)
        if not dest.exists():
            print(f"  MISS  {name}  (no file)  {url}")
            continue
        head = dest.read_bytes()[:5]
        size = dest.stat().st_size
        if head.startswith(b"%PDF") and size > 100_000:
            print(f"  GOT   {name}  {size:,} bytes  {url}")
            got.append(name)
        else:
            print(f"  MISS  {name}  {size:,} bytes, starts {head!r}  {url}")
            dest.unlink(missing_ok=True)
    print(f"\n{len(got)} usable PDF(s): {got}")
    return 0 if got else 1


if __name__ == "__main__":
    sys.exit(main())
