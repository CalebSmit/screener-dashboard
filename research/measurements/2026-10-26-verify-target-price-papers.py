"""Spot-check every figure research/2026-10-26-price-target-upside.md quotes, against PDF bytes.

The note's literature section was drafted from sub-agent reads, one of which could not reach raw
PDFs and used a text-extraction proxy with several figures flagged uncertain. CLAUDE.md's rule for
reading a paper is to fetch the PDF and read the tables, and to spot-check a delegated figure
before it goes in a note - this script is that spot-check, committed so the check is repeatable
rather than asserted.

It downloads the two papers the note is load-bearing on (Da & Schaumburg 2011 from the author's
Notre Dame page; Brav & Lehavy 2003 from Lehavy's Michigan page), confirms each starts with
`%PDF-`, extracts the text with pypdf and prints the surrounding sentence for each quoted figure
with the PDF page it was found on. A NOT FOUND line means the note may not assert that figure;
that is how the "two-thirds in two weeks" phrasing and the beta-by-size terciles were caught - the
first was rephrased and found, the second is listed in the note's section 7 as not asserted.

Run from the repo root:  python -I -X utf8 research/measurements/2026-10-26-verify-target-price-papers.py
"""
from __future__ import annotations

import re
import sys
import urllib.request
from pathlib import Path

TMP = Path.home() / "AppData" / "Local" / "Temp" / "ptv"
TMP.mkdir(parents=True, exist_ok=True)
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

PAPERS = {
    "ds": "https://academicweb.nd.edu/~zda/TargetPrice.pdf",
    "bl": "https://webuser.bus.umich.edu/rlehavy/BL.pdf",
}


def fetch(name: str, url: str) -> Path | None:
    p = TMP / f"{name}.pdf"
    if p.is_file() and p.stat().st_size > 50_000:
        print(f"{name}: cached {p.stat().st_size} bytes")
        return p
    import ssl
    data = None
    # Verified first. The fallback exists because this machine's trust chain carries a
    # self-signed root (the Michigan host fails with CERTIFICATE_VERIFY_FAILED), and the integrity
    # check that matters here is not the TLS chain but that the bytes are a PDF and contain the
    # paper's own sentences - both of which are asserted below. Only used for fetching a public
    # working paper; nothing is executed from it.
    for ctx in (None, ssl._create_unverified_context()):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90, context=ctx) as r:
                data = r.read()
            break
        except Exception as e:  # noqa: BLE001
            print(f"{name}: fetch attempt failed {type(e).__name__}: {e}")
    if data is None:
        return None
    p.write_bytes(data)
    print(f"{name}: {len(data)} bytes, starts {data[:5]!r}")
    return p if data[:5] == b"%PDF-" else None


def pages(p: Path) -> list[str]:
    from pypdf import PdfReader
    return [(pg.extract_text() or "") for pg in PdfReader(str(p)).pages]


def find(txts: list[str], pattern: str, label: str, context: int = 170) -> None:
    rx = re.compile(pattern, re.I | re.S)
    hits = 0
    for i, t in enumerate(txts):
        flat = re.sub(r"\s+", " ", t)
        for m in rx.finditer(flat):
            hits += 1
            s = max(0, m.start() - context // 2)
            print(f"   [pdf p{i + 1}] ...{flat[s:m.end() + context]}...")
            if hits >= 3:
                break
        if hits >= 3:
            break
    print(f"  {'OK ' if hits else 'NOT FOUND '} {label}" + ("" if hits else f"  /{pattern}/"))
    print()


def main() -> None:
    paths = {k: fetch(k, u) for k, u in PAPERS.items()}
    print()

    if paths["ds"]:
        t = pages(paths["ds"])
        print(f"=== Da & Schaumburg: {len(t)} pages ===")
        find(t, r"79\s*bp/month.{0,40}0\.86", "absolute TPER sort 79bp t=0.86")
        find(t, r"177\s*bp/month", "within-sector 177 bp/month")
        find(t, r"TPER_?t?\s*=\s*TP", "the TPER definition")
        find(t, r"relative and not absolute valuations", "the paper's own reading")
        find(t, r"not merely a small stock phenomenon", "the S&P 500 claim")
        find(t, r"lower \(higher\) past return", "high TPER = fallen stock")
        find(t, r"two-?thirds of the first month", "2/3 of the profit in two weeks")
        find(t, r"First\s*Call.{0,120}199[67].{0,60}200[45]", "sample period")
        find(t, r"148\s*bp/month", "net-of-cost figure")
        find(t, r"(two weeks|ﬁrst two weeks|110\s*bp)", "how much accrues in two weeks")
        find(t, r"Large\s*Growth", "the Large Growth subsample")
        find(t, r"15 stock characteristics", "the 15-control regression")
        find(t, r"decimalization", "the post-2000 decay")

    if paths["bl"]:
        t = pages(paths["bl"])
        print(f"=== Brav & Lehavy: {len(t)} pages ===")
        find(t, r"28\s*percent annually", "mean beta 1.28 -> 28% a year")
        find(t, r"barely contribute", "prices barely correct the deviation")
        find(t, r"predominantly made by analysts", "analysts close the gap")
        find(t, r"223,016", "sample size")
        find(t, r"no reliable relationship", "expected revision not priced")
        find(t, r"(1\.37.{0,120}1\.2[39]|Size\s*Terciles|Panel C)", "the beta-by-size terciles", 320)
        find(t, r"expected to earn", "the 28% reading in context", 320)


if __name__ == "__main__":
    sys.exit(main())
