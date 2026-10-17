"""Download the primary sources behind the market backdrop's sentences, and verify every DOI.

Two jobs, both of which the evidence rule asks for and neither of which should be done from
memory (CLAUDE.md "Your mandate"; prompts/nightly.md "How to read a paper"):

  --pdfs  download each paper to the gitignored data/papers/ and confirm it is really a PDF
  --dois  resolve each DOI through Crossref and print the title/journal/year it returns,
          so a DOI in the note is one that was checked rather than recalled

Usage:  python -I -X utf8 research/measurements/2026-10-17-fetch-macro-sources.py --pdfs --dois
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPERS = ROOT / "data" / "papers"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/129.0.0.0 Safari/537.36")

PDFS = {
    # Sahm's own chapter - the source the jobs sentence cites
    "sahm-2019.pdf": "https://www.hamiltonproject.org/assets/files/Sahm_web_20190506.pdf",
    # Estrella & Mishkin's own working paper behind the REStat article
    "estrella-mishkin-1996.pdf":
        "https://www.newyorkfed.org/medialibrary/media/research/staff_reports/research_papers/9609.pdf",
    # Gilchrist & Zakrajsek, the credit-spread paper (Fed FEDS working-paper version)
    "gilchrist-zakrajsek-2011.pdf":
        "https://www.federalreserve.gov/pubs/feds/2011/201122/201122pap.pdf",
    # The NY Fed research paper scan above carries no text layer; Mishkin's own copy of the
    # published REStat article does.
    "estrella-mishkin-1998-restat.pdf":
        "https://www.newyorkfed.org/medialibrary/media/research/economists/mishkin/pdf/restat98.pdf",
    # Estrella & Trubin (2006) is where the NY Fed states the practical facts the page quotes -
    # the lag range and the false signal.
    "estrella-trubin-2006.pdf":
        "https://www.newyorkfed.org/medialibrary/media/research/current_issues/ci12-5.pdf",
    # Asness, Frazzini & Pedersen, "Quality minus junk" - the flight-to-quality evidence
    "qmj-asness-frazzini-pedersen.pdf":
        "https://www.aqr.com/-/media/AQR/Documents/Insights/Journal-Article/Quality-Minus-Junk.pdf",
}

DOIS = [
    "10.1162/003465398557320",            # Estrella & Mishkin 1998, REStat
    "10.1257/aer.102.4.1692",             # Gilchrist & Zakrajsek 2012, AER
    "10.1007/s11142-018-9470-2",          # Asness, Frazzini & Pedersen 2019, RAS (Quality Minus Junk)
    "10.1016/j.jfineco.2018.03.003",      # Weber 2018, JFE (cash flow duration)
    "10.1023/B:RAST.0000028186.44328.3f",  # Dechow, Sloan & Soliman 2004, RAS (implied equity duration)
    "10.1016/j.jfineco.2015.12.002",      # Daniel & Moskowitz 2016, JFE (momentum crashes)
    "10.1111/jofi.12365",                 # McLean & Pontiff 2016, JF (post-publication decay)
]


def get_pdfs() -> None:
    import requests

    PAPERS.mkdir(parents=True, exist_ok=True)
    for name, url in PDFS.items():
        path = PAPERS / name
        if path.exists() and path.read_bytes()[:4] == b"%PDF":
            print(f"{name:<34} cached, {path.stat().st_size:,} bytes")
            continue
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=120)
            r.raise_for_status()
            if r.content[:4] != b"%PDF":
                print(f"{name:<34} NOT A PDF ({r.headers.get('content-type')}, "
                      f"{len(r.content):,} bytes, starts {r.content[:40]!r})")
                continue
            path.write_bytes(r.content)
            print(f"{name:<34} downloaded, {len(r.content):,} bytes")
        except Exception as e:  # noqa: BLE001 - a failed download is a reportable outcome
            print(f"{name:<34} FAILED: {type(e).__name__}: {str(e)[:120]}")


def check_dois() -> None:
    import requests

    for doi in DOIS:
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}",
                             headers={"User-Agent": "screener-research (caleb.smit@icloud.com)"},
                             timeout=60)
            r.raise_for_status()
            m = r.json()["message"]
            auth = "; ".join(f"{a.get('family','?')}" for a in m.get("author", [])[:4])
            year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
            print(f"\n{doi}\n  OK  {m.get('title', ['?'])[0]}\n      {auth} ({year}) - "
                  f"{(m.get('container-title') or ['?'])[0]} "
                  f"{m.get('volume','')}({m.get('issue','')}) {m.get('page','')}")
        except Exception as e:  # noqa: BLE001
            print(f"\n{doi}\n  FAILED: {type(e).__name__}: {str(e)[:160]}")


#: The four FRED series the two record measurements read, copied into tests/fixtures/fred/ so
#: the tests behind claims context.sahm and context.macro_curve can **recompute the record** the
#: page's sentences state rather than compare them with a stored summary of it. A test that reads
#: a summary checks the plumbing; the lesson from plan/context-layer.md item 2 is that this is not
#: a check at all. Trimmed to 1959 onward (the Sahm series' own start) to keep them small.
FIXTURE_SERIES = ("SAHMREALTIME", "USREC", "GS10", "TB3MS")
FIXTURE_FROM = "1959-01-01"


def write_fixtures() -> None:
    src = ROOT / "data" / "measurements"
    dst = ROOT / "tests" / "fixtures" / "fred"
    dst.mkdir(parents=True, exist_ok=True)
    for sid in FIXTURE_SERIES:
        path = src / f"fred-{sid}-full.csv"
        if not path.exists():
            print(f"{sid:<14} MISSING - run the two record scripts first")
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        kept = [lines[0]] + [ln for ln in lines[1:] if ln[:10] >= FIXTURE_FROM]
        out = dst / f"{sid}.csv"
        out.write_text("\n".join(kept) + "\n", encoding="utf-8")
        print(f"{sid:<14} {len(kept) - 1} rows, {out.stat().st_size:,} bytes -> {out}")


if __name__ == "__main__":
    if "--pdfs" in sys.argv:
        get_pdfs()
    if "--dois" in sys.argv:
        check_dois()
    if "--fixtures" in sys.argv:
        write_fixtures()
    if len(sys.argv) == 1:
        print(__doc__)
