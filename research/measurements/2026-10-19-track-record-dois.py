"""Resolve every DOI cited in research/2026-10-19-track-record.md through Crossref.

Run: python research/measurements/2026-10-19-track-record-dois.py

Prints, per DOI, the title, the journal, the volume/issue/pages and the year Crossref holds, so
the citation in the note can be checked against the registry rather than against memory. A DOI
that does not resolve prints FAILED and the script exits 1.
"""

from __future__ import annotations

import sys

import requests

DOIS = {
    "McLean & Pontiff 2016": "10.1111/jofi.12365",
    "Lo 2002": "10.2469/faj.v58.n4.2453",
    "Bailey & Lopez de Prado 2012": "10.21314/JOR.2012.255",
    "Fama & French 2010": "10.1111/j.1540-6261.2010.01598.x",
    "Novy-Marx & Velikov 2016": "10.1093/rfs/hhv063",
    "Frazzini, Israel & Moskowitz (SSRN)": "10.2139/ssrn.3229719",
}


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    bad = 0
    for who, doi in DOIS.items():
        try:
            r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=30,
                             headers={"User-Agent": "screener-research (mailto:research@example.invalid)"})
            r.raise_for_status()
            m = r.json()["message"]
            title = (m.get("title") or [""])[0]
            journal = (m.get("container-title") or [""])[0]
            year = (m.get("issued", {}).get("date-parts") or [[None]])[0][0]
            authors = ", ".join(a.get("family", "") for a in m.get("author", []))
            print(f"OK  {who:38s} {doi}\n    {authors} ({year}) {title} | {journal} "
                  f"{m.get('volume', '')}({m.get('issue', '')}) {m.get('page', '')}")
        except Exception as exc:  # noqa: BLE001
            bad += 1
            print(f"FAILED {who} {doi}: {exc}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
