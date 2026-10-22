"""How many of January 2012's S&P 500 members had XBRL balance sheets before 2012?

Supports research/2026-10-22-free-data-that-adds-evidence.md, candidate "v2 before 2012".
Backtest v2 starts in 2012-01. SEC's XBRL phase-in (Release 33-9002, 2009) began with large
accelerated filers above $5bn of public float for periods ending after 2009-06-15, the rest of the
large accelerated filers a year later, everyone else in 2011. This reads SEC's free `frames` API
(one fact per filer per calendar period) for total assets at quarter-ends 2009-2012 and counts how
many of the 2012-01-31 members (Wikipedia revision, as backtest v2 uses it) appear, by CIK.

Read-only; no returns are read (CLAUDE.md rules 4-5). Downloads go to a cache directory passed on
the command line (default: the session scratchpad), never into the repo. SEC User-Agent read from
the main checkout's gitignored data/sec/user_agent.txt; requests paced well under SEC's 10/s.

    python research/measurements/2026-10-22-xbrl-phase-in-coverage.py [cache_dir]
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
MAIN = Path(r"C:/Users/smitc/OneDrive/Documents/Screener")
DATA = MAIN / "data" if (MAIN / "data" / "sec" / "user_agent.txt").exists() else REPO / "data"
OUT = HERE.with_suffix(".json")
PERIODS = ["CY2009Q2I", "CY2009Q3I", "CY2009Q4I", "CY2010Q1I", "CY2010Q2I", "CY2010Q3I",
           "CY2010Q4I", "CY2011Q2I", "CY2011Q4I", "CY2012Q2I"]


def fetch(period: str, cache: Path, ua: str) -> dict:
    path = cache / f"Assets_{period}.json"
    if not path.exists():
        url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/Assets/USD/{period}.json"
        req = urllib.request.Request(url, headers={"User-Agent": ua})
        path.write_bytes(urllib.request.urlopen(req, timeout=60).read())
        time.sleep(0.25)
    return json.loads(path.read_text())


def main() -> None:
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd() / "xbrl_frames_cache"
    cache.mkdir(parents=True, exist_ok=True)
    ua = (DATA / "sec" / "user_agent.txt").read_text().strip()
    snaps = json.loads((DATA / "universe_history" / "sp500_membership.json").read_text())["snapshots"]
    members = next(s["tickers"] for s in snaps if s["as_of"] == "2012-01-31")
    tick_cik = json.loads((DATA / "universe_history" / "ticker_ciks.json").read_text())["ciks"]
    ciks = {t: tick_cik[t] for t in members if t in tick_cik}
    out = {"members_2012_01": len(members), "members_with_cik": len(ciks), "periods": {}}
    for p in PERIODS:
        frame = fetch(p, cache, ua)
        rows = {int(r["cik"]): r for r in frame["data"]}
        have = [t for t, c in ciks.items() if int(c) in rows]
        # accession numbers carry the filing year in their middle field (0001193125-09-...)
        first_year = int(p[2:6])
        same_or_next = [t for t in have
                        if int(rows[int(ciks[t])]["accn"].split("-")[1]) + 2000 <= first_year + 1]
        out["periods"][p] = {"filers_in_frame": len(rows), "members_present": len(have),
                             "share_of_members_with_cik": round(len(have) / len(ciks), 3),
                             "present_and_filed_within_a_year": len(same_or_next)}
        print(p, out["periods"][p])
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
