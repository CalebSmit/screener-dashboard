"""Does "Its filings are flagged stale (N days old)" say something true? - 2026-11-18

Read-only, offline. Reads the payload the site is serving (``dashboard_data.js`` and its shards
as committed at HEAD) and, for every stock whose staleness flag is on, sets the sentence the
page prints beside three facts the same payload carries:

- ``flags.annual_instead`` - the fetch read the last fiscal year because the source's quarterly
  statement could not be summed (2026-11-17);
- ``newer_filing``        - the company's own newer 10-Q/10-K in the SEC cache, with its filing
  date (2026-10-15);
- ``asof``                - the period end of each statement actually read.

Usage:  python research/measurements/2026-11-18-stale-sentence-against-the-filings.py [git-ref]
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import check_run_health as h  # noqa: E402


def main(ref: str = "HEAD") -> int:
    D = h.published_payload_whole(ref)
    if D is None:
        print(f"no payload at {ref}")
        return 2
    run_day = date.fromisoformat(D["kpis"]["run_timestamp"][:10])
    sd = D["stock_detail"]
    rank = {r["Ticker"]: r["Rank"] for r in D["table_data"]}

    stale = {t: s for t, s in sd.items() if (s.get("flags") or {}).get("stale_data")}
    annual = {t for t, s in sd.items() if (s.get("flags") or {}).get("annual_instead")}
    said = {t for t, s in sd.items()
            if any("flagged stale" in f.get("t", "") for f in s.get("summary") or [])}

    print(f"run {D['kpis']['run_timestamp'][:10]} ({ref}): {len(sd)} stocks")
    print(f"staleness flag on:                        {len(stale)}")
    print(f"  ...and the summary says 'flagged stale': {len(said & set(stale))}"
          f"   (said without the flag: {len(said - set(stale))})")
    print(f"read from the annual statement instead:   {len(annual)}")
    print(f"  ...of which flagged stale:              {len(annual & set(stale))}")
    print(f"stale but NOT annual-instead:             {sorted(set(stale) - annual)}")
    print(f"annual-instead but NOT stale:             {sorted(annual - set(stale))}")

    newer, recent = [], []
    print("\nticker rank  age  is-as-of    bs-as-of    newer filing (form, period, filed, days ago)")
    for t in sorted(stale, key=lambda x: rank.get(x, 999)):
        s = sd[t]
        fl, asof, nf = s["flags"], s.get("asof") or {}, s.get("newer_filing")
        line = (f"{t:6} {rank.get(t, 0):4} {int(fl.get('stmt_age_days') or 0):4}  "
                f"{asof.get('is', '-'):10}  {asof.get('bs', '-'):10}  ")
        if nf:
            ago = (run_day - date.fromisoformat(nf["filed"])).days
            newer.append(t)
            if ago <= 165:
                recent.append(t)
            line += f"{nf.get('form')}, {nf.get('end')}, {nf.get('filed')}, {ago}"
        else:
            line += "-"
        print(line + ("   [annual instead]" if t in annual else ""))

    print(f"\nflagged stale AND the SEC cache holds a newer filing: {len(newer)} of {len(stale)}")
    print(f"  ...filed within 165 days of the run:                {len(recent)}")
    bs_fresh = [t for t, s in stale.items()
                if (s.get("asof") or {}).get("bs") and
                (run_day - date.fromisoformat(s["asof"]["bs"])).days <= 165]
    print(f"flagged stale while its balance sheet is inside the bar: {len(bs_fresh)} of {len(stale)}")
    top40 = sorted(t for t in stale if rank.get(t, 999) <= 40)
    print(f"flagged stale in the top 40: {top40}")
    return 0


def history(since: str = "2026-11-03") -> int:
    """The same three counts on every page published under the 165-day bar (2026-11-02 night).

    One row per data-run commit. ``newer`` = flagged stale while the SEC cache held a newer
    filing; ``bs ok`` = flagged stale while the balance sheet read was inside the bar."""
    import subprocess
    out = subprocess.run(["git", "-C", str(ROOT), "log", f"--since={since}", "--format=%H %s",
                          "--", "dashboard_data.js"], capture_output=True, text=True).stdout
    print("run date    flagged  newer filing  bs inside bar  top-40")
    seen = set()
    for line in reversed(out.strip().splitlines()):
        sha, subject = line.split(" ", 1)
        if not subject.startswith("data: screener run"):
            continue
        D = h.published_payload_whole(sha)
        if D is None:
            continue
        day = D["kpis"]["run_timestamp"][:10]
        if day in seen:
            continue
        seen.add(day)
        run_day = date.fromisoformat(day)
        rank = {r["Ticker"]: r["Rank"] for r in D["table_data"]}
        stale = {t: s for t, s in D["stock_detail"].items()
                 if (s.get("flags") or {}).get("stale_data")}
        newer = sum(1 for s in stale.values() if s.get("newer_filing"))
        bs_ok = sum(1 for s in stale.values() if (s.get("asof") or {}).get("bs") and
                    (run_day - date.fromisoformat(s["asof"]["bs"])).days <= 165)
        top = sum(1 for t in stale if rank.get(t, 999) <= 40)
        print(f"{day}  {len(stale):7}  {newer:12}  {bs_ok:13}  {top:6}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--history"]:
        raise SystemExit(history(*sys.argv[2:3]))
    raise SystemExit(main(*sys.argv[1:2]))
