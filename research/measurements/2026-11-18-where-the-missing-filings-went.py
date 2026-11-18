"""Where did the filings the SEC cache does not hold go? - 2026-11-18 (13:00 session)

The morning's synthesis found 20 of 499 scored companies with no 10-K or 10-Q in the SEC
`companyfacts` cache filed in the last 165 days, showed the gap was in the SEC's record and not
in our refresh, and did not establish why. This script follows each company's filings:

1. EDGAR's `submissions` record: which 10-K / 10-Q filings the company has made since the
   newest one `companyfacts` carries for it, and whether each is XBRL.
2. Each such filing's SGML header: every registrant (FILER block) on it, in the order filed.
3. `companyfacts` for every *other* registrant on those filings: does one of them carry facts
   from the accession the listed company's record lacks?

Network: about 6-10 requests per company through the project's own polite client. Writes
nothing but its printed output.

    python research/measurements/2026-11-18-where-the-missing-filings-went.py            # the 20
    python research/measurements/2026-11-18-where-the-missing-filings-went.py REG C      # named
    python research/measurements/2026-11-18-where-the-missing-filings-went.py --control 12
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import check_run_health as H  # noqa: E402
import sec_crosscheck as X  # noqa: E402
import sec_fundamentals as SF  # noqa: E402
from insider_activity import Edgar, ticker_map, user_agent  # noqa: E402

FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A")


def facts_by_accession(edgar: Edgar, cik: int) -> tuple[str, dict]:
    """entity name, {accession: [n facts, newest filed, form]} from one companyfacts record."""
    try:
        j = edgar.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cik):010d}.json").json()
    except Exception as e:  # noqa: BLE001 - a registrant with no record is an answer, not an error
        return f"<no companyfacts: {type(e).__name__}>", {}
    out: dict = {}
    for concepts in (j.get("facts") or {}).values():
        for body in concepts.values():
            for facts in (body.get("units") or {}).values():
                for f in facts:
                    a = out.setdefault(f.get("accn"), [0, "", f.get("form")])
                    a[0] += 1
                    a[1] = max(a[1], f.get("filed") or "")
    return str(j.get("entityName")), out


def periodic_filings(edgar: Edgar, cik: int, since: str) -> list[dict]:
    j = edgar.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json").json()
    r = j["filings"]["recent"]
    out = []
    for i, form in enumerate(r["form"]):
        if form in FORMS and r["filingDate"][i] > since:
            out.append({"form": form, "filed": r["filingDate"][i], "accn": r["accessionNumber"][i],
                        "period": r["reportDate"][i], "xbrl": r["isXBRL"][i], "ixbrl": r["isInlineXBRL"][i]})
    return sorted(out, key=lambda d: d["filed"])


def registrants(edgar: Edgar, cik: int, accn: str) -> list[tuple[int, str]]:
    """Every FILER on a filing, in the header's order: [(cik, conformed name)]."""
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accn.replace('-', '')}/{accn}-index-headers.html"
    txt = edgar.get(url).text
    out = []
    for block in re.split(r"&lt;FILER&gt;|<FILER>", txt)[1:]:
        name = re.search(r"CONFORMED[- ]NAME(?:&gt;|>|:)\s*([^\n<&]+)", block)
        c = re.search(r"(?:&lt;CIK&gt;|<CIK>|CENTRAL INDEX KEY:)\s*(\d+)", block)
        if c:
            out.append((int(c.group(1)), (name.group(1).strip() if name else "?")))
    return out


def follow(edgar: Edgar, t: str, cik: int, newest_cached: str | None) -> dict:
    name, own = facts_by_accession(edgar, cik)
    newest_own = max((v[1] for a, v in own.items() if v[2] in FORMS), default="")
    since = newest_cached or newest_own or "2025-12-31"
    filings = periodic_filings(edgar, cik, since)
    print(f"\n{t}  CIK {cik}  {name}: companyfacts' newest 10-K/10-Q fact filed {newest_own or 'none'}; "
          f"{len(filings)} periodic filing(s) on EDGAR since {since}")
    res = {"ticker": t, "filings": len(filings), "missing": 0, "found_under": set(), "first_filer_is_listed": []}
    seen: dict = {}
    for f in filings:
        have = own.get(f["accn"])
        regs = registrants(edgar, cik, f["accn"])
        first = regs[0][0] if regs else None
        res["first_filer_is_listed"].append(first == cik)
        print(f"  {f['form']:6} filed {f['filed']} period {f['period']} {f['accn']} xbrl={f['xbrl']} "
              f"ixbrl={f['ixbrl']}: {have[0] if have else 0} facts in its own record; "
              f"{len(regs)} registrant(s), first {regs[0] if regs else None}")
        if have:
            continue
        res["missing"] += 1
        for rc, rn in regs:
            if rc == cik:
                continue
            if rc not in seen:
                seen[rc] = facts_by_accession(edgar, rc)
            got = seen[rc][1].get(f["accn"])
            if got:
                res["found_under"].add((rc, rn))
                print(f"      -> {got[0]} facts from this accession are in CIK {rc} ({rn})")
        if not any(seen.get(rc, ("", {}))[1].get(f["accn"]) for rc, _ in regs if rc != cik):
            print("      -> in no co-registrant's record either")
    return res


def main(argv: list[str]) -> int:
    D = H.published_payload_whole("HEAD")
    facts = pd.read_parquet(SF.FACTS_PATH)
    blind = X.no_recent_filing(D, facts)
    edgar = Edgar(user_agent())
    cmap = ticker_map(edgar)
    if argv[:1] == ["--control"]:
        # the same walk for companies the cache *does* hold, spread through the ranking: how many
        # registrants do their filings carry, and who is first?
        n = int(argv[1]) if len(argv) > 1 else 12
        ranked = [r["Ticker"] for r in sorted(D["table_data"], key=lambda r: r["Rank"]) if r["Ticker"] not in blind]
        chosen = ranked[:: max(1, len(ranked) // n)][:n]
        newest = facts[facts["form"].isin(FORMS)].groupby("ticker")["filed"].max()
        results = []
        for t in chosen:
            back = str((pd.Timestamp(newest[t]) - pd.Timedelta(days=200)).date())
            results.append(follow(edgar, t, cmap[t], back))
        print(f"\ncontrol: {len(results)} companies the cache holds; filings with no facts in the "
              f"company's own record: {sum(r['missing'] for r in results)} of {sum(r['filings'] for r in results)}")
        return 0
    tickers = [a.upper() for a in argv] or sorted(blind)
    results = [follow(edgar, t, cmap[t], blind.get(t)) for t in tickers]
    print("\n" + "=" * 78)
    tot = sum(r["filings"] for r in results)
    miss = sum(r["missing"] for r in results)
    moved = [r for r in results if r["found_under"]]
    print(f"{len(results)} companies; {tot} periodic filings on EDGAR since each one's newest cached; "
          f"{miss} have no facts in the listed company's own companyfacts record")
    print(f"{len(moved)} companies' missing facts are in a co-registrant's record:")
    for r in moved:
        print(f"  {r['ticker']:5} -> {sorted(r['found_under'])}")
    print("nowhere found:", [r["ticker"] for r in results if r["missing"] and not r["found_under"]])
    print("listed company is the first registrant on every such filing:",
          [r["ticker"] for r in results if r["first_filer_is_listed"] and all(r["first_filer_is_listed"])])
    print(f"({edgar.requests} requests)")
    print(json.dumps({r["ticker"]: sorted(c for c, _ in r["found_under"]) for r in results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
