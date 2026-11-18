"""Which scored companies has the SEC cache stopped receiving filings for? - 2026-11-18

Offline by default: the published payload (HEAD) and `data/sec/pit/facts.parquet`. Prints the
health line's new count (`sec_crosscheck.no_recent_filing`), each company's newest cached
filing, its sector, whether the filing cross-check could compare anything for it, and the
distribution of "days since the newest cached 10-K/10-Q" across the universe.

`--ask TICKER [TICKER ...]` makes one request per ticker to the SEC's companyfacts API with the
project's own configured identity and prints the newest fact it carries - which is how the gap
was shown to be the SEC's record and not what `refresh_companyfacts` keeps of it.

    python research/measurements/2026-11-18-filings-the-sec-cache-does-not-hold.py
    python research/measurements/2026-11-18-filings-the-sec-cache-does-not-hold.py --ask REG C
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd  # noqa: E402

import check_run_health as H  # noqa: E402
import sec_crosscheck as X  # noqa: E402
import sec_fundamentals as SF  # noqa: E402


def ask(tickers: list[str]) -> None:
    from insider_activity import Edgar, ticker_map, user_agent
    e = Edgar(user_agent())
    cmap = ticker_map(e)
    for t in tickers:
        j = e.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(cmap[t]):010d}.json").json()
        newest_any, newest_kept, since = "", "", Counter()
        for ns, concepts in (j.get("facts") or {}).items():
            for c, body in concepts.items():
                for facts in (body.get("units") or {}).values():
                    for f in facts:
                        filed = f.get("filed") or ""
                        newest_any = max(newest_any, filed)
                        if f.get("form") in X.FORMS:
                            newest_kept = max(newest_kept, filed)
                        if filed >= "2026-06-01":
                            since[f.get("form")] += 1
        print(f"{t} ({j.get('entityName')}): newest fact of any kind filed {newest_any}; newest on a "
              f"10-K/10-Q {newest_kept}; facts filed since 2026-06-01 by form: {dict(since)}")


def main() -> int:
    D = H.published_payload_whole("HEAD")
    facts = pd.read_parquet(SF.FACTS_PATH)
    run = pd.Timestamp(D["kpis"]["run_timestamp"][:10])
    sd = D["stock_detail"]
    rank = {r["Ticker"]: r["Rank"] for r in D["table_data"]}
    blind = X.no_recent_filing(D, facts)
    print(f"run {run.date()}: {len(blind)} of {len(sd)} scored stocks have no 10-K or 10-Q in the SEC "
          f"cache filed in the last {X.NO_RECENT_FILING_DAYS} days\n")
    print("ticker rank  newest cached  days  sector                   compared fields  annual-instead")
    for t in sorted(blind, key=lambda x: blind[x] or ""):
        s = sd[t]
        days = (run - pd.Timestamp(blind[t])).days if blind[t] else "-"
        n_x = sum(1 for v in (s.get("xchk") or {}).values() if isinstance(v, (int, float)))
        print(f"{t:6} {rank.get(t, 0):4}  {blind[t] or 'none':13}  {days!s:>4}  {s.get('sector', ''):24} "
              f"{n_x:15}  {(s.get('flags') or {}).get('annual_instead') or ''}")
    print("\nby sector:", dict(Counter(sd[t].get("sector") for t in blind).most_common()))
    top = sorted(t for t in blind if rank.get(t, 999) <= 40)
    print("in the top 40:", top)

    f = facts[facts["form"].isin(X.FORMS) & (facts["filed"] <= run)]
    age = (run - f.groupby("ticker")["filed"].max()).dt.days
    age = age[[t for t in sd if t in age.index]]
    print("\ndays since the newest cached 10-K/10-Q, all scored stocks:")
    for lo, hi in ((0, 45), (46, 100), (101, 165), (166, 250), (251, 10_000)):
        print(f"  {lo:>3}-{hi if hi < 10_000 else '':<5} {int(((age >= lo) & (age <= hi)).sum()):4}")
    compared = sum(1 for s in sd.values()
                   if any(isinstance(v, (int, float)) for v in (s.get("xchk") or {}).values()))
    print(f"\nfiling cross-check compared at least one field for {compared} of {len(sd)}; "
          f"of the {len(blind)} above, {sum(1 for t in blind if any(isinstance(v, (int, float)) for v in (sd[t].get('xchk') or {}).values()))}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--ask"]:
        ask(sys.argv[2:])
        raise SystemExit(0)
    raise SystemExit(main())
