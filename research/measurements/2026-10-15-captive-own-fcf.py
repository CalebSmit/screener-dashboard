"""What do the captives themselves publish as free cash flow, and over what scope?

For the review of the open capex proposal. The proposal's authority is that GM, Ford and
Caterpillar all define free cash flow without their finance arms' lease purchases - which is
true, and is only half of each definition: the other half is that the operating cash flow they
start from is the industrial segment's, not the consolidated figure. This reads the companies'
own 8-K earnings releases on sec.gov and prints the sentences around "free cash flow", so the
scope of each definition is read rather than assumed.

    python research/measurements/2026-10-15-captive-own-fcf.py [TICKER ...]
"""
from __future__ import annotations

import re
import sys
from html import unescape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TICKERS = sys.argv[1:] or ["GM", "CAT", "DE", "PCAR"]
KEY = re.compile(r"free cash flow", re.I)


def text_of(html: str) -> str:
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", html)
    t = re.sub(r"(?s)<[^>]+>", " ", t)
    return re.sub(r"\s+", " ", unescape(t))


def main() -> None:
    from insider_activity import Edgar, ticker_map, user_agent
    ua = user_agent()
    if not ua:
        raise SystemExit("no SEC identity configured")
    edgar = Edgar(ua)
    cmap = ticker_map(edgar)
    for t in TICKERS:
        cik = int(cmap[t.upper()])
        sub = edgar.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
        r = sub["filings"]["recent"]
        found = 0
        order = sorted(range(len(r["form"])), key=lambda i: r["filingDate"][i], reverse=True)
        for i in order:
            if r["form"][i] != "8-K" or found:
                continue
            acc = r["accessionNumber"][i].replace("-", "")
            idx = edgar.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/index.json").json()
            for item in idx["directory"]["item"]:
                name = item["name"]
                if not name.lower().endswith((".htm", ".html")) or "ex99" not in name.lower().replace("-", "").replace("_", ""):
                    continue
                body = text_of(edgar.get(f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{name}").text)
                if not KEY.search(body):
                    continue
                print(f"=== {t}  8-K filed {r['filingDate'][i]}  {name}")
                for m in KEY.finditer(body):
                    print("   ..." + body[max(0, m.start() - 320): m.end() + 320].strip() + "...\n")
                found = 1
                break
        if not found:
            print(f"=== {t}: no 8-K exhibit mentioning free cash flow in the recent index")
        print()


if __name__ == "__main__":
    main()
